import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_rotate,
)

from envs.components.LowLevelController import LowLevelController
from envs.components.ballWorld import BallWorld


class TrapBallEnv:
    def __init__(self, cfg, target_xy):
        self.controller = LowLevelController(cfg)
        self.target_xy = target_xy
        self.ball_world = BallWorld(self.controller, default_z=0.12)
        self._init_buffers()

    def _init_buffers(self):
        cfg = self.controller.cfg
        dev = self.controller.device

        self.num_obs = cfg["env"]["num_observations"]
        self.num_privileged_obs = cfg["env"]["num_privileged_obs"]
        self.num_actions = cfg["env"]["num_actions"]
        self.dt = cfg["control"]["decimation"] * cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(1, self.num_obs, dtype=torch.float, device=dev)
        self.rew_buf = torch.zeros(0, dtype=torch.float, device=dev)
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=dev)
        self.episode_length_buf = torch.zeros(1, device=dev, dtype=torch.long)
        self.time_out_buf = torch.zeros(1, device=dev, dtype=torch.bool)
        self.extras = {"rew_terms": {}}

        actor_root_state = self.controller.gym.acquire_actor_root_state_tensor(self.controller.sim)
        dof_state_tensor = self.controller.gym.acquire_dof_state_tensor(self.controller.sim)
        body_state = self.controller.gym.acquire_rigid_body_state_tensor(self.controller.sim)

        self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
        self.controller.gym.refresh_actor_root_state_tensor(self.controller.sim)
        self.controller.gym.refresh_dof_force_tensor(self.controller.sim)
        self.controller.gym.refresh_rigid_body_state_tensor(self.controller.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states_robot = self.root_states[0:1, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(1, self.controller.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(1, self.controller.num_dofs, 2)[..., 1]
        self.body_states = gymtorch.wrap_tensor(body_state).view(
            1, self.controller.num_bodies_robot + self.controller.addtional_rigid_num, 13
        )
        self.base_pos = self.root_states_robot[:, 0:3]
        self.base_quat = self.root_states_robot[:, 3:7]
        self.feet_pos = self.body_states[:, self.controller.feet_indices, 0:3]

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.controller.up_axis_idx), device=dev).repeat((1, 1))
        self.actions = torch.zeros(1, self.num_actions, dtype=torch.float, device=dev)
        self.last_actions = torch.zeros(1, self.num_actions, dtype=torch.float, device=dev)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states_robot[:, 7:13])
        self.last_dof_targets = torch.zeros(1, self.controller.num_dofs, dtype=torch.float, device=dev)

        self.delay_steps = torch.zeros(1, dtype=torch.long, device=dev)
        self.torques = torch.zeros(1, self.controller.num_dofs, dtype=torch.float, device=dev)

        self.commands = torch.zeros(1, cfg["commands"]["num_commands"], dtype=torch.float, device=dev)
        self.cmd_resample_time = torch.zeros(1, dtype=torch.long, device=dev)
        self.gait_frequency = torch.zeros(1, dtype=torch.float, device=dev)
        self.gait_process = torch.zeros(1, dtype=torch.float, device=dev)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()

        self.dof_pos_ref = torch.zeros(1, self.controller.num_dofs, dtype=torch.float, device=dev)
        self.default_dof_pos = torch.zeros(1, self.controller.num_dofs, dtype=torch.float, device=dev)
        for i in range(self.controller.num_dofs):
            found = False
            for name in cfg["init_state"]["default_joint_angles"].keys():
                if name in self.controller.dof_names[i]:
                    self.default_dof_pos[:, i] = cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = cfg["init_state"]["default_joint_angles"]["default"]

        self._episode_init_dist = 0.0
        self._reset_task_state()

    def _reset_task_state(self):
        device = self.controller.device
        dtype = self.base_pos.dtype
        self._has_touched_ball = False
        self._init_robot_xy = torch.zeros(2, device=device, dtype=dtype)
        self._defense_forward = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        self._defense_left = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        self._episode_init_dist = 0.0

    def _world_robot_velocity(self):
        v_body = self.base_lin_vel[0, :3]
        return quat_rotate(self.base_quat[0:1], v_body[None, :]).squeeze(0)

    def _capture_initial_line(self):
        device = self.controller.device
        dtype = self.base_pos.dtype
        self._init_robot_xy = self.base_pos[0, :2].clone()
        forward_world = quat_rotate(
            self.base_quat[0:1],
            torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype),
        ).squeeze(0)[:2]
        norm = torch.norm(forward_world)
        if norm < 1e-6:
            self._defense_forward = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        else:
            self._defense_forward = forward_world / norm
        self._defense_left = torch.stack(
            (-self._defense_forward[1], self._defense_forward[0])
        ).to(device=device, dtype=dtype)

    def _set_ball_incoming(self):
        base_x = float(self.base_pos[0, 0].item())
        base_y = float(self.base_pos[0, 1].item())
        z = self.ball_world.default_z

        spawn_dist = np.random.uniform(2.5, 4.0)
        lateral = np.random.uniform(-0.8, 0.8)
        speed = np.random.uniform(1.0, 2.4)

        forward = self._defense_forward.detach().cpu().numpy()
        left = self._defense_left.detach().cpu().numpy()
        ball_xy = np.array([base_x, base_y]) + forward * spawn_dist + left * lateral

        target_lateral = np.random.uniform(-0.25, 0.25)
        target_xy = np.array([base_x, base_y]) + left * target_lateral
        vel_dir = target_xy - ball_xy
        vel_norm = np.linalg.norm(vel_dir)
        if vel_norm < 1e-6:
            vel_dir = -forward
        else:
            vel_dir = vel_dir / vel_norm
        vel_xy = vel_dir * speed

        self.root_states[1, 0] = float(ball_xy[0])
        self.root_states[1, 1] = float(ball_xy[1])
        self.root_states[1, 2] = float(z)
        self.root_states[1, 3:7] = 0.0
        self.root_states[1, 6] = 1.0
        self.root_states[1, 7] = float(vel_xy[0])
        self.root_states[1, 8] = float(vel_xy[1])
        self.root_states[1, 9:13] = 0.0
        self.controller.gym.set_actor_root_state_tensor(
            self.controller.sim, gymtorch.unwrap_tensor(self.root_states)
        )

        self._episode_init_dist = float(np.linalg.norm(ball_xy - np.array([base_x, base_y])))

    def _trap_geometry(self):
        device = self.controller.device
        dtype = self.base_pos.dtype
        robot_xy = self.base_pos[0, :2]
        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]
        ball_vel_xy = ball_lin_vel[:2]
        ball_speed = torch.norm(ball_vel_xy)
        ball_dir = ball_vel_xy / (ball_speed + 1e-6)

        signed_line_dist = torch.dot(ball_xy - self._init_robot_xy, self._defense_forward)
        lateral_offset = torch.dot(ball_xy - self._init_robot_xy, self._defense_left)
        robot_to_ball = ball_xy - robot_xy
        robot_to_ball_dist = torch.norm(robot_to_ball) + 1e-6
        dir_rb = robot_to_ball / robot_to_ball_dist

        approach_speed = -torch.dot(ball_vel_xy, self._defense_forward)
        crossing_time = torch.tensor(float("inf"), device=device, dtype=dtype)
        if approach_speed > 1e-5:
            crossing_time = torch.clamp(signed_line_dist / (approach_speed + 1e-6), min=0.0)

        t_proj = torch.clamp(torch.dot(robot_xy - ball_xy, ball_dir), min=0.0)
        if torch.isfinite(crossing_time):
            t_max = torch.clamp(ball_speed * crossing_time - 0.15, min=0.0)
            t_proj = torch.clamp(t_proj, max=t_max)
        intercept_xy = ball_xy + ball_dir * t_proj
        if ball_speed < 0.1:
            intercept_xy = ball_xy.clone()

        self.target_xy = intercept_xy.detach().clone()
        intercept_offset = intercept_xy - robot_xy
        intercept_dist = torch.norm(intercept_offset) + 1e-6
        intercept_dir = intercept_offset / intercept_dist

        return {
            "robot_xy": robot_xy,
            "ball_xy": ball_xy,
            "ball_vel_xy": ball_vel_xy,
            "ball_speed": ball_speed,
            "ball_dir": ball_dir,
            "signed_line_dist": signed_line_dist,
            "lateral_offset": lateral_offset,
            "robot_to_ball": robot_to_ball,
            "robot_to_ball_dist": robot_to_ball_dist,
            "dir_rb": dir_rb,
            "intercept_xy": intercept_xy,
            "intercept_dist": intercept_dist,
            "intercept_dir": intercept_dir,
            "approach_speed": approach_speed,
            "crossing_time": crossing_time,
        }

    def reset(self):
        obs, infos = self.controller.reset(
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.dof_state,
            self.root_states_robot, self.root_states,
            self.last_dof_targets, self.last_root_vel,
            self.episode_length_buf, self.filtered_lin_vel, self.filtered_ang_vel,
            self.cmd_resample_time, self.delay_steps, self.time_out_buf,
            self.extras, self.commands, self.gait_frequency, self.dt,
            self.projected_gravity, self.base_ang_vel, self.gait_process, self.actions
        )

        self.reset_buf[:] = False
        self.time_out_buf[:] = False
        self._reset_task_state()
        self._capture_initial_line()
        self._set_ball_incoming()
        self.extras = {"rew_terms": {}, "success": False, "fail": False, "fall": False, "hit": False}
        return obs, infos

    def pre_step(self, actions):
        self.actions[:] = torch.clip(
            actions,
            -self.controller.cfg["normalization"]["clip_actions"],
            self.controller.cfg["normalization"]["clip_actions"],
        )
        dof_targets = self.default_dof_pos + self.controller.cfg["control"]["action_scale"] * self.actions
        return dof_targets

    def physics_step(self, dof_targets):
        self.torques.zero_()
        for i in range(self.controller.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = (
                self.controller.dof_stiffness * (self.last_dof_targets - self.dof_pos)
                - self.controller.dof_damping * self.dof_vel
            )
            friction = torch.min(self.controller.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(
                dof_torques - friction,
                min=-self.controller.torque_limits,
                max=self.controller.torque_limits,
            )
            self.torques += dof_torques
            self.controller.gym.set_dof_actuation_force_tensor(
                self.controller.sim, gymtorch.unwrap_tensor(dof_torques)
            )
            self.controller.gym.simulate(self.controller.sim)
            if self.controller.device == "cpu":
                self.controller.gym.fetch_results(self.controller.sim, True)
            self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
            self.controller.gym.refresh_dof_force_tensor(self.controller.sim)
        self.torques /= self.controller.cfg["control"]["decimation"]

        if getattr(self.controller, "viewer", None) is not None:
            self.controller.render()

    def post_step(self):
        self.controller.gym.refresh_actor_root_state_tensor(self.controller.sim)
        self.controller.gym.refresh_rigid_body_state_tensor(self.controller.sim)
        self.base_pos[:] = self.root_states_robot[:, 0:3]
        self.base_quat[:] = self.root_states_robot[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        w = self.controller.cfg["normalization"]["filter_weight"]
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * w + self.filtered_lin_vel[:] * (1.0 - w)
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * w + self.filtered_ang_vel[:] * (1.0 - w)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        fall_now = self._is_fallen()
        self.extras["fall"] = bool(fall_now)
        if fall_now:
            self.reset_buf[:] = True

        self.controller._compute_observations(
            self.projected_gravity, self.base_ang_vel, self.commands,
            self.gait_frequency, self.gait_process,
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.actions
        )

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states_robot[:, 7:13]

    def step(self, actions):
        dof_targets = self.pre_step(actions)
        self.physics_step(dof_targets)
        self.post_step()

        obs = self.controller.obs_buf
        reward = self.compute_midlevel_reward()
        done = self.reset_buf
        info = self.extras
        return obs, reward, done, info

    def _is_fallen(self):
        base_z = float(self.base_pos[0, 2])
        low_z = base_z < 0.2

        device = self.controller.device
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        up_body = quat_rotate(self.base_quat[0:1], world_up[None, :]).squeeze(0)
        tilt = torch.arccos(torch.clamp(up_body[2], -1.0, 1.0))
        return low_z or (tilt > 0.75)

    def get_high_level_action_space(self):
        cmd_cfg = self.controller.cfg["commands"]
        gait_freq = 0.5 * (cmd_cfg["gait_frequency"][0] + cmd_cfg["gait_frequency"][1])
        return [0.0, 0.0, 0.0, gait_freq]

    def apply_high_level_command(self, cmd, smooth=None):
        device = self.controller.device
        new_cmd = torch.tensor(cmd[:3], device=device, dtype=self.commands.dtype).view(1, 3)
        if smooth is None:
            self.commands[:, :3] = new_cmd
            self.gait_frequency[:] = float(cmd[3])
        else:
            alpha = float(smooth)
            self.commands[:, :3] = alpha * self.commands[:, :3] + (1 - alpha) * new_cmd
            self.gait_frequency[:] = alpha * self.gait_frequency + (1 - alpha) * float(cmd[3])

    def is_feet_contact_ball(self):
        foot_positions = self.body_states[:, self.controller.feet_indices, 0:3]
        ball_position = self.body_states[:, self.controller.num_bodies_robot, 0:3]
        distances = torch.norm(foot_positions - ball_position, dim=2)
        return bool(torch.any(distances < 0.25))

    def compute_midlevel_reward(self):
        device = self.controller.device
        dtype = self.base_pos.dtype
        geom = self._trap_geometry()

        v_world_xy = self._world_robot_velocity()[:2]
        robot_speed = torch.norm(v_world_xy)
        move_thresh = torch.tensor(0.1, device=device, dtype=dtype)
        if robot_speed > move_thresh:
            v_dir = v_world_xy / (robot_speed + 1e-6)
            approach_cos = torch.clamp(torch.dot(v_dir, geom["dir_rb"]), -1.0, 1.0)
            intercept_cos = torch.clamp(torch.dot(v_dir, geom["intercept_dir"]), -1.0, 1.0)
        else:
            v_dir = torch.zeros(2, device=device, dtype=dtype)
            approach_cos = torch.zeros((), device=device, dtype=dtype)
            intercept_cos = torch.zeros((), device=device, dtype=dtype)

        touch_now = self.is_feet_contact_ball()
        first_touch = (not self._has_touched_ball) and touch_now and (geom["signed_line_dist"] >= -0.05)
        if touch_now:
            self._has_touched_ball = True

        crossed = bool((geom["signed_line_dist"] < -0.05).item())
        success = bool(first_touch)
        fail = bool((crossed and (not self._has_touched_ball)) or self.extras.get("fall", False))

        r_contact = torch.tensor(60.0 if success else 0.0, device=device, dtype=dtype)
        r_fail = torch.tensor(80.0 if fail and (not self.extras.get("fall", False)) else 0.0, device=device, dtype=dtype)
        r_near_ball = 1.2 * torch.exp(-geom["robot_to_ball_dist"] / 1.2)
        r_intercept = 1.0 * torch.exp(-geom["intercept_dist"] / 0.8)
        r_approach = 0.5 * (approach_cos + 1.0)
        r_path = 0.5 * (intercept_cos + 1.0)
        line_penalty = 0.25 * torch.relu(-geom["signed_line_dist"])
        lateral_penalty = 0.10 * torch.abs(geom["lateral_offset"])
        time_penalty = torch.tensor(0.01, device=device, dtype=dtype)
        fall_penalty = torch.tensor(15.0 if self.extras.get("fall", False) else 0.0, device=device, dtype=dtype)

        reward = (
            r_near_ball
            + r_intercept
            + 0.4 * r_approach
            + 0.6 * r_path
            + r_contact
            - r_fail
            - line_penalty
            - lateral_penalty
            - time_penalty
            - fall_penalty
        )

        self.extras["success"] = success
        self.extras["fail"] = bool(fail and (not self.extras.get("fall", False)))
        self.extras["hit"] = bool(first_touch)

        if success or fail:
            self.reset_buf[:] = True

        terms = {
            "ball_speed": geom["ball_speed"].detach(),
            "robot_speed": robot_speed.detach(),
            "robot_to_ball_dist": geom["robot_to_ball_dist"].detach(),
            "intercept_dist": geom["intercept_dist"].detach(),
            "signed_line_dist": geom["signed_line_dist"].detach(),
            "lateral_offset": geom["lateral_offset"].detach(),
            "approach_speed": geom["approach_speed"].detach(),
            "approach_cos": approach_cos.detach(),
            "intercept_cos": intercept_cos.detach(),
            "r_near_ball": r_near_ball.detach(),
            "r_intercept": r_intercept.detach(),
            "r_approach": (0.4 * r_approach).detach(),
            "r_path": (0.6 * r_path).detach(),
            "r_contact": r_contact.detach(),
            "fail_penalty": (-r_fail).detach(),
            "line_penalty": (-line_penalty).detach(),
            "lateral_penalty": (-lateral_penalty).detach(),
            "time_penalty": (-time_penalty).detach(),
            "fall_penalty": (-fall_penalty).detach(),
            "touch_now": torch.tensor(1.0 if touch_now else 0.0, device=device, dtype=dtype),
            "first_touch": torch.tensor(1.0 if first_touch else 0.0, device=device, dtype=dtype),
            "crossed_line": torch.tensor(1.0 if crossed else 0.0, device=device, dtype=dtype),
        }
        self.extras["rew_terms"] = terms
        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        device = self.base_pos.device
        dtype = self.base_pos.dtype
        geom = self._trap_geometry()
        v_world = self._world_robot_velocity()

        delta_world = torch.zeros(3, device=device, dtype=dtype)
        delta_world[:2] = geom["ball_xy"] - self.base_pos[0, :2]
        delta_body = quat_rotate_inverse(self.base_quat[0:1], delta_world[None, :]).squeeze(0)
        rel_ball_x = delta_body[0]
        rel_ball_y = delta_body[1]
        rel_ball_ang = torch.atan2(rel_ball_y, rel_ball_x + 1e-6)

        obs = torch.stack(
            [
                self.base_pos[0, 0],
                self.base_pos[0, 1],
                geom["ball_xy"][0],
                geom["ball_xy"][1],
                v_world[0],
                v_world[1],
                self.target_xy[0],
                self.target_xy[1],
                rel_ball_x,
                rel_ball_y,
                rel_ball_ang,
                geom["signed_line_dist"],
            ],
            dim=0,
        ).to(device=device, dtype=dtype)
        return obs.view(1, -1)

    def get_initial_dist_xy(self):
        return float(self._episode_init_dist)

    def on_episode_end(self, success: bool, episode_idx: int):
        info = {"reason": "trap curriculum reserved"}
        return 0.0, 0.0, False, info
