import torch
import numpy as np
from collections import deque
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_rotate,
)
from typing import Dict, List, Tuple

from envs.components.LowLevelController import LowLevelController
from envs.components.ballWorld import BallWorld


class DribbleBallEnv:
    def __init__(self, cfg, target_xy):
        self.controller = LowLevelController(cfg)
        self.target_xy = target_xy
        self.ball_world = BallWorld(self.controller, default_z=0.12)
        self._init_buffers()
        self._init_dribble_curriculum()

    @staticmethod
    def _ensure_stage_weights(values, default_values: List[float]) -> List[float]:
        if not isinstance(values, (list, tuple)):
            return list(default_values)
        vals = [float(x) for x in values]
        if len(vals) < len(default_values):
            vals = vals + list(default_values[len(vals):])
        return vals[: len(default_values)]

    def _stage_value(self, values: List[float]) -> float:
        if not values:
            return 0.0
        idx = int(np.clip(self.reward_stage, 0, len(values) - 1))
        return float(values[idx])

    def _init_dribble_curriculum(self):
        curr_cfg = self.controller.cfg.get("curriculum", {})

        self.reward_stage_enabled = bool(curr_cfg.get("dribble_stage_enabled", True))
        self._num_reward_stages = 3
        self.reward_stage = int(curr_cfg.get("dribble_stage_initial", 0))
        self.reward_stage = int(np.clip(self.reward_stage, 0, self._num_reward_stages - 1))

        self._stage_up_thresh = float(curr_cfg.get("dribble_stage_up_thresh", 0.65))
        self._stage_down_thresh = float(curr_cfg.get("dribble_stage_down_thresh", 0.30))
        self._stage_min_eps = int(curr_cfg.get("dribble_stage_min_episodes", 60))
        self._stage_cooldown = int(curr_cfg.get("dribble_stage_cooldown", 12))
        self._stage_success_hist = deque(maxlen=max(20, int(curr_cfg.get("dribble_stage_success_window", 100))))
        self._stage_episodes = 0
        self._stage_successes = 0.0
        self._stage_last_change_ep = -10**9
        self._control_streak_target = max(1, int(curr_cfg.get("dribble_control_streak_target", 24)))

        self._w_control = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_control", [1.20, 0.95, 0.75]),
            [1.20, 0.95, 0.75],
        )
        self._w_turn = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_turn", [0.10, 1.10, 0.90]),
            [0.10, 1.10, 0.90],
        )
        self._w_drive = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_drive", [0.00, 0.35, 1.20]),
            [0.00, 0.35, 1.20],
        )
        self._w_soft_pass = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_soft_pass", [0.85, 1.15, 0.90]),
            [0.85, 1.15, 0.90],
        )
        self._w_reposition = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_reposition", [0.20, 0.90, 1.20]),
            [0.20, 0.90, 1.20],
        )
        self._w_ball_progress = self._ensure_stage_weights(
            curr_cfg.get("dribble_w_ball_progress", [0.45, 0.65, 0.85]),
            [0.45, 0.65, 0.85],
        )

        self._control_dist = float(curr_cfg.get("dribble_control_dist", 0.55))
        self._control_front_min = float(curr_cfg.get("dribble_control_front_min", 0.04))
        self._control_lateral_max = float(curr_cfg.get("dribble_control_lateral_max", 0.30))
        self._success_target_radius = float(curr_cfg.get("dribble_success_target_radius", 0.45))

        self._heading_drive_gate_cos = float(curr_cfg.get("dribble_heading_gate_cos", 0.70))
        self._pass_speed_ref = float(curr_cfg.get("dribble_pass_speed_ref", 0.85))
        self._pass_speed_sigma = max(1e-4, float(curr_cfg.get("dribble_pass_speed_sigma", 0.30)))
        self._pass_reposition_min_toward = float(curr_cfg.get("dribble_pass_reposition_min_toward", 0.12))
        self._reposition_front_ref = float(curr_cfg.get("dribble_reposition_front_ref", 0.20))
        self._reposition_front_sigma = max(1e-4, float(curr_cfg.get("dribble_reposition_front_sigma", 0.16)))
        self._reposition_lateral_sigma = max(1e-4, float(curr_cfg.get("dribble_reposition_lateral_sigma", 0.22)))

        self._control_hold_steps = int(curr_cfg.get("dribble_control_hold_steps", 6))
        self._success_target_hold_steps = int(
            curr_cfg.get(
                "dribble_success_target_hold_steps",
                curr_cfg.get("dribble_success_hold_steps", 8),
            )
        )

        self._spin_yaw_thr = float(curr_cfg.get("dribble_spin_yaw_thr", 1.2))
        self._ball_speed_soft = float(curr_cfg.get("dribble_ball_speed_soft", 1.80))
        self._ball_rel_speed_soft = float(curr_cfg.get("dribble_ball_rel_speed_soft", 1.40))
        self._lost_ball_soft = float(curr_cfg.get("dribble_lost_ball_soft", 0.95))
        self._fail_ball_dist = float(curr_cfg.get("dribble_fail_ball_dist", 1.60))

        self._time_penalty = float(curr_cfg.get("dribble_time_penalty", 0.010))
        self._spin_penalty_scale = float(curr_cfg.get("dribble_spin_penalty_scale", 0.25))
        self._ball_speed_penalty_scale = float(curr_cfg.get("dribble_ball_speed_penalty_scale", 0.18))
        self._rel_speed_penalty_scale = float(curr_cfg.get("dribble_rel_speed_penalty_scale", 0.12))
        self._lost_ball_penalty_scale = float(curr_cfg.get("dribble_lost_ball_penalty_scale", 0.75))
        self._fall_penalty = float(curr_cfg.get("dribble_fall_penalty", 18.0))
        self._fail_penalty = float(curr_cfg.get("dribble_fail_penalty", 6.0))
        self._success_bonus = float(curr_cfg.get("dribble_success_bonus", 8.0))

        base_tmin = float(curr_cfg.get("dribble_target_dist_min", 2.0))
        base_tmax = float(curr_cfg.get("dribble_target_dist_max", 3.0))
        base_ang_min = float(curr_cfg.get("dribble_target_angle_abs_min_deg", 35.0))
        base_ang_max = float(curr_cfg.get("dribble_target_angle_abs_max_deg", 150.0))
        base_spawn_min = float(curr_cfg.get("dribble_ball_spawn_dist_min", 0.25))
        base_spawn_max = float(curr_cfg.get("dribble_ball_spawn_dist_max", 0.42))
        base_spawn_lat = abs(float(curr_cfg.get("dribble_ball_spawn_lateral_abs", 0.12)))

        self._target_dist_min_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_target_dist_min_stages", [base_tmin, base_tmin + 0.4, base_tmin + 0.8]),
            [base_tmin, base_tmin + 0.4, base_tmin + 0.8],
        )
        self._target_dist_max_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_target_dist_max_stages", [base_tmax, base_tmax + 0.5, base_tmax + 1.0]),
            [base_tmax, base_tmax + 0.5, base_tmax + 1.0],
        )
        self._target_angle_abs_min_deg_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_target_angle_abs_min_deg_stages", [max(0.0, base_ang_min - 15.0), base_ang_min, min(175.0, base_ang_min + 20.0)]),
            [max(0.0, base_ang_min - 15.0), base_ang_min, min(175.0, base_ang_min + 20.0)],
        )
        self._target_angle_abs_max_deg_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_target_angle_abs_max_deg_stages", [max(30.0, base_ang_max - 40.0), base_ang_max, min(179.0, base_ang_max + 20.0)]),
            [max(30.0, base_ang_max - 40.0), base_ang_max, min(179.0, base_ang_max + 20.0)],
        )
        self._ball_spawn_dist_min_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_ball_spawn_dist_min_stages", [max(0.05, base_spawn_min - 0.04), base_spawn_min, base_spawn_min + 0.05]),
            [max(0.05, base_spawn_min - 0.04), base_spawn_min, base_spawn_min + 0.05],
        )
        self._ball_spawn_dist_max_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_ball_spawn_dist_max_stages", [max(0.08, base_spawn_max - 0.05), base_spawn_max, base_spawn_max + 0.08]),
            [max(0.08, base_spawn_max - 0.05), base_spawn_max, base_spawn_max + 0.08],
        )
        self._ball_spawn_lateral_abs_stages = self._ensure_stage_weights(
            curr_cfg.get("dribble_ball_spawn_lateral_abs_stages", [max(0.02, base_spawn_lat - 0.05), base_spawn_lat, base_spawn_lat + 0.08]),
            [max(0.02, base_spawn_lat - 0.05), base_spawn_lat, base_spawn_lat + 0.08],
        )

        self._max_episode_steps = int(self.controller.cfg["rewards"]["episode_length_s"] / self.dt)

        self.cur_r_min = 0.0
        self.cur_r_max = float(self.reward_stage + 1)
        self._episode_target_angle_deg = 0.0
        self._episode_target_dist = 0.0
        self._episode_ball_spawn_dist = 0.0
        self._episode_ball_spawn_lateral = 0.0

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
        self.controller.root_states = self.root_states
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

        self._reset_task_state()

    def _reset_task_state(self):
        self.initial_dist_xy = 0.0
        self._prev_heading_abs = None
        self._control_hold_counter = 0
        self._success_hold_counter = 0
        self._episode_target_angle_deg = 0.0
        self._episode_target_dist = 0.0
        self._episode_ball_spawn_dist = 0.0
        self._episode_ball_spawn_lateral = 0.0
        self._prev_ball_to_target_dist = None
        self._episode_max_control_streak = 0
        self.extras["success"] = False
        self.extras["fail"] = False

    def _forward_left_world(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.controller.device
        dtype = self.base_pos.dtype
        fwd_local = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        fwd_world = quat_rotate(self.base_quat[0:1], fwd_local).squeeze(0)[:2]
        fwd_norm = torch.norm(fwd_world)
        if float(fwd_norm.item()) < 1e-6:
            fwd_xy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        else:
            fwd_xy = fwd_world / fwd_norm
        left_xy = torch.stack((-fwd_xy[1], fwd_xy[0]))
        return fwd_xy, left_xy

    def _sample_new_target(self):
        base_xy = self.base_pos[0, :2]
        fwd_xy, left_xy = self._forward_left_world()

        dist_min = self._stage_value(self._target_dist_min_stages)
        dist_max = self._stage_value(self._target_dist_max_stages)
        dist_min = float(min(dist_min, dist_max))
        dist_max = float(max(dist_min, dist_max))
        target_dist = np.random.uniform(dist_min, dist_max)

        angle_min = float(np.clip(self._stage_value(self._target_angle_abs_min_deg_stages), 0.0, 179.0))
        angle_max = float(np.clip(self._stage_value(self._target_angle_abs_max_deg_stages), 0.0, 179.0))
        if angle_max < angle_min:
            angle_min, angle_max = angle_max, angle_min
        angle_abs_deg = float(np.random.uniform(angle_min, angle_max))
        sign = -1.0 if np.random.rand() < 0.5 else 1.0
        angle_deg = sign * angle_abs_deg
        angle_rad = float(np.deg2rad(angle_deg))

        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        dir_xy = c * fwd_xy + s * left_xy
        dir_xy = dir_xy / (torch.norm(dir_xy) + 1e-6)

        target_xy = base_xy + float(target_dist) * dir_xy
        self.target_xy = target_xy.detach().clone()
        self._episode_target_angle_deg = float(angle_deg)
        self._episode_target_dist = float(target_dist)

    def _place_ball_for_dribble(self):
        base_xy = self.base_pos[0, :2]
        fwd_xy, left_xy = self._forward_left_world()

        dist_min = self._stage_value(self._ball_spawn_dist_min_stages)
        dist_max = self._stage_value(self._ball_spawn_dist_max_stages)
        dist_min = float(min(dist_min, dist_max))
        dist_max = float(max(dist_min, dist_max))
        lateral_abs = abs(self._stage_value(self._ball_spawn_lateral_abs_stages))
        spawn_dist = float(np.random.uniform(dist_min, dist_max))
        lateral = float(np.random.uniform(-lateral_abs, lateral_abs))

        ball_xy = base_xy + spawn_dist * fwd_xy + lateral * left_xy
        self._episode_ball_spawn_dist = float(spawn_dist)
        self._episode_ball_spawn_lateral = float(lateral)
        self.ball_world.set_pose(
            self.root_states,
            (float(ball_xy[0].item()), float(ball_xy[1].item()), float(self.ball_world.default_z)),
            zero_velocity=True,
        )

    def _world_robot_velocity(self):
        v_body = self.base_lin_vel[0, :3]
        return quat_rotate(self.base_quat[0:1], v_body[None, :]).squeeze(0)

    def _current_heading_metrics(self):
        robot_xy = self.base_pos[0, :2]
        target_delta = self.target_xy - robot_xy
        target_dist = torch.norm(target_delta) + 1e-6
        to_target = target_delta / target_dist
        fwd_xy, _ = self._forward_left_world()
        heading_cos = torch.clamp(torch.dot(fwd_xy, to_target), -1.0, 1.0)
        heading_abs = torch.acos(heading_cos)
        return heading_abs, heading_cos, target_dist, to_target

    def on_episode_end(self, success: bool, episode_idx: int):
        # 课程指标改为：本回合“最大连续控球步数”
        max_streak = float(self._episode_max_control_streak)
        streak_score = float(np.clip(max_streak / float(self._control_streak_target), 0.0, 1.0))
        self._stage_success_hist.append(streak_score)
        self._stage_episodes += 1
        self._stage_successes += streak_score

        rate = float(sum(self._stage_success_hist) / max(1, len(self._stage_success_hist)))
        changed = False
        reason = "hold"

        if self.reward_stage_enabled:
            can_change = (episode_idx - self._stage_last_change_ep) >= self._stage_cooldown
            enough = self._stage_episodes >= self._stage_min_eps
            if can_change and enough:
                if rate >= self._stage_up_thresh and self.reward_stage < self._num_reward_stages - 1:
                    prev = self.reward_stage
                    self.reward_stage += 1
                    changed = True
                    reason = (
                        f"stage-up:{prev}->{self.reward_stage} "
                        f"(score={rate:.3f}, streak={max_streak:.1f}/{self._control_streak_target})"
                    )
                elif rate <= self._stage_down_thresh and self.reward_stage > 0:
                    prev = self.reward_stage
                    self.reward_stage -= 1
                    changed = True
                    reason = (
                        f"stage-down:{prev}->{self.reward_stage} "
                        f"(score={rate:.3f}, streak={max_streak:.1f}/{self._control_streak_target})"
                    )

        if changed:
            self._stage_last_change_ep = int(episode_idx)
            self._stage_success_hist.clear()
            self._stage_episodes = 0
            self._stage_successes = 0.0

        self.cur_r_min = 0.0
        self.cur_r_max = float(self.reward_stage + 1)

        info = {
            "rate_global": rate,
            "rate_global_raw": rate,
            "rate_curr": rate,
            "rate_curr_raw": rate,
            "episodes_at_level": int(self._stage_episodes),
            "successes_at_level": float(self._stage_successes),
            "max_control_streak": float(max_streak),
            "control_streak_score": float(streak_score),
            "control_streak_target": float(self._control_streak_target),
            "changed": bool(changed),
            "reason": reason,
            "reward_stage": float(self.reward_stage),
            "reward_stage_max": float(self._num_reward_stages - 1),
            "target_dist_min_curr": self._stage_value(self._target_dist_min_stages),
            "target_dist_max_curr": self._stage_value(self._target_dist_max_stages),
            "target_angle_abs_min_deg_curr": self._stage_value(self._target_angle_abs_min_deg_stages),
            "target_angle_abs_max_deg_curr": self._stage_value(self._target_angle_abs_max_deg_stages),
            "ball_spawn_dist_min_curr": self._stage_value(self._ball_spawn_dist_min_stages),
            "ball_spawn_dist_max_curr": self._stage_value(self._ball_spawn_dist_max_stages),
            "ball_spawn_lateral_abs_curr": self._stage_value(self._ball_spawn_lateral_abs_stages),
            "episode_target_angle_deg": float(self._episode_target_angle_deg),
            "episode_target_dist": float(self._episode_target_dist),
            "episode_ball_spawn_dist": float(self._episode_ball_spawn_dist),
            "episode_ball_spawn_lateral": float(self._episode_ball_spawn_lateral),
        }
        return float(self.cur_r_min), float(self.cur_r_max), bool(changed), info

    def reset(self):
        obs, infos = self.controller.reset(
            self.default_dof_pos,
            self.dof_pos,
            self.dof_vel,
            self.dof_state,
            self.root_states_robot,
            self.root_states,
            self.last_dof_targets,
            self.last_root_vel,
            self.episode_length_buf,
            self.filtered_lin_vel,
            self.filtered_ang_vel,
            self.cmd_resample_time,
            self.delay_steps,
            self.time_out_buf,
            self.extras,
            self.commands,
            self.gait_frequency,
            self.dt,
            self.projected_gravity,
            self.base_ang_vel,
            self.gait_process,
            self.actions,
        )

        self._reset_task_state()
        self._sample_new_target()
        self._place_ball_for_dribble()
        ball_pos, _, _ = self.ball_world.get_pose(self.root_states)
        self._prev_ball_to_target_dist = torch.norm(self.target_xy - ball_pos[:2]).detach()

        self.initial_dist_xy = float(torch.norm(self.target_xy - self.base_pos[0, :2]).item())
        heading_abs, _, _, _ = self._current_heading_metrics()
        self._prev_heading_abs = heading_abs.detach()
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

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.controller._reset_idx(
            env_ids,
            self.default_dof_pos,
            self.dof_pos,
            self.dof_vel,
            self.dof_state,
            self.root_states_robot,
            self.root_states,
            self.last_dof_targets,
            self.last_root_vel,
            self.episode_length_buf,
            self.filtered_lin_vel,
            self.filtered_ang_vel,
            self.cmd_resample_time,
            self.delay_steps,
            self.time_out_buf,
            self.extras,
        )

        if env_ids.numel() > 0:
            self.reset_buf[env_ids] = False
            self.time_out_buf[env_ids] = False
            self._reset_task_state()

        self.controller._compute_observations(
            self.projected_gravity,
            self.base_ang_vel,
            self.commands,
            self.gait_frequency,
            self.gait_process,
            self.default_dof_pos,
            self.dof_pos,
            self.dof_vel,
            self.actions,
        )

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states_robot[:, 7:13]

    def compute_midlevel_reward(self):
        device = self.controller.device
        dtype = self.base_pos.dtype

        robot_xy = self.base_pos[0, :2]
        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]
        ball_v_xy = ball_lin_vel[:2]

        fwd_xy, left_xy = self._forward_left_world()
        robot_v_world = self._world_robot_velocity()[:2]

        ball_vec = ball_xy - robot_xy
        ball_dist = torch.norm(ball_vec) + 1e-6
        ball_forward = torch.dot(ball_vec, fwd_xy)
        ball_lateral = torch.dot(ball_vec, left_xy)
        ball_to_target_vec = self.target_xy - ball_xy
        ball_to_target_dist = torch.norm(ball_to_target_vec) + 1e-6
        ball_to_target_dir = ball_to_target_vec / ball_to_target_dist
        ball_speed_toward = torch.dot(ball_v_xy, ball_to_target_dir)
        ball_speed = torch.norm(ball_v_xy)
        ball_rel_speed = torch.norm(ball_v_xy - robot_v_world)

        heading_abs, heading_cos, target_dist, to_target = self._current_heading_metrics()
        speed_toward = torch.dot(robot_v_world, to_target)
        yaw_rate = self.base_ang_vel[0, 2]

        has_control = (
            (ball_dist < self._control_dist)
            & (ball_forward > self._control_front_min)
            & (torch.abs(ball_lateral) < self._control_lateral_max)
        )
        has_control_f = has_control.float()

        if bool(has_control.item()):
            self._control_hold_counter += 1
        else:
            self._control_hold_counter = 0
        self._episode_max_control_streak = max(self._episode_max_control_streak, int(self._control_hold_counter))

        prev_heading = self._prev_heading_abs
        if prev_heading is None:
            heading_progress = torch.zeros((), device=device, dtype=dtype)
        else:
            heading_progress = torch.clamp(prev_heading - heading_abs, -0.35, 0.35)
        self._prev_heading_abs = heading_abs.detach()

        control_reward = torch.tanh(4.0 * (self._control_dist - ball_dist))
        control_reward = control_reward + 0.15 * torch.tanh(2.0 * ball_forward) - 0.20 * torch.abs(ball_lateral)
        turn_reward = torch.tanh(3.0 * heading_progress)
        drive_reward = torch.tanh(1.8 * speed_toward)

        prev_ball_to_target = self._prev_ball_to_target_dist
        if prev_ball_to_target is None:
            ball_progress = torch.zeros((), device=device, dtype=dtype)
        else:
            ball_progress = torch.clamp(prev_ball_to_target - ball_to_target_dist, -0.30, 0.30)
        self._prev_ball_to_target_dist = ball_to_target_dist.detach()
        ball_progress_reward = torch.tanh(6.0 * ball_progress)

        ball_dir = ball_v_xy / (ball_speed + 1e-6)
        ball_align_cos = torch.clamp(torch.dot(ball_dir, ball_to_target_dir), -1.0, 1.0)
        pass_align_reward = torch.clamp(ball_align_cos, 0.0, 1.0)
        pass_speed_reward = torch.exp(
            -0.5 * ((ball_speed_toward - self._pass_speed_ref) / self._pass_speed_sigma) ** 2
        ) * (ball_speed_toward > 0.0).float()
        soft_pass_reward = 0.65 * pass_align_reward + 0.35 * pass_speed_reward

        # 传球后走到球前方：沿球->目标方向投影在前方，且横向偏差小
        robot_rel_ball = robot_xy - ball_xy
        front_proj = torch.dot(robot_rel_ball, ball_to_target_dir)
        side_proj = torch.dot(robot_rel_ball, torch.stack((-ball_to_target_dir[1], ball_to_target_dir[0])))
        front_reward = torch.exp(-0.5 * ((front_proj - self._reposition_front_ref) / self._reposition_front_sigma) ** 2)
        lateral_reward = torch.exp(-0.5 * (side_proj / self._reposition_lateral_sigma) ** 2)
        reposition_reward = front_reward * lateral_reward

        stage = int(np.clip(self.reward_stage, 0, self._num_reward_stages - 1))
        w_control = float(self._w_control[stage])
        w_turn = float(self._w_turn[stage])
        w_drive = float(self._w_drive[stage])
        w_soft_pass = float(self._w_soft_pass[stage])
        w_reposition = float(self._w_reposition[stage])
        w_ball_progress = float(self._w_ball_progress[stage])

        turn_gate = has_control_f
        drive_gate = has_control_f * (heading_cos > self._heading_drive_gate_cos).float()
        soft_pass_gate = has_control_f
        reposition_gate = (ball_speed_toward > self._pass_reposition_min_toward).float()

        spin_penalty = torch.where(
            (torch.abs(speed_toward) < 0.05) & (torch.abs(yaw_rate) > self._spin_yaw_thr),
            torch.abs(yaw_rate),
            torch.zeros((), device=device, dtype=dtype),
        )
        ball_speed_penalty = torch.relu(ball_speed - self._ball_speed_soft)
        rel_speed_penalty = torch.relu(ball_rel_speed - self._ball_rel_speed_soft)
        lost_ball_penalty = torch.relu(ball_dist - self._lost_ball_soft)

        in_target_zone = bool((ball_to_target_dist < self._success_target_radius).item())
        success_gate = bool(has_control.item()) and (self._control_hold_counter >= self._control_hold_steps) and in_target_zone
        if success_gate:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        success = self._success_hold_counter >= self._success_target_hold_steps
        time_out = bool(int(self.episode_length_buf[0].item()) >= self._max_episode_steps)
        lost_ball = bool(float(ball_dist.item()) > self._fail_ball_dist)
        fail = bool((time_out or lost_ball) and (not success))
        fall = bool(self.extras.get("fall", False))

        if success or fail:
            self.reset_buf[:] = True

        reward = (
            w_control * control_reward
            + w_turn * turn_gate * turn_reward
            + w_drive * drive_gate * drive_reward
            + w_soft_pass * soft_pass_gate * soft_pass_reward
            + w_reposition * reposition_gate * reposition_reward
            + w_ball_progress * ball_progress_reward
            + 0.15 * has_control_f * heading_cos
            - self._spin_penalty_scale * spin_penalty
            - self._ball_speed_penalty_scale * ball_speed_penalty
            - self._rel_speed_penalty_scale * rel_speed_penalty
            - self._lost_ball_penalty_scale * lost_ball_penalty
            - self._time_penalty
        )

        if success:
            reward = reward + self._success_bonus
        if fail:
            reward = reward - self._fail_penalty
        if fall:
            reward = reward - self._fall_penalty

        terms = {
            "reward_stage": torch.tensor(float(stage), device=device, dtype=dtype),
            "control": has_control_f.detach(),
            "control_reward": control_reward.detach(),
            "turn_reward": turn_reward.detach(),
            "drive_reward": drive_reward.detach(),
            "soft_pass_reward": soft_pass_reward.detach(),
            "pass_align_reward": pass_align_reward.detach(),
            "pass_speed_reward": pass_speed_reward.detach(),
            "reposition_reward": reposition_reward.detach(),
            "front_reward": front_reward.detach(),
            "lateral_reward": lateral_reward.detach(),
            "ball_progress_reward": ball_progress_reward.detach(),
            "ball_progress": ball_progress.detach(),
            "heading_cos": heading_cos.detach(),
            "heading_progress": heading_progress.detach(),
            "target_dist": target_dist.detach(),
            "ball_dist": ball_dist.detach(),
            "ball_to_target_dist": ball_to_target_dist.detach(),
            "in_target_zone": torch.tensor(1.0 if in_target_zone else 0.0, device=device, dtype=dtype),
            "ball_forward": ball_forward.detach(),
            "ball_lateral": ball_lateral.detach(),
            "ball_speed": ball_speed.detach(),
            "ball_speed_toward": ball_speed_toward.detach(),
            "ball_align_cos": ball_align_cos.detach(),
            "ball_rel_speed": ball_rel_speed.detach(),
            "speed_toward": speed_toward.detach(),
            "front_proj": front_proj.detach(),
            "side_proj": side_proj.detach(),
            "yaw_rate": yaw_rate.detach(),
            "spin_penalty": spin_penalty.detach(),
            "ball_speed_penalty": ball_speed_penalty.detach(),
            "rel_speed_penalty": rel_speed_penalty.detach(),
            "lost_ball_penalty": lost_ball_penalty.detach(),
            "success_hold": torch.tensor(float(self._success_hold_counter), device=device, dtype=dtype),
            "control_hold": torch.tensor(float(self._control_hold_counter), device=device, dtype=dtype),
            "success": torch.tensor(1.0 if success else 0.0, device=device, dtype=dtype),
            "fail": torch.tensor(1.0 if fail else 0.0, device=device, dtype=dtype),
        }
        self.extras["rew_terms"] = terms
        self.extras["success"] = bool(success and (not fall))
        self.extras["fail"] = bool(fail)

        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        device = self.controller.device
        dtype = self.base_pos.dtype

        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        robot_pos = self.base_pos[0, :3]
        robot_v_world = self._world_robot_velocity()

        delta_ball_world = torch.zeros(3, device=device, dtype=dtype)
        delta_ball_world[:2] = ball_pos[:2] - robot_pos[:2]
        delta_ball_body = quat_rotate_inverse(self.base_quat[0:1], delta_ball_world[None, :]).squeeze(0)

        rel_ball_v_world = torch.zeros(3, device=device, dtype=dtype)
        rel_ball_v_world[:2] = ball_lin_vel[:2] - robot_v_world[:2]
        rel_ball_v_body = quat_rotate_inverse(self.base_quat[0:1], rel_ball_v_world[None, :]).squeeze(0)

        delta_target_world = torch.zeros(3, device=device, dtype=dtype)
        delta_target_world[:2] = self.target_xy - robot_pos[:2]
        delta_target_body = quat_rotate_inverse(self.base_quat[0:1], delta_target_world[None, :]).squeeze(0)

        ball_dist = torch.norm(delta_ball_body[:2]) + 1e-6
        target_dist = torch.norm(delta_target_body[:2]) + 1e-6
        bearing = torch.atan2(delta_target_body[1], delta_target_body[0])
        cos_b = torch.cos(bearing)
        sin_b = torch.sin(bearing)
        yaw_rate = self.base_ang_vel[0, 2]

        has_control = (
            (ball_dist < self._control_dist)
            and (float(delta_ball_body[0].item()) > self._control_front_min)
            and (abs(float(delta_ball_body[1].item())) < self._control_lateral_max)
        )

        obs_vec = torch.stack(
            (
                delta_ball_body[0],
                delta_ball_body[1],
                ball_dist,
                rel_ball_v_body[0],
                rel_ball_v_body[1],
                delta_target_body[0],
                delta_target_body[1],
                target_dist,
                cos_b,
                sin_b,
                yaw_rate,
                torch.tensor(1.0 if has_control else 0.0, device=device, dtype=dtype),
            ),
            dim=0,
        )
        return obs_vec.unsqueeze(0)

    def get_initial_dist_xy(self):
        return float(self.initial_dist_xy)

    def step(self, actions):
        dof_targets = self.pre_step(actions)
        self.physics_step(dof_targets)
        self.post_step()

        obs = self.controller.obs_buf
        reward = self.compute_midlevel_reward()
        done = self.reset_buf
        info = self.extras
        return obs, reward, done, info

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

    def _is_fallen(self):
        base_z = float(self.base_pos[0, 2])
        low_z = base_z < 0.2

        device = self.controller.device
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        up_body = quat_rotate(self.base_quat[0:1], world_up[None, :]).squeeze(0)
        cos_tilt = torch.clamp(up_body[2], -1.0, 1.0)
        tilt = torch.arccos(cos_tilt)
        large_tilt = bool((tilt > 0.75).item())

        return bool(low_z or large_tilt)
