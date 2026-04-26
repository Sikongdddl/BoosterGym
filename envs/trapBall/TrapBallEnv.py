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

from envs.components.LowLevelController import LowLevelController
from envs.components.ballWorld import BallWorld


class TrapBallEnv:
    def __init__(self, cfg, target_xy):
        self.controller = LowLevelController(cfg)
        self.target_xy = target_xy
        self.ball_world = BallWorld(self.controller, default_z=0.12)
        self._init_buffers()
        self._init_angle_curriculum()

    def _init_angle_curriculum(self):
        """
        Trap 课程学习（离散平台）：
        - 难度由“机器人-球初始连线 与 球速度方向夹角上限（度）”定义
        - 成功率上升 -> 切到更高平台（更大 angle_max_deg）
        """
        curr_cfg = self.controller.cfg.get("curriculum", {})
        levels = curr_cfg.get("angle_levels_deg", [5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
        if not isinstance(levels, (list, tuple)) or len(levels) == 0:
            levels = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
        levels = sorted(float(np.clip(float(x), 0.1, 89.0)) for x in levels)
        self._perp_levels = levels

        init_idx = int(curr_cfg.get("angle_initial_level", curr_cfg.get("perp_initial_level", 0)))
        self._perp_level_idx = int(np.clip(init_idx, 0, len(self._perp_levels) - 1))

        self._perp_up_thresh = float(curr_cfg.get("high_thresh_curr", curr_cfg.get("high_thresh", 0.6)))
        self._perp_down_thresh = float(curr_cfg.get("low_thresh", 0.3))
        self._perp_min_episodes = int(curr_cfg.get("min_episodes_at_level", 30))
        self._perp_cooldown = int(curr_cfg.get("cooldown_episodes", 8))
        win = int(curr_cfg.get("success_window", curr_cfg.get("window_global", 50)))
        self._perp_success_hist = deque(maxlen=max(10, win))
        # 成功判定：机器人在不摔倒前提下到达预设 target
        self._target_reach_radius = float(curr_cfg.get("target_reach_radius", 0.5))
        self._target_stop_speed = float(curr_cfg.get("target_stop_speed", 0.15))
        self._target_stop_hold_steps = int(curr_cfg.get("target_stop_hold_steps", 3))
        # 速度整形：远处鼓励快，近处鼓励刹车
        self._target_speed_gate_dist = float(curr_cfg.get("target_speed_gate_dist", 1.0))
        self._target_brake_zone = float(curr_cfg.get("target_brake_zone", 1.0))
        self._target_brake_weight = float(curr_cfg.get("target_brake_weight", 0.8))
        # Trap 奖励权重（强调沿球路正向接球 + 强惩罚跌倒）
        self._trap_path_align_weight = float(curr_cfg.get("trap_path_align_weight", 1.2))
        self._trap_forward_catch_weight = float(curr_cfg.get("trap_forward_catch_weight", 0.9))
        self._trap_reverse_toward_penalty = float(curr_cfg.get("trap_reverse_toward_penalty", 0.35))
        self._trap_pre_target_dash_weight = float(curr_cfg.get("trap_pre_target_dash_weight", 1.0))
        self._trap_post_target_turn_weight = float(curr_cfg.get("trap_post_target_turn_weight", 0.9))
        self._trap_post_target_turn_penalty = float(curr_cfg.get("trap_post_target_turn_penalty", 0.25))
        self._trap_post_target_gate_scale = float(curr_cfg.get("trap_post_target_gate_scale", 1.5))
        # 到点后转身方向：True=面向来球（-ball_dir），False=沿球速方向（+ball_dir）
        self._trap_post_target_face_incoming = bool(curr_cfg.get("trap_post_target_face_incoming", True))
        self._trap_post_target_intercept_weight = float(curr_cfg.get("trap_post_target_intercept_weight", 1.1))
        self._trap_post_target_reverse_penalty = float(curr_cfg.get("trap_post_target_reverse_penalty", 0.35))
        self._trap_reach_target_bonus = float(curr_cfg.get("trap_reach_target_bonus", 18.0))
        # 默认要求“到达 target 之后触球”才判成功，才能真正学习第二阶段
        self._trap_success_touch_after_target = bool(curr_cfg.get("trap_success_touch_after_target", True))
        self._trap_fall_penalty = float(curr_cfg.get("trap_fall_penalty", 120.0))
        self._trap_fall_terminal_penalty = float(curr_cfg.get("trap_fall_terminal_penalty", 40.0))
        # 惩罚偏离“初始水平线法向（defense_forward）”的速度，鼓励纯平移
        self._trap_line_normal_speed_penalty = float(
            curr_cfg.get("trap_line_normal_speed_penalty", 0.25)
        )
        # 提前稳定性约束：在真正跌倒前就对大倾斜进行惩罚
        self._trap_tilt_penalty_weight = float(curr_cfg.get("trap_tilt_penalty_weight", 6.0))
        self._trap_tilt_safe_rad = float(curr_cfg.get("trap_tilt_safe_rad", 0.35))

        self._perp_episodes_at_level = 0
        self._perp_successes_at_level = 0
        self._perp_last_change_ep = -10**9

        # 复用 runner 现有 curriculum 日志字段名，r_max 表示当前 angle_max_deg
        self.cur_r_min = 0.0
        self.cur_r_max = float(self._perp_levels[self._perp_level_idx])
        self._episode_init_angle_deg = 0.0

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
        # BaseTask.render() 在 controller 上读取 root_states 做相机跟随与离屏采帧
        # 这里把 env 的 root_states 显式挂给 controller，避免 headless 录像时报属性缺失
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
        # 过渡 shaping: 记录上一步拦截点距离，鼓励“尽快逼近拦截位”
        self._prev_intercept_dist = None
        self._reached_target_once = False

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

        curr_cfg = self.controller.cfg.get("curriculum", {})
        spawn_dist_min = float(curr_cfg.get("trap_spawn_dist_min", 2.5))
        spawn_dist_max = float(curr_cfg.get("trap_spawn_dist_max", 4.0))
        if spawn_dist_max < spawn_dist_min:
            spawn_dist_min, spawn_dist_max = spawn_dist_max, spawn_dist_min
        lateral_abs = abs(float(curr_cfg.get("trap_spawn_lateral_abs", 0.8)))
        speed_min = float(curr_cfg.get("trap_ball_speed_min", 1.0))
        speed_max = float(curr_cfg.get("trap_ball_speed_max", 2.4))
        if speed_max < speed_min:
            speed_min, speed_max = speed_max, speed_min

        spawn_dist = np.random.uniform(spawn_dist_min, spawn_dist_max)
        lateral = np.random.uniform(-lateral_abs, lateral_abs)
        speed = np.random.uniform(speed_min, speed_max)

        forward = self._defense_forward.detach().cpu().numpy()
        left = self._defense_left.detach().cpu().numpy()
        ball_xy = np.array([base_x, base_y]) + forward * spawn_dist + left * lateral

        # 课程控制：直接控制“机器人-球连线 与 球速方向”的夹角上限（度）
        to_robot = np.array([base_x, base_y], dtype=np.float32) - ball_xy.astype(np.float32)
        to_robot_norm = np.linalg.norm(to_robot)
        if to_robot_norm < 1e-6:
            to_robot = (-forward).astype(np.float32)
            to_robot_norm = np.linalg.norm(to_robot)
        dir_to_robot = to_robot / max(1e-6, to_robot_norm)

        angle_max_deg = float(getattr(self, "cur_r_max", 10.0))
        # 避免“太正”的来球：按绝对夹角采样，再随机左右符号
        angle_abs_min_deg = float(curr_cfg.get("angle_abs_min_deg", 0.0))
        angle_abs_min_deg = float(np.clip(angle_abs_min_deg, 0.0, 89.0))
        angle_abs_max_deg = float(np.clip(angle_max_deg, 0.0, 89.0))
        if angle_abs_max_deg < angle_abs_min_deg:
            angle_abs_max_deg = angle_abs_min_deg
        delta_abs_deg = float(np.random.uniform(angle_abs_min_deg, angle_abs_max_deg))
        delta_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        delta_deg = float(delta_sign * delta_abs_deg)
        delta_rad = float(np.deg2rad(delta_deg))
        c, s = np.cos(delta_rad), np.sin(delta_rad)
        vel_dir = np.array(
            [
                c * dir_to_robot[0] - s * dir_to_robot[1],
                s * dir_to_robot[0] + c * dir_to_robot[1],
            ],
            dtype=np.float32,
        )
        vel_norm = np.linalg.norm(vel_dir)
        if vel_norm < 1e-6:
            vel_dir = dir_to_robot.copy()
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
        # 初始夹角（度）：机器人-球连线 vs 球速方向（便于诊断）
        cosang = float(np.clip(np.dot(dir_to_robot, vel_dir), -1.0, 1.0))
        self._episode_init_angle_deg = float(np.rad2deg(np.arccos(cosang)))

    def _trap_geometry(self):
        device = self.controller.device
        dtype = self.base_pos.dtype
        robot_xy = self.base_pos[0, :2]
        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]
        ball_vel_xy = ball_lin_vel[:2]
        ball_speed = torch.norm(ball_vel_xy)
        # 以球速度方向定义拦截线；低速时退化为防守朝向，避免数值抖动
        if ball_speed > 0.1:
            ball_dir = ball_vel_xy / (ball_speed + 1e-6)
        else:
            ball_dir = -self._defense_forward

        signed_line_dist = torch.dot(ball_xy - self._init_robot_xy, self._defense_forward)
        lateral_offset = torch.dot(ball_xy - self._init_robot_xy, self._defense_left)
        robot_to_ball = ball_xy - robot_xy
        robot_to_ball_dist = torch.norm(robot_to_ball) + 1e-6
        dir_rb = robot_to_ball / robot_to_ball_dist

        approach_speed = -torch.dot(ball_vel_xy, self._defense_forward)
        crossing_time = torch.tensor(float("inf"), device=device, dtype=dtype)
        if approach_speed > 1e-5:
            crossing_time = torch.clamp(signed_line_dist / (approach_speed + 1e-6), min=0.0)

        # 目标点：机器人“初始位置”水平线（沿 reset 时 defense_left）与球速方向直线的交点
        # line_robot_init: init_robot_xy + s * defense_left
        # line_ball:       ball_xy       + t * ball_dir
        p0 = self._init_robot_xy
        h = self._defense_left
        d = ball_dir
        den = d[0] * h[1] - d[1] * h[0]  # 2D cross(d, h)
        rb = p0 - ball_xy
        # 始终按直线交点公式计算；仅做分母数值保护，避免 den≈0 时数值爆炸
        den_sign = torch.where(den >= 0.0, torch.ones_like(den), -torch.ones_like(den))
        den_safe = torch.where(torch.abs(den) < 1e-6, den_sign * 1e-6, den)
        t_line = (rb[0] * h[1] - rb[1] * h[0]) / den_safe  # 2D cross(rb, h) / cross(d, h)
        intercept_xy = ball_xy + d * t_line
        if ball_speed < 0.1:
            intercept_xy = ball_xy.clone()

        self.target_xy = intercept_xy.detach().clone()
        intercept_offset = intercept_xy - robot_xy
        intercept_dist = torch.norm(intercept_offset) + 1e-6
        intercept_dir = intercept_offset / intercept_dist
        line_err_signed = torch.dot(intercept_offset, h)
        target_line_residual = torch.dot(intercept_xy - p0, self._defense_forward)
        robot_line_offset = torch.dot(robot_xy - p0, self._defense_forward)

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
            "line_dir": h,
            "line_err_signed": line_err_signed,
            "target_line_residual": target_line_residual,
            "robot_line_offset": robot_line_offset,
            "den": den,
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

        # 支持无窗口（viewer 创建失败/--headless）时的离屏录像
        if (getattr(self.controller, "viewer", None) is not None) or bool(
            self.controller.cfg.get("viewer", {}).get("record_video", False)
        ):
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
        heading_world = quat_rotate(
            self.base_quat[0:1],
            torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype),
        ).squeeze(0)[:2]
        heading_world = heading_world / (torch.norm(heading_world) + 1e-6)
        move_thresh = torch.tensor(0.1, device=device, dtype=dtype)
        if robot_speed > move_thresh:
            v_dir = v_world_xy / (robot_speed + 1e-6)
            approach_cos = torch.clamp(torch.dot(v_dir, geom["dir_rb"]), -1.0, 1.0)
            intercept_cos = torch.clamp(torch.dot(v_dir, geom["intercept_dir"]), -1.0, 1.0)
        else:
            v_dir = torch.zeros(2, device=device, dtype=dtype)
            approach_cos = torch.zeros((), device=device, dtype=dtype)
            intercept_cos = torch.zeros((), device=device, dtype=dtype)

        line_err_signed = geom["line_err_signed"]
        line_err_abs = torch.abs(line_err_signed)
        line_move_sign = torch.sign(line_err_signed)
        v_toward_line = torch.dot(v_world_xy, geom["line_dir"]) * line_move_sign
        if (robot_speed > move_thresh) and bool((line_err_abs > 1e-4).item()):
            desired_line_dir = geom["line_dir"] * line_move_sign
            line_cos = torch.clamp(torch.dot(v_dir, desired_line_dir), -1.0, 1.0)
        else:
            line_cos = torch.zeros((), device=device, dtype=dtype)

        touch_now = self.is_feet_contact_ball()
        first_touch = (not self._has_touched_ball) and touch_now and (geom["signed_line_dist"] >= -0.05)
        if touch_now:
            self._has_touched_ball = True

        # 成功定义：机器人在不摔倒前提下进入 target 半径
        target_dist = geom["intercept_dist"]
        reach_target = bool((target_dist <= self._target_reach_radius).item())
        first_reach_target = bool((not self._reached_target_once) and reach_target)
        if reach_target:
            self._reached_target_once = True
        speed_ok = bool((robot_speed <= self._target_stop_speed).item())
        # 方向 shaping 仅保留“沿拦截线平移前进”，不再做到点后的转向阶段切换
        post_target_gate = torch.tensor(0.0, device=device, dtype=dtype)
        pre_target_gate = torch.tensor(1.0, device=device, dtype=dtype)

        crossed = bool((geom["signed_line_dist"] < -0.05).item())
        touch_after_target = bool(self._reached_target_once and touch_now)
        if self._trap_success_touch_after_target:
            success = bool(touch_after_target and (not self.extras.get("fall", False)))
        else:
            success = bool(reach_target and (not self.extras.get("fall", False)))
        fail = bool(self.extras.get("fall", False) and (not success))

        r_success = torch.tensor(60.0 if success else 0.0, device=device, dtype=dtype)
        r_reach_target_bonus = torch.tensor(
            float(self._trap_reach_target_bonus) if first_reach_target else 0.0, device=device, dtype=dtype
        )
        r_near_ball = 1.0 * torch.exp(-geom["robot_to_ball_dist"] / 1.2)
        r_intercept = 1.4 * torch.exp(-geom["intercept_dist"] / 0.75)
        r_approach = 0.5 * (approach_cos + 1.0)
        # 对齐奖励改为“沿初始水平线朝目标侧移动”，避免法向漂移成为优势策略
        r_path_align = torch.relu(line_cos)

        # ---- 过渡 shaping（新增）----
        # 1) 拦截点进度奖励：当前步比上一步更接近拦截点则给正奖，反之给负奖
        if self._prev_intercept_dist is None:
            progress_to_intercept = torch.zeros((), device=device, dtype=dtype)
        else:
            progress_to_intercept = self._prev_intercept_dist - geom["intercept_dist"]
        self._prev_intercept_dist = geom["intercept_dist"].detach()
        r_progress = 3.0 * torch.clamp(progress_to_intercept, min=-0.25, max=0.25)

        # 2) 朝目标侧速度分量：仅鼓励沿水平线（line_dir）向目标侧移动
        v_toward_intercept = torch.dot(v_world_xy, geom["intercept_dir"])
        speed_gate = torch.clamp(
            target_dist / max(1e-6, float(self._target_speed_gate_dist)),
            min=0.0,
            max=1.0,
        )
        # 方向奖励：严格沿水平线方向推进（line_dir），不再鼓励法向分量
        r_move_along_intercept = (
            float(self._trap_pre_target_dash_weight)
            * speed_gate
            * torch.tanh(torch.relu(v_toward_line) / 0.6)
        )

        # 3) 时间裕度（urgency）：t_ball - t_robot
        #    球越接近防线（deadline 越近）时，该项权重越大。
        v_robot_ref = torch.tensor(1.2, device=device, dtype=dtype)
        t_robot = torch.clamp(geom["intercept_dist"] / (v_robot_ref + 1e-6), min=0.0, max=6.0)
        t_ball = torch.tensor(6.0, device=device, dtype=dtype)
        if geom["approach_speed"] > 1e-5:
            t_ball = torch.clamp(
                geom["signed_line_dist"] / (geom["approach_speed"] + 1e-6),
                min=0.0,
                max=6.0,
            )
        time_margin = torch.clamp(t_ball - t_robot, min=-2.0, max=2.0)
        urgency_gate = torch.exp(-torch.clamp(geom["signed_line_dist"], min=0.0) / 1.5)
        r_urgency = 0.65 * speed_gate * urgency_gate * torch.tanh(time_margin)

        # 4) 近目标刹车惩罚：越接近 target，越惩罚过高速度
        brake_gate = torch.clamp(
            1.0 - target_dist / max(1e-6, float(self._target_brake_zone)),
            min=0.0,
            max=1.0,
        )
        r_brake = -float(self._target_brake_weight) * brake_gate * torch.relu(robot_speed - self._target_stop_speed)
        settle_gate = torch.clamp(
            1.0 - target_dist / max(1e-6, float(self._target_reach_radius) * 2.0),
            min=0.0,
            max=1.0,
        )
        r_settle = 0.8 * settle_gate * torch.exp(
            -robot_speed / max(1e-6, float(self._target_stop_speed))
        )
        line_normal_speed = torch.abs(torch.dot(v_world_xy, self._defense_forward))
        line_normal_penalty = (
            float(self._trap_line_normal_speed_penalty)
            * speed_gate
            * line_normal_speed
        )

        line_penalty = 0.25 * torch.relu(-geom["signed_line_dist"])
        lateral_penalty = 0.10 * torch.abs(geom["lateral_offset"])
        time_penalty = torch.tensor(0.01, device=device, dtype=dtype)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        up_body = quat_rotate(self.base_quat[0:1], world_up[None, :]).squeeze(0)
        tilt = torch.arccos(torch.clamp(up_body[2], -1.0, 1.0))
        tilt_penalty = float(self._trap_tilt_penalty_weight) * torch.relu(
            tilt - float(self._trap_tilt_safe_rad)
        )
        fall_penalty = torch.tensor(
            float(self._trap_fall_penalty) if self.extras.get("fall", False) else 0.0,
            device=device,
            dtype=dtype,
        )
        fall_terminal_penalty = torch.tensor(
            float(self._trap_fall_terminal_penalty) if fail else 0.0,
            device=device,
            dtype=dtype,
        )

        reward = (
            r_near_ball
            + r_intercept
            + 0.4 * r_approach
            + float(self._trap_path_align_weight) * r_path_align
            + r_progress
            + r_move_along_intercept
            + r_urgency
            + r_brake
            + r_settle
            + r_reach_target_bonus
            + r_success
            - line_penalty
            - lateral_penalty
            - line_normal_penalty
            - time_penalty
            - tilt_penalty
            - fall_penalty
            - fall_terminal_penalty
        )

        self.extras["success"] = success
        self.extras["fail"] = bool(fail)
        self.extras["hit"] = bool(reach_target)

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
            "line_cos": line_cos.detach(),
            "r_near_ball": r_near_ball.detach(),
            "r_intercept": r_intercept.detach(),
            "r_approach": (0.4 * r_approach).detach(),
            "r_path": (float(self._trap_path_align_weight) * r_path_align).detach(),
            "r_path_align_raw": r_path_align.detach(),
            "r_progress": r_progress.detach(),
            "r_move_along_intercept": r_move_along_intercept.detach(),
            "pre_target_gate": pre_target_gate.detach(),
            "post_target_gate": post_target_gate.detach(),
            "r_urgency": r_urgency.detach(),
            "r_brake": r_brake.detach(),
            "r_settle": r_settle.detach(),
            "progress_to_intercept": progress_to_intercept.detach(),
            "line_err_signed": line_err_signed.detach(),
            "v_toward_line": v_toward_line.detach(),
            "v_toward_intercept": v_toward_intercept.detach(),
            "line_normal_speed": line_normal_speed.detach(),
            "line_normal_penalty": (-line_normal_penalty).detach(),
            "speed_gate": speed_gate.detach(),
            "brake_gate": brake_gate.detach(),
            "time_margin": time_margin.detach(),
            "r_reach_target_bonus": r_reach_target_bonus.detach(),
            "r_success": r_success.detach(),
            "r_contact": r_success.detach(),
            "line_penalty": (-line_penalty).detach(),
            "lateral_penalty": (-lateral_penalty).detach(),
            "time_penalty": (-time_penalty).detach(),
            "tilt_rad": tilt.detach(),
            "tilt_penalty": (-tilt_penalty).detach(),
            "fall_penalty": (-fall_penalty).detach(),
            "fall_terminal_penalty": (-fall_terminal_penalty).detach(),
            "touch_now": torch.tensor(1.0 if touch_now else 0.0, device=device, dtype=dtype),
            "first_touch": torch.tensor(1.0 if first_touch else 0.0, device=device, dtype=dtype),
            "target_dist": target_dist.detach(),
            "reach_target": torch.tensor(1.0 if reach_target else 0.0, device=device, dtype=dtype),
            "speed_ok": torch.tensor(1.0 if speed_ok else 0.0, device=device, dtype=dtype),
            "target_stop_hold_counter": torch.tensor(0.0, device=device, dtype=dtype),
            "target_stop_hold_steps": torch.tensor(float(self._target_stop_hold_steps), device=device, dtype=dtype),
            "stop_success": torch.tensor(1.0 if success else 0.0, device=device, dtype=dtype),
            "reached_target_once": torch.tensor(1.0 if self._reached_target_once else 0.0, device=device, dtype=dtype),
            "touch_after_reach": torch.tensor(1.0 if touch_after_target else 0.0, device=device, dtype=dtype),
            "target_reach_radius": torch.tensor(float(self._target_reach_radius), device=device, dtype=dtype),
            "target_stop_speed": torch.tensor(float(self._target_stop_speed), device=device, dtype=dtype),
            "crossed_line": torch.tensor(1.0 if crossed else 0.0, device=device, dtype=dtype),
            "target_line_residual": geom["target_line_residual"].detach(),
            "robot_line_offset": geom["robot_line_offset"].detach(),
            "cross_d_h": geom["den"].detach(),
        }
        self.extras["rew_terms"] = terms
        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        """
        与 ChaseBall 对齐的高层观测，shape (1, 8)
        0-1: delta_xy_body（target 在机体系的相对位置）
        2:   dist_xy
        3-4: cos(bearing), sin(bearing)
        5-6: v_body_xy
        7:   speed_toward（沿 target 方向速度分量）
        """
        device = self.controller.device
        geom = self._trap_geometry()
        robot_pos = self.base_pos[0, :3]

        delta_world = torch.zeros(3, device=device, dtype=robot_pos.dtype)
        delta_world[:2] = self.target_xy - robot_pos[:2]
        delta_body = quat_rotate_inverse(self.base_quat[0:1], delta_world[None, :]).squeeze(0)
        delta_xy_body = delta_body[:2]

        dist_xy = torch.norm(delta_xy_body) + 1e-6
        bearing = torch.atan2(delta_xy_body[1], delta_xy_body[0])
        cos_b = torch.cos(bearing)
        sin_b = torch.sin(bearing)

        v_body_xy = self.base_lin_vel[0, :2]
        speed_toward = v_body_xy[0] * cos_b + v_body_xy[1] * sin_b

        obs_vec = torch.stack(
            (
                delta_xy_body[0], delta_xy_body[1],
                dist_xy, cos_b, sin_b,
                v_body_xy[0], v_body_xy[1],
                speed_toward,
            ),
            dim=0,
        )
        return obs_vec.unsqueeze(0)

    def get_initial_dist_xy(self):
        return float(self._episode_init_dist)

    def on_episode_end(self, success: bool, episode_idx: int):
        s = 1 if bool(success) else 0
        self._perp_success_hist.append(s)
        self._perp_episodes_at_level += 1
        self._perp_successes_at_level += s

        rate = float(sum(self._perp_success_hist) / max(1, len(self._perp_success_hist)))
        changed = False
        reason = "hold"

        can_change = (episode_idx - self._perp_last_change_ep) >= self._perp_cooldown
        enough_samples = self._perp_episodes_at_level >= self._perp_min_episodes

        if can_change and enough_samples:
            if rate >= self._perp_up_thresh and self._perp_level_idx < len(self._perp_levels) - 1:
                prev = self._perp_level_idx
                self._perp_level_idx += 1
                changed = True
                reason = f"angle-up:{prev}->{self._perp_level_idx} (rate={rate:.3f})"
            elif rate <= self._perp_down_thresh and self._perp_level_idx > 0:
                prev = self._perp_level_idx
                self._perp_level_idx -= 1
                changed = True
                reason = f"angle-down:{prev}->{self._perp_level_idx} (rate={rate:.3f})"

        if changed:
            self._perp_last_change_ep = int(episode_idx)
            self._perp_success_hist.clear()
            self._perp_episodes_at_level = 0
            self._perp_successes_at_level = 0

        self.cur_r_min = 0.0
        self.cur_r_max = float(self._perp_levels[self._perp_level_idx])

        info = {
            "rate_global": rate,
            "rate_global_raw": rate,
            "rate_curr": rate,
            "rate_curr_raw": rate,
            "episodes_at_level": int(self._perp_episodes_at_level),
            "successes_at_level": int(self._perp_successes_at_level),
            "changed": bool(changed),
            "reason": reason,
            "angle_level": float(self._perp_level_idx),
            "angle_level_max": float(len(self._perp_levels) - 1),
            "angle_max_deg": float(self.cur_r_max),
            "episode_init_angle_deg": float(self._episode_init_angle_deg),
            # 兼容旧键名
            "perp_level": float(self._perp_level_idx),
            "perp_level_max": float(len(self._perp_levels) - 1),
            "perp_dist_max": float(self.cur_r_max),
            "episode_init_perp_dist": float(self._episode_init_angle_deg),
        }
        return float(self.cur_r_min), float(self.cur_r_max), changed, info
