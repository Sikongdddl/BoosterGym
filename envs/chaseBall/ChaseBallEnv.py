import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_rotate,
)
from typing import Dict, Tuple, Optional
from envs.components.LowLevelController import LowLevelController
from envs.components.ballWorld import BallWorld
from envs.components.curriculum import CurriculumPolicy

class ChaseBallEnv:
    def __init__(self, cfg, target_xy):
        self.controller = LowLevelController(cfg)
        #target_xy: Tensor, shape (2,), 目标在世界坐标系的XY位置
        self.target_xy = target_xy
        self._init_buffers()
        self.ball_world = BallWorld(self.controller, default_z = 0.12)
        self.curriculum = CurriculumPolicy.from_dict(self.controller.cfg.get("curriculum"))
        self.cur_r_min, self.cur_r_max = self.curriculum.get_window()
        self._last_curr_info: Optional[Dict] = None  # 记录最近一次课程信息，便于runner取数

    def _init_buffers(self):
        cfg = self.controller.cfg
        dev = self.controller.device

        self.num_obs = cfg["env"]["num_observations"]
        self.num_privileged_obs = cfg["env"]["num_privileged_obs"]
        self.num_actions = cfg["env"]["num_actions"]
        self.dt = cfg["control"]["decimation"] * cfg["sim"]["dt"]

        self.cur_r_min = 1.0
        self.cur_r_max = 1.5

        self._milestones = [4.0,3.0,2.0,1.5,1.0]
        self._milestones_passed = set()

        # core buffers
        self.obs_buf = torch.zeros(1, self.num_obs, dtype=torch.float, device=dev)
        self.rew_buf = torch.zeros(0, dtype=torch.float, device=dev)
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=dev)
        self.episode_length_buf = torch.zeros(1, device=dev, dtype=torch.long)
        self.time_out_buf = torch.zeros(1, device=dev, dtype=torch.bool)
        self.extras = {"rew_terms": {}}

        self._prev_dist_xy = None  # 用于计算进步奖励
        self.initial_dist_xy = 0.0

        # get gym state tensors
        actor_root_state = self.controller.gym.acquire_actor_root_state_tensor(self.controller.sim)
        dof_state_tensor = self.controller.gym.acquire_dof_state_tensor(self.controller.sim)
        body_state = self.controller.gym.acquire_rigid_body_state_tensor(self.controller.sim)

        self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
        self.controller.gym.refresh_actor_root_state_tensor(self.controller.sim)
        self.controller.gym.refresh_dof_force_tensor(self.controller.sim)
        self.controller.gym.refresh_rigid_body_state_tensor(self.controller.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # we only care robot states instead of other assets now so:
        self.root_states_robot = self.root_states[0:1,:]  
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(1, self.controller.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(1, self.controller.num_dofs, 2)[..., 1]
        self.body_states = gymtorch.wrap_tensor(body_state).view(1, self.controller.num_bodies_robot + self.controller.addtional_rigid_num, 13)
        self.base_pos = self.root_states_robot[:, 0:3]
        self.base_quat = self.root_states_robot[:, 3:7]
        self.feet_pos = self.body_states[:, self.controller.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.controller.feet_indices, 3:7]

        # initialize some data used later on
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
    
    def on_episode_end(self, success: bool, episode_idx: int) -> Tuple[float, float, bool, Dict]:
        """
        新课程学习入口：每个 episode 结束时调用。
        使用“同级验证 + 最小驻留 + 冷却 + 比例化步长”的策略更新难度。
        返回: (r_min, r_max, changed, info)
        - changed: 本次是否调整了 r_max
        - info: 包含 rate_global/rate_curr/episodes_at_level 等统计
        """
        r_min, r_max, changed, info = self.curriculum.update_on_episode_end(success, episode_idx)
        self.cur_r_min, self.cur_r_max = r_min, r_max
        self._last_curr_info = dict(info)
        if changed:
            print(f"[Curriculum] ep#{episode_idx} r_max -> {self.cur_r_max:.2f} | {info.get('reason')}")
        return r_min, r_max, changed, info

    def reset(self):
        obs, infos = self.controller.reset(
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.dof_state, 
            self.root_states_robot, self.root_states,
            self.last_dof_targets, self.last_root_vel,
            self.episode_length_buf,self.filtered_lin_vel,self.filtered_ang_vel,
            self.cmd_resample_time, self.delay_steps, self.time_out_buf,
            self.extras, self.commands, self.gait_frequency, self.dt,
            self.projected_gravity, self.base_ang_vel, self.gait_process, self.actions)
        
        self.cur_r_min, self.cur_r_max = self.curriculum.get_window()
        base_xy = (float(self.root_states[0, 0].item()), float(self.root_states[0, 1].item()))
        self.ball_world.reset_ring(self.root_states, r_min=self.cur_r_min, r_max=self.cur_r_max, base_xy=base_xy)
        
        self._prev_dist_xy = None  # 重置进步奖励计算
        self._milestones_passed.clear()
        self.initial_dist_xy = self.get_dist_xy()
        return obs, infos

    def pre_step(self, actions):
        self.actions[:] = torch.clip(
            actions, 
            -self.controller.cfg["normalization"]["clip_actions"],
            self.controller.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.controller.cfg["control"]["action_scale"] * self.actions
        return dof_targets
        
    def physics_step(self,dof_targets):
        self.torques.zero_()
        for i in range(self.controller.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.controller.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.controller.dof_damping * self.dof_vel
            friction = torch.min(self.controller.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.controller.torque_limits, max=self.controller.torque_limits)
            self.torques += dof_torques
            self.controller.gym.set_dof_actuation_force_tensor(self.controller.sim, gymtorch.unwrap_tensor(dof_torques))
            self.controller.gym.simulate(self.controller.sim)
            if self.controller.device == "cpu":
                self.controller.gym.fetch_results(self.controller.sim, True)
            self.controller.gym.refresh_dof_state_tensor(self.controller.sim)
            self.controller.gym.refresh_dof_force_tensor(self.controller.sim)
        self.torques /= self.controller.cfg["control"]["decimation"]
        
        if getattr(self.controller, 'viewer', None) is not None:
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

        # check fall early stop
        fall_now = self._is_fallen()
        if fall_now:
            self.reset_buf[:] = True
            self.extras["fall"] = True
        else:
            self.extras["fall"] = False

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.controller._reset_idx(env_ids, 
            self.default_dof_pos, self.dof_pos, self.dof_vel,self.dof_state, 
            self.root_states_robot, self.root_states,
            self.last_dof_targets, self.last_root_vel,
            self.episode_length_buf,self.filtered_lin_vel,self.filtered_ang_vel,
            self.cmd_resample_time, self.delay_steps,self.time_out_buf,
            self.extras)

        if env_ids.numel() > 0:
            self.reset_buf[env_ids] = False
            self.time_out_buf[env_ids] = False
            # 进步奖励基线清空（下一回合第一步不做错误差分）
            self._prev_dist_xy = None
            self._milestones_passed.clear()
            self.initial_dist_xy = self.get_dist_xy()

        self.controller._compute_observations(
            self.projected_gravity,self.base_ang_vel,self.commands,
            self.gait_frequency, self.gait_process,
            self.default_dof_pos,self.dof_pos, self.dof_vel, self.actions)

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states_robot[:, 7:13]

    def compute_midlevel_reward(self):
        device = self.controller.device
        dtype  = self.base_pos.dtype

        # --- 机器人位姿（世界系） ---
        robot_pos = self.base_pos[0, :3]  # (3,)

        # --- 目标位置（世界系，XY） ---
        target_xy = self.target_xy
        # --- 相对几何（仅平面） ---
        delta_xy = target_xy - robot_pos[:2]              # (2,)
        dist_xy  = torch.norm(delta_xy) + 1e-6
        to_t_xy  = delta_xy / dist_xy                     # 单位指向向量

        # --- 前向与朝向 ---
        FORWARD_LOCAL = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        fwd_world = quat_rotate(self.base_quat[0:1], FORWARD_LOCAL[None, :]).squeeze(0)  # (3,)
        fwd_xy = fwd_world[:2]
        fwd_xy = fwd_xy / (torch.norm(fwd_xy) + 1e-6)
        heading_cos = torch.clamp(torch.dot(fwd_xy, to_t_xy), -1.0, 1.0)
        heading_reward = 0.5 * (heading_cos + 1.0)  # [0,1]

        # --- 速度分解 ---
        v_xy = self.base_lin_vel[0, :2]                   # 已在 post_step 旋到机体系
        speed_toward = torch.dot(v_xy, to_t_xy)           # 朝目标速度(可正可负)
        to_t_perp = torch.stack([-to_t_xy[1], to_t_xy[0]])
        speed_orth = torch.dot(v_xy, to_t_perp)           # 侧滑速度
        speed_reward = torch.tanh(0.8 * speed_toward)     # [-1,1]

        # --- 反转圈惩罚 ---
        yaw_rate = self.base_ang_vel[0, 2]
        spinning = (torch.abs(speed_toward) < 0.02) * (torch.abs(yaw_rate) > 0.8)
        spin_penalty = torch.where(spinning, torch.abs(yaw_rate), torch.tensor(0.0, device=device, dtype=dtype))
        spin_penalty = torch.clamp(spin_penalty, 0.0, 3.0)

        # --- 进步奖励（序列依赖） ---
        prev_dist = getattr(self, "_prev_dist_xy", None)
        if prev_dist is None:
            progress_raw = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            progress_raw = torch.clamp(prev_dist - dist_xy, 0.0, 0.5)  # 只奖励正进步
        progress_gain = progress_raw * (1.0 / (dist_xy + 0.5))
        self._prev_dist_xy = dist_xy.detach()

        # --- 门控后的朝向项 ---
        moving = (speed_toward > 0.03).float()
        heading_term = moving * heading_reward

        # --- 成功判定与一次性奖励 ---
        success_thresh = 0.60
        success = (dist_xy < success_thresh)
        success_bonus = 3.0 if (success and speed_toward > 0.05) else 0.0

        # --- 时间与摔倒惩罚 ---
        time_penalty = 0.01
        fallen_penalty = 10.0 if self.extras.get("fall", False) else 0.0

        # --- 合成 ---
        reward = (
            1.2 * progress_gain
            + 0.6 * speed_reward
            + 0.2 * heading_term
            - 0.2 * torch.abs(speed_orth)
            - 0.2 * spin_penalty
            + success_bonus
            - time_penalty
            - fallen_penalty
        )

        # --- logging ---
        self.extras["rew_terms"]["heading_cos"]   = heading_cos.detach()
        self.extras["rew_terms"]["heading_term"]  = heading_term.detach()
        self.extras["rew_terms"]["dist_xy"]       = dist_xy.detach()
        self.extras["rew_terms"]["progress_gain"] = progress_gain.detach()
        self.extras["rew_terms"]["speed_toward"]  = speed_toward.detach()
        self.extras["rew_terms"]["speed_orth"]    = speed_orth.detach()
        self.extras["rew_terms"]["spin_penalty"]  = spin_penalty.detach()
        self.extras["success"] = bool(success)

        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        """
        高层观测（自车系，相对量），返回 shape (1, 8)
        各分量：
        0-1: delta_xy_body（target在自车坐标系的XY相对位置）
        2:   dist_xy（平面距离）
        3-4: cos(bearing), sin(bearing)（指向target的方位角）
        5-6: v_body_xy（自车系下机体线速度XY）
        7:   speed_toward（沿着target方向的速度分量）
        """
        device = self.controller.device
        target_xy = self.target_xy
        # 取世界系位姿
        robot_pos = self.base_pos[0, :3]                 # (3,)

        # 世界 -> 自车系：把相对位移旋到机体坐标系
        delta_world = torch.zeros(3, device=device, dtype=robot_pos.dtype)
        delta_world[:2] = target_xy - robot_pos[:2]
        # quat_rotate_inverse 接受 (N,3)，这里用 (1,3) 再 squeeze
        delta_body  = quat_rotate_inverse(self.base_quat[0:1], delta_world[None, :]).squeeze(0)  # (3,)
        delta_xy_body = delta_body[:2]                   # (2,)

        # 距离 & 方位
        dist_xy = torch.norm(delta_xy_body) + 1e-6
        bearing = torch.atan2(delta_xy_body[1], delta_xy_body[0])   # 自车系下的方位角
        cos_b   = torch.cos(bearing)
        sin_b   = torch.sin(bearing)

        # 自车系速度 & 朝向球的速度分量
        v_body_xy = self.base_lin_vel[0, :2]             # 这本来就是在自车系（你在 post_step 里已做了 quat_rotate_inverse）
        speed_toward = v_body_xy[0] * cos_b + v_body_xy[1] * sin_b

        # 拼接观测向量
        obs_vec = torch.stack((
            delta_xy_body[0], delta_xy_body[1],
            dist_xy, cos_b, sin_b,
            v_body_xy[0], v_body_xy[1],
            speed_toward,
        ), dim=0)  # (8,)

        return obs_vec.unsqueeze(0)  # (1, 8)

    def get_dist_xy(self, frame: str = "world"):
        """
        返回机器人与球的平面距离（单位：米）。
        frame:
        - "world": 直接用世界坐标 (默认)
        - "body" : 先把相对位移旋到自车系再取范数
        返回: Python float
        """
        device = self.controller.device
        dtype  = self.base_pos.dtype

        # 取位姿
        robot_pos = self.base_pos[0, :3]                 # (3,)
        delta_world = torch.zeros(3, device=device, dtype=dtype)
        delta_world[:2] = self.target_xy - robot_pos[:2]               # (3,)

        if frame == "world":
            dist_xy = torch.norm(delta_world[:2])        # torch scalar
        else:
            # 旋到机体系
            delta_body = quat_rotate_inverse(self.base_quat[0:1], delta_world[None, :]).squeeze(0)
            dist_xy = torch.norm(delta_body[:2])

        # 护航，避免偶发 NaN 干扰日志
        if torch.isnan(dist_xy) or torch.isinf(dist_xy):
            dist_xy = torch.tensor(float("nan"), device=dist_xy.device)

        return float(dist_xy.item())
        
    def get_initial_dist_xy(self):
        return float(self.initial_dist_xy)

    def step(self, actions):
        # locomotion原始step，保持不变
        dof_targets = self.pre_step(actions)
        self.physics_step(dof_targets)
        self.post_step()

        obs = self.controller.obs_buf
        reward = self.compute_midlevel_reward()
        done = self.reset_buf
        info = self.extras
        return obs, reward, done, info

    def get_high_level_action_space(self):
        """
        返回高层action的默认模板（4维向量），顺序为：
        [lin_vel_x, lin_vel_y, ang_vel_yaw, gait_frequency]
        """
        cmd_cfg = self.controller.cfg["commands"]
        gait_freq = 0.5 * (cmd_cfg["gait_frequency"][0] + cmd_cfg["gait_frequency"][1])
        return [0.0, 0.0, 0.0, gait_freq]
    
    def apply_high_level_command(self, cmd, smooth=None):
        """
        cmd: [lin_vel_x, lin_vel_y, ang_vel_yaw, gait_freq]
        smooth: 可选的低通平滑系数 alpha∈[0,1)，None 表示直写
        """
        device = self.controller.device
        # 三个速度指令
        new_cmd = torch.tensor(cmd[:3], device=device, dtype=self.commands.dtype).view(1, 3)
        if smooth is None:
            self.commands[:, :3] = new_cmd
            self.gait_frequency[:] = float(cmd[3])
        else:
            alpha = float(smooth)
            self.commands[:, :3] = alpha * self.commands[:, :3] + (1 - alpha) * new_cmd
            self.gait_frequency[:] = alpha * self.gait_frequency + (1 - alpha) * float(cmd[3])
    
    def _is_fallen(self):
        # height threshold of base
        base_z = float(self.base_pos[0, 2])
        z_threshold = 0.2
        low_z = base_z < z_threshold

        # angular threshold of robot
        device = self.controller.device
        WORLD_UP = torch.tensor([0.0, 0.0, 1.0], device=device)
        up_body = quat_rotate(self.base_quat[0:1], WORLD_UP[None, :]).squeeze(0)
        cos_tilt = torch.clamp(up_body[2], -1.0, 1.0)  # [-1, 1]
        tilt = torch.arccos(cos_tilt)
        tilt_threshold = 0.75
        large_tilt = tilt > tilt_threshold

        return low_z or large_tilt

    def debug_print_positions(self):
        # 假设你只有一个env实例，env_id=0
        env_id = 0

        # 机器人刚体数目
        num_robot_bodies = self.controller.num_bodies_robot

        # 打印机器人各刚体位置（取前三个坐标）
        print(f"Robot body positions (env {env_id}):")
        for i in range(num_robot_bodies):
            pos = self.body_states[env_id, i, 0:3]
            print(f"  Body {i}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

        # 球的刚体索引
        ball_idx = num_robot_bodies
        ball_pos = self.body_states[env_id, ball_idx, 0:3]
        print(f"Ball position (env {env_id}): x={ball_pos[0]:.3f}, y={ball_pos[1]:.3f}, z={ball_pos[2]:.3f}")

        # 打印机器人底座位置
        base_pos = self.base_pos[env_id, :3]
        print(f"Robot base position (env {env_id}): x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f}")

