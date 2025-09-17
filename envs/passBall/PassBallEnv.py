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

class PassBallEnv:
    def __init__(self, cfg):
        self.controller = LowLevelController(cfg)
        self._init_buffers()
        self.ball_world = BallWorld(self.controller, default_z = 0.12)
    
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

        self.D_REF = 5.0
        self.V_REF = 2.0
        self.OMEGA_REF = 3.0

    def reset(self):
        obs, infos = self.controller.reset(
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.dof_state, 
            self.root_states_robot, self.root_states,
            self.last_dof_targets, self.last_root_vel,
            self.episode_length_buf,self.filtered_lin_vel,self.filtered_ang_vel,
            self.cmd_resample_time, self.delay_steps, self.time_out_buf,
            self.extras, self.commands, self.gait_frequency, self.dt,
            self.projected_gravity, self.base_ang_vel, self.gait_process, self.actions)
        
        self.ball_world.reset_at_feet(
            root_states=self.root_states,
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            forward_dist=0.30,      # 约= 球半径0.11 + 6~7cm 缓冲，可按实际踢球距离微调
            lateral_offset=0.0,     # 如果想让球偏左/偏右一点，可设成 ±0.05
            z=None,                 # 默认用 self.default_z
            zero_velocity=True
        )

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

    def compute_high_level_reward(self):
        """
        极简、无接触力的传球奖励（仅修改本函数）：
        - 主驱动：球→目标距离的“正向进步”
        - 辅助：球速度朝目标、几何对齐（机器人-球-目标、以及朝向对齐）、合理靠近球
        - 约束：侧滑/原地旋转/时间成本
        - 成功：球接近目标且基本停住
        输出：
        - 返回标量张量 reward.shape = (1,)
        - 在 self.extras["rew_terms"] 写入分解项，保持与现有 TB 键一致
        - 在 self.extras["success"] 写入布尔成功标志
        """
        dev = self.controller.device

        # ---------- 目标点（默认：机体前方 3m） ----------
        if not hasattr(self, "pass_target_xy") or self.pass_target_xy is None:
            fwd_world = quat_rotate(self.base_quat[0:1], torch.tensor([[1., 0., 0.]], device=dev)).squeeze(0)
            fwd_xy = fwd_world[:2] / (torch.norm(fwd_world[:2]) + 1e-6)
            self.pass_target_xy = (self.base_pos[0, :2] + 3.0 * fwd_xy).detach()
        target_xy = self.pass_target_xy  # (2,)

        # ---------- 关键状态（世界系） ----------
        robot_xy = self.base_pos[0, :2]
        ball_idx = self.controller.num_bodies_robot
        ball_xy  = self.body_states[0, ball_idx, 0:3][:2]
        ball_vxy = self.body_states[0, ball_idx, 7:10][:2]

        # 机体线速度：已有机体系 v_body，把它旋回世界系更直观
        v_body_xy = self.base_lin_vel[0, :2]
        v_world_xy = quat_rotate(
            self.base_quat[0:1],
            torch.cat([v_body_xy, torch.zeros(1, device=dev)], dim=0)[None, :]
        ).squeeze(0)[:2]

        # 前向单位向量（世界系）
        fwd_world = quat_rotate(self.base_quat[0:1], torch.tensor([[1., 0., 0.]], device=dev)).squeeze(0)[:2]
        fwd_world = fwd_world / (torch.norm(fwd_world) + 1e-6)

        # ---------- 几何向量 ----------
        # 机器人→球
        to_ball = ball_xy - robot_xy
        dist_rb = torch.norm(to_ball) + 1e-6
        dir_rb  = to_ball / dist_rb

        # 球→目标
        to_goal = target_xy - ball_xy
        dist_bg = torch.norm(to_goal) + 1e-6
        dir_bg  = to_goal / dist_bg
        dir_bg_orth = torch.tensor([-dir_bg[1], dir_bg[0]], device=dev)

        # ---------- 核心信号 ----------
        # 1) 进步：球→目标距离“正向减少”
        prev = getattr(self, "_prev_ball2goal", None)
        if prev is None:
            progress_raw = torch.tensor(0.0, device=dev)
        else:
            progress_raw = torch.clamp(prev - dist_bg, 0.0, 0.5)  # 只奖拉近，不奖拉远
        # 越接近目标，进步增益越大（dense & shaping）
        progress_gain = progress_raw * (1.0 / (0.5 + dist_bg))
        # 缓存当前距离用于下步差分（只在本函数内部使用，不改外部结构）
        self._prev_ball2goal = dist_bg.detach()

        # 2) 球速度朝目标的分量（希望球真的往目标走）
        ball_speed_toward_goal = torch.dot(ball_vxy, dir_bg)
        ball_speed_term = torch.tanh(0.7 * ball_speed_toward_goal)  # [-1,1] 内平滑

        # 3) 几何对齐
        #   a) 机器人→球 与 球→目标 共线（鼓励从球的“背面”踢向目标）
        align_rb_bg = torch.dot(dir_rb, dir_bg)                    # [-1, 1]
        align_term  = 0.5 * (align_rb_bg + 1.0)                    # [0, 1]
        #   b) 机器人朝向 与 球→目标方向一致
        heading_goal_cos = torch.dot(fwd_world, dir_bg)            # [-1, 1]
        heading_goal_term = 0.5 * (heading_goal_cos + 1.0)         # [0, 1]

        # 4) 合理靠近：机体朝球方向的速度分量（避免“绕圈靠近”）
        approach_speed = torch.dot(v_world_xy, dir_rb)
        approach_term  = torch.tanh(0.8 * approach_speed)          # [-1,1]

        # 5) 约束：侧滑/自旋/时间
        speed_orth   = torch.dot(v_world_xy, dir_bg_orth)          # 相对目标线的侧滑速度
        yaw_rate     = self.base_ang_vel[0, 2]
        spin_penalty = torch.clamp(torch.abs(yaw_rate), 0.0, 3.0)  # 简单抑制原地转

        time_penalty = 0.01
        fallen_penalty = 10.0 if self.extras.get("fall", False) else 0.0

        # ---------- 成功判定（不依赖接触/命中） ----------
        success_radius = 0.35   # 到点半径
        settle_speed   = 0.20   # 球基本停住阈值
        success = (dist_bg < success_radius) and (torch.norm(ball_vxy) < settle_speed)
        success_bonus = 6.0 if success else 0.0

        # ---------- 合成 ----------
        reward = (
            1.5 * progress_gain           # 主驱动：球向目标的距离进步
            + 0.8 * ball_speed_term       # 球速度朝目标
            + 0.5 * align_term            # 机器人-球-目标三点共线
            + 0.3 * heading_goal_term     # 朝向与目标方向一致
            + 0.3 * approach_term         # 合理靠近球
            - 0.2 * torch.abs(speed_orth) # 侧滑抑制
            - 0.2 * spin_penalty          # 原地自旋抑制
            - time_penalty                # 时间成本
            + success_bonus               # 成功奖励
        )

        # ---------- 填写日志键（与现有 TB 对齐） ----------
        # 注意：dist_xy 按原项目约定记录“机器人-球”的平面距离，保持兼容
        self.extras["rew_terms"]["dist_xy"]        = dist_rb.detach()
        self.extras["rew_terms"]["heading_cos"]    = heading_goal_cos.detach()
        self.extras["rew_terms"]["progress_gain"]  = progress_gain.detach()
        self.extras["rew_terms"]["speed_toward"]   = approach_speed.detach()   # 这里沿用“朝球”的速度分量命名
        self.extras["rew_terms"]["speed_orth"]     = speed_orth.detach()
        self.extras["rew_terms"]["spin_penalty"]   = spin_penalty.detach()
        self.extras["success"] = bool(success)

        return reward.view(1).to(dev)


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
    
    def compute_high_level_obs(self):
        """
        高层观测（机体系，相对量），返回 shape (1, 15)

        顺序/维度：
          0-1  p_ball_rel    球相对机体位置 [x,y]（机体系）
          2-3  v_ball_rel    球相对机体速度 [vx,vy]（机体系）
          4-5  p_goal_rel    目标相对机体位置 [x,y]（机体系）
          6-7  goal_dir_rel  球->目标方向单位向量 [ex,ey]（机体系）
          8-9  v_base_body   机体线速度 [vx,vy]（机体系）
          10   omega_z       机体角速度 ω_z（机体系）
          11-12 heading_err  机体朝向相对“球->目标线”的偏差 [sinΔψ, cosΔψ]
          13-14 gait_phase   步态相位 [sinφ, cosφ]

        注：已移除“左右足接触标志”两维，以避免依赖接触力。
        """
        device = self.controller.device

        # === 世界系的关键状态 ===
        robot_pos_w = self.base_pos[0, :3]                          # (3,)
        robot_quat  = self.base_quat[0:1]                           # (1,4)
        v_base_body = self.base_lin_vel[0, :2]                      # (2,) 已在 post_step 旋到机体系
        omega_z     = self.base_ang_vel[0, 2:3]                     # (1,) 已在机体系

        ball_idx    = self.controller.num_bodies_robot
        ball_pos_w  = self.body_states[0, ball_idx, 0:3]            # (3,)
        ball_vel_w  = self.body_states[0, ball_idx, 7:10]           # (3,)

        # 目标点（默认：机体前方 3m；保持与奖励一致）
        if not hasattr(self, "pass_target_xy") or self.pass_target_xy is None:
            fwd_world = quat_rotate(robot_quat, torch.tensor([[1., 0., 0.]], device=device)).squeeze(0)  # (3,)
            fwd_xy = fwd_world[:2]; fwd_xy = fwd_xy / (torch.norm(fwd_xy) + 1e-6)
            self.pass_target_xy = (robot_pos_w[:2] + 3.0 * fwd_xy).detach()
        target_xy = self.pass_target_xy  # (2,)

        # === 机体系旋转 ===
        # 相对位置（机体系）
        delta_ball_w = ball_pos_w - robot_pos_w
        delta_ball_b = quat_rotate_inverse(robot_quat, delta_ball_w[None, :]).squeeze(0)  # (3,)
        p_ball_rel   = delta_ball_b[:2]                                                   # (2,)

        delta_goal_w = torch.cat([target_xy - robot_pos_w[:2], torch.zeros(1, device=device)], dim=0)  # (3,)
        delta_goal_b = quat_rotate_inverse(robot_quat, delta_goal_w[None, :]).squeeze(0)               # (3,)
        p_goal_rel   = delta_goal_b[:2]                                                                 # (2,)

        # 球速度：先转机体系，再相对机体速度
        v_ball_b   = quat_rotate_inverse(robot_quat, ball_vel_w[None, :]).squeeze(0)[:2]               # (2,)
        v_ball_rel = v_ball_b - v_base_body                                                             # (2,)

        # 球->目标方向（世界系单位向量）→ 机体系
        ball_xy = ball_pos_w[:2]
        e_world_xy = (target_xy - ball_xy)
        e_world_xy = e_world_xy / (torch.norm(e_world_xy) + 1e-6)
        e_world = torch.cat([e_world_xy, torch.zeros(1, device=device)], dim=0)                         # (3,)
        e_body  = quat_rotate_inverse(robot_quat, e_world[None, :]).squeeze(0)[:2]                      # (2,)
        goal_dir_rel = e_body

        # 机体朝向（世界系 yaws）：用机体前向与 x 轴夹角近似
        fwd_world = quat_rotate(robot_quat, torch.tensor([[1., 0., 0.]], device=device)).squeeze(0)     # (3,)
        yaw_base  = torch.atan2(fwd_world[1], fwd_world[0])
        e_yaw     = torch.atan2(e_world_xy[1], e_world_xy[0])
        dpsi      = (yaw_base - e_yaw + np.pi) % (2*np.pi) - np.pi
        heading_err = torch.stack([torch.sin(dpsi), torch.cos(dpsi)], dim=0)                            # (2,)

        # 相位特征
        phi = self.gait_process[0]  # ∈ [0,1)
        gait_phase = torch.tensor([torch.sin(2*np.pi*phi), torch.cos(2*np.pi*phi)], device=device)      # (2,)

        # === 归一化 & 拼接 ===
        obs = torch.cat([
            p_ball_rel / self.D_REF,                 # 0-1
            v_ball_rel / self.V_REF,                 # 2-3
            p_goal_rel / self.D_REF,                 # 4-5
            goal_dir_rel,                            # 6-7 (单位向量不归一化)
            v_base_body / self.V_REF,                # 8-9
            omega_z / self.OMEGA_REF,                # 10
            heading_err,                             # 11-12
            gait_phase,                              # 13-14
        ], dim=0).clamp_(-2.0, 2.0)  # (15,)

        return obs.unsqueeze(0)  # (1, 15)

    def get_dist_xy(self, frame: str = "world"):
        """
        返回机器人与球的平面距离（单位：米）。
        frame:
        - "world": 直接用世界坐标 (默认)
        - "body" : 先把相对位移旋到自车系再取范数
        返回: Python float
        """
        # 取位姿
        robot_pos = self.base_pos[0, :3]                 # (3,)
        ball_idx  = self.controller.num_bodies_robot
        ball_pos  = self.body_states[0, ball_idx, 0:3]   # (3,)

        delta_world = ball_pos - robot_pos               # (3,)

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
        reward = self.compute_high_level_reward()
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

    def high_level_action_id_to_vector(self,action_id):
        """
        将DQN高层action的ID转换为向量形式
        due to the discrete action space, the action_id is a single integer
        """
        VX_FWD = 0.40
        YAW = 0.8
        FREQ = 1.5

        self.action_table = {
            0: [VX_FWD, 0.0, 0.0, FREQ],  # 前进
            1: [0.0, 0.0, +YAW, FREQ],  # 向右转
            2: [0.0, 0.0, -YAW, FREQ], # 向左转
            3: [0.0, 0.0, 0.0, FREQ],   # 停止
            4: [VX_FWD, 0.0, +0.5*YAW, FREQ], # 前进并稍微向右转
            5: [VX_FWD, 0.0, -0.5*YAW, FREQ], # 前进并稍微向左转
        }
        return self.action_table.get(int(action_id), [0.0, 0.0, 0.0, FREQ])

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

