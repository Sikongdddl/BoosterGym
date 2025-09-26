# TestHLPEnv.py
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, quat_rotate_inverse, get_axis_params

from envs.components.LowLevelController import LowLevelController


class TestHLPEnv:
    """
    Dumb test env:
    - 只做：创建 -> reset -> (obs, step with PD) -> obs
    - 不训练、不做高层任务，保留与原工程兼容的若干属性
    - 观测与动作均为 per-agent 形状 [num_agents, ...]
    """

    def __init__(self, cfg):
        self.controller = LowLevelController(cfg)
        self._init_buffers()

    # --------------------------------------------------------------------- #
    # Buffers & tensors
    # --------------------------------------------------------------------- #
    def _init_buffers(self):
        cfg = self.controller.cfg
        dev = self.controller.device
        N = int(self.controller.num_agents)

        # ===== 基本尺寸（与原风格保持一致） =====
        self.num_obs = int(cfg["env"]["num_observations"])
        self.num_privileged_obs = int(cfg["env"]["num_privileged_obs"])
        self.num_actions = int(cfg["env"]["num_actions"])
        self.dt = cfg["control"]["decimation"] * cfg["sim"]["dt"]

        # 与原工程兼容的占位字段
        self.cur_r_min = 1.0
        self.cur_r_max = 1.5
        self._milestones = [4.0, 3.0, 2.0, 1.5, 1.0]
        self._milestones_passed = set()

        # ===== core buffers =====
        self.obs_buf = torch.zeros(N, self.num_obs, dtype=torch.float, device=dev)
        self.privileged_obs_buf = torch.zeros(N, self.num_privileged_obs, dtype=torch.float, device=dev)
        self.rew_buf = torch.zeros(0, dtype=torch.float, device=dev)  # 不用于 dumb env
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=dev)  # 单 env：仍保持 (1,)
        self.episode_length_buf = torch.zeros(1, device=dev, dtype=torch.long)
        self.time_out_buf = torch.zeros(1, device=dev, dtype=torch.bool)
        self.extras = {"rew_terms": {}}
        self._prev_dist_xy = None
        self.initial_dist_xy = 0.0

        # ===== 绑定仿真张量视图 =====
        g, sim = self.controller.gym, self.controller.sim
        actor_root_state = g.acquire_actor_root_state_tensor(sim)
        dof_state_tensor = g.acquire_dof_state_tensor(sim)
        body_state = g.acquire_rigid_body_state_tensor(sim)

        g.refresh_dof_state_tensor(sim)
        g.refresh_actor_root_state_tensor(sim)
        g.refresh_dof_force_tensor(sim)
        g.refresh_rigid_body_state_tensor(sim)

        # 全局视图
        self.root_states = gymtorch.wrap_tensor(actor_root_state)              # [num_actors_total, 13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)               # [total_dofs, 2]
        # body_states 维度：这里将其 reshape 为 [1, total_bodies(+extra), 13] 以兼容老逻辑
        self.body_states = gymtorch.wrap_tensor(body_state).view(
            1, self.controller.total_robot_bodies + self.controller.addtional_rigid_num, 13
        )

        # 仅取机器人 root（约定：创建顺序在前 N 条）
        self.root_states_robot = self.root_states[0:N, :]  # [N, 13]

        # ===== 初始化每台机器人的 DOF 状态缓存（按映射采样）=====
        D = int(self.controller.dofs_per_agent)
        self.dof_pos = torch.zeros(N, D, device=dev)
        self.dof_vel = torch.zeros(N, D, device=dev)
        self._gather_dof_pos_vel()  # 首次填充

        # ===== 基础状态（机体系速度、重力投影）=====
        self.base_pos = self.root_states_robot[:, 0:3]
        self.base_quat = self.root_states_robot[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.controller.up_axis_idx), device=dev).repeat((N, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()

        # ===== 动作与 PD 相关 =====
        self.actions = torch.zeros(N, self.num_actions, dtype=torch.float, device=dev)
        self.last_actions = torch.zeros(N, self.num_actions, dtype=torch.float, device=dev)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states_robot[:, 7:13])
        self.last_dof_targets = torch.zeros(N, self.controller.dofs_per_agent, dtype=torch.float, device=dev)
        self.torques = torch.zeros(N, self.controller.dofs_per_agent, dtype=torch.float, device=dev)

        # ===== 命令 & 步态（供 _compute_observations 使用）=====
        C = int(cfg["commands"]["num_commands"])
        self.commands = torch.zeros(N, C, dtype=torch.float, device=dev)      # [N, C]，函数内部用 commands[:, :3]
        self.cmd_resample_time = torch.zeros(1, dtype=torch.long, device=dev) # 兼容字段
        self.gait_frequency = torch.zeros(N, dtype=torch.float, device=dev)   # [N]
        self.gait_process = torch.zeros(N, dtype=torch.float, device=dev)     # [N]

        # ===== 关节默认角 =====
        self.dof_pos_ref = torch.zeros(N, self.controller.dofs_per_agent, dtype=torch.float, device=dev)
        self.default_dof_pos = torch.zeros(N, self.controller.dofs_per_agent, dtype=torch.float, device=dev)
        for j in range(self.controller.num_dofs):
            found = False
            for name in cfg["init_state"]["default_joint_angles"].keys():
                if name in self.controller.dof_names[j]:
                    self.default_dof_pos[:, j] = cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, j] = cfg["init_state"]["default_joint_angles"]["default"]

        # 兼容其它框架可能读取的属性
        self.num_envs = 1
        self.num_agents = N

    # 从全局 dof_state 刷新并按 dof_slices 采样到 per-agent 张量
    def _gather_dof_pos_vel(self):
        g, sim = self.controller.gym, self.controller.sim
        g.refresh_dof_state_tensor(sim)
        ds = self.dof_state.view(-1, 2)  # [total_dofs, 2]
        pos_all, vel_all = ds[:, 0], ds[:, 1]

        N = int(self.controller.num_agents)
        D = int(self.controller.dofs_per_agent)
        if (self.dof_pos.shape[0] != N) or (self.dof_pos.shape[1] != D):
            self.dof_pos = torch.zeros(N, D, device=self.controller.device)
            self.dof_vel = torch.zeros(N, D, device=self.controller.device)

        for i, sl in enumerate(self.controller.dof_slices):
            if isinstance(sl, slice):
                self.dof_pos[i, :] = pos_all[sl]
                self.dof_vel[i, :] = vel_all[sl]
            else:
                self.dof_pos[i, :] = pos_all.index_select(0, sl)
                self.dof_vel[i, :] = vel_all.index_select(0, sl)

    # --------------------------------------------------------------------- #
    # External API
    # --------------------------------------------------------------------- #
    def reset(self):
        """
        调用底层 controller.reset 完成复位与初始观测；
        保留 privileged_obs（全 0 占位）与 dumb 渲染。
        """
        obs, infos = self.controller.reset(
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.dof_state,
            self.root_states_robot, self.root_states,
            self.last_dof_targets, self.last_root_vel,
            self.episode_length_buf,
            self.base_lin_vel.clone(), self.base_ang_vel.clone(),
            self.cmd_resample_time,
            self.delay_steps if hasattr(self, "delay_steps") else torch.zeros(1, dtype=torch.long, device=self.controller.device),
            self.time_out_buf,
            self.extras,
            self.commands,              # [N, C]
            self.gait_frequency,        # [N]
            self.dt,
            self.projected_gravity, self.base_ang_vel,
            self.gait_process,          # [N]
            self.actions
        )

        # dumb env：特权观测全 0
        self.privileged_obs_buf.zero_()

        # （可选）短暂 settle，减小复位接触噪声
        try:
            g, sim = self.controller.gym, self.controller.sim
            for _ in range(20):
                g.simulate(sim)
                if self.controller.device == "cpu":
                    g.fetch_results(sim, True)
                g.refresh_dof_state_tensor(sim)
                g.refresh_actor_root_state_tensor(sim)
                g.refresh_rigid_body_state_tensor(sim)
        except Exception:
            pass

        # 初次刷新 per-agent DOF
        self._gather_dof_pos_vel()

        # 渲染一帧（如果 viewer 存在）
        if getattr(self.controller, "viewer", None) is not None:
            self.controller.render()

        return obs, infos

    def step(self, actions):
        """
        最小闭环：
        - clamp → 目标 → 每个子步：刷新DOF→PD→拼flat下发→simulate
        - 返回 obs（[N, num_obs]）、零奖励、未终止、空 info
        """
        N = self.controller.num_agents
        D = self.controller.dofs_per_agent
        assert actions.shape[0] == N, f"actions batch={actions.shape[0]} vs num_agents={N}"
        assert actions.shape[1] == D, f"per-agent act_dim mismatch: expect {D}, got {actions.shape[1]}"

        # 1) 动作（per-agent）
        clip_actions = float(self.controller.cfg["normalization"]["clip_actions"])
        act_scale = float(self.controller.cfg["control"]["action_scale"])
        self.actions[:] = torch.clamp(actions, -clip_actions, clip_actions)
        dof_targets = self.default_dof_pos + act_scale * self.actions

        # 2) decimation 子步
        g, sim = self.controller.gym, self.controller.sim
        self.torques.zero_()
        for _ in range(self.controller.cfg["control"]["decimation"]):
            # 2.1 刷新 DOF → 采样 per-agent 状态
            self._gather_dof_pos_vel()
            sub_dt = float(self.controller.cfg["sim"]["dt"])
            self.gait_process = (self.gait_process + self.gait_frequency * sub_dt) % 1.0
            # 2.2 目标（不做时延）
            self.last_dof_targets[:] = dof_targets

            # 2.3 计算 PD 力矩
            dof_torques = (
                self.controller.dof_stiffness * (self.last_dof_targets - self.dof_pos)
                - self.controller.dof_damping * self.dof_vel
            )
            friction = torch.min(self.controller.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            tl = self.controller.torque_limits.view(1, -1).expand_as(dof_torques)
            dof_torques = torch.clamp(dof_torques - friction, min=-tl, max=tl)
            self.torques += dof_torques

            # 2.4 拼成全局一维并一次性下发
            flat = torch.zeros(self.controller.total_dofs, device=self.controller.device)
            for ag_idx, sl in enumerate(self.controller.dof_slices):
                if isinstance(sl, slice):
                    flat[sl] = dof_torques[ag_idx]
                else:
                    flat.index_copy_(0, sl, dof_torques[ag_idx])
            g.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(flat))

            # 2.5 物理推进 + 必要刷新（下一子步前会再次 gather）
            g.simulate(sim)
            if self.controller.device == "cpu":
                g.fetch_results(sim, True)
            g.refresh_dof_force_tensor(sim)

        self.torques /= self.controller.cfg["control"]["decimation"]

        # 3) 刷新 root/body，更新基础量
        g.refresh_actor_root_state_tensor(sim)
        g.refresh_rigid_body_state_tensor(sim)

        self.base_pos[:] = self.root_states_robot[:, 0:3]
        self.base_quat[:] = self.root_states_robot[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robot[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 4) 由底层计算观测（per-agent）
        self.controller._compute_observations(
            self.projected_gravity, self.base_ang_vel,
            self.commands,                 # [N, C]
            self.gait_frequency,           # [N]
            self.gait_process,             # [N]
            self.default_dof_pos, self.dof_pos, self.dof_vel, self.actions
        )

        obs = self.controller.obs_buf
        reward = torch.zeros(1, device=self.controller.device)
        done = torch.zeros(1, dtype=torch.bool, device=self.controller.device)
        info = {}

        # —— 渲染（与 PassBallEnv 同步）——
        if getattr(self.controller, "viewer", None) is not None:
            self.controller.render()

        return obs, reward, done, info

    # --------------------------------------------------------------------- #
    # Minimal stubs for upper-level compatibility
    # --------------------------------------------------------------------- #
    def get_privileged_obs(self):
        return self.privileged_obs_buf

    def compute_high_level_obs(self):
        """Dumb桩：返回全零高层观测 [N, 8]，仅用于跑通流程"""
        N = int(self.controller.num_agents)
        return torch.zeros(N, 8, device=self.controller.device)

    def respawn_ball_far(self, r_min=1.0, r_max=1.5):
        # 哑实现：不做任何事，避免上层报错
        return

    def set_curriculum_window(self, succ_count: int, epi_count: int):
        # 哑实现：不做任何事
        return

    def apply_high_level_command(self, cmd, smooth: float = 0.0):
        """
        Runner 会调用这里传入高层命令:
        cmd: [vx, vy, yaw_rate, gait_freq]
        行为：
        - 把命令写入 self.commands（[N, C]）
        - 可选地做一次 EMA 平滑（与 Runner 的 smooth 参数一致）
        """
        vx, vy, wz, gait_f = float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])
        N = int(self.controller.num_agents)
        dev = self.controller.device

        target = torch.tensor([vx, vy, wz], device=dev).view(1, 3).repeat(N, 1)   # [N,3]
        if smooth and smooth > 0.0:
            # 简单 EMA: new = (1-s)*old + s*target
            s = float(smooth)
            self.commands[:, :3] = (1.0 - s) * self.commands[:, :3] + s * target
        else:
            self.commands[:, :3] = target

        # 步频：全体 agent 先一致（需要的话可以改成 per-agent）
        self.gait_frequency[:] = gait_f