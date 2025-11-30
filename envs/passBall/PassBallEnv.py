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

class PassBallEnv:
    def __init__(self, cfg, target_xy):
        self.controller = LowLevelController(cfg)
        #target_xy: Tensor, shape (2,), 目标在世界坐标系的XY位置
        self.target_xy = target_xy
        self.pass_radius = 0.2 # 球路过 target 的容差
        self.pass_max_speed = 0.5 # 球别太快直接飞过去 太难断球不能算成功
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

        self.ball_world.reset_pass_ball(
            root_states=self.root_states,
            base_xy=base_xy,
        )        
        self._prev_dist_xy = None  # 重置进步奖励计算
        self._prev_ball_dist = None
        self._has_touched_ball = False
        self._milestones_passed.clear()
        self.initial_dist_xy = None

        
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
        """
        passBall 任务的高层奖励（新版，事件驱动）：

        1) 触球前：用一个很小的 shaping 奖励机器人移动方向对齐 球->target
        2) 首次触球：给一个很大的一次性奖励
        3) 成功（球进入 target 邻域）：给更大的终止奖励
        4) 时间惩罚 + 摔倒惩罚
        5) 其它几何量（球速方向、距离等）只做 logging，不进 reward
        """
        device = self.controller.device
        dtype  = self.base_pos.dtype

        # --- 目标位置 ---
        target_xy = self.target_xy  # (2,)

        # --- 机器人 / 球 的位姿 & 球速度 ---
        robot_pos = self.base_pos[0, :3]  # (3,)

        # 用 BallWorld 的接口拿球的位置和线速度（世界系）
        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]          # (2,)
        ball_vel_xy = ball_lin_vel[:2]  # (2,)

        # ========== 1. 球 -> target 的几何量（仅用作 log） ==========
        delta_bt = target_xy - ball_xy
        ball_dist = torch.norm(delta_bt) + 1e-6      # 当前球到 target 的距离
        dir_bt = delta_bt / ball_dist                # 单位向量：球 -> target (2,)

        # ========== 2. 机器人速度方向 vs 球->target 方向 ==========
        # base_lin_vel 当前是自车系，先旋回世界系
        v_body = self.base_lin_vel[0, :3]  # (3,)
        v_world = quat_rotate(self.base_quat[0:1], v_body[None, :]).squeeze(0)  # (3,)
        v_world_xy = v_world[:2]

        robot_speed = torch.norm(v_world_xy)
        robot_move_thresh = torch.tensor(0.1, device=device, dtype=dtype)  # 认为“在走”的最小速度

        if robot_speed > robot_move_thresh:
            v_dir = v_world_xy / (robot_speed + 1e-6)
            robot_align_cos = torch.clamp(torch.dot(v_dir, dir_bt), -1.0, 1.0)
            # 映射到 [0,1]，越对准球->target，值越接近 1
            robot_align_reward = 0.5 * (robot_align_cos + 1.0)
        else:
            robot_align_cos = torch.tensor(0.0, device=device, dtype=dtype)
            robot_align_reward = torch.tensor(0.0, device=device, dtype=dtype)

        # ========== 3. 球速方向 vs 球->target 方向（只做 log） ==========
        ball_speed = torch.norm(ball_vel_xy)
        ball_move_thresh = torch.tensor(0.2, device=device, dtype=dtype)

        if ball_speed > ball_move_thresh:
            ball_dir = ball_vel_xy / (ball_speed + 1e-6)
            ball_align_cos = torch.clamp(torch.dot(ball_dir, dir_bt), -1.0, 1.0)
            ball_align_reward = 0.5 * (ball_align_cos + 1.0)
        else:
            ball_align_cos = torch.tensor(0.0, device=device, dtype=dtype)
            ball_align_reward = torch.tensor(0.0, device=device, dtype=dtype)

        # ========== 4. 触球检测 & 奖励 ==========
        # 用“球离机器人是否足够近”来近似触球
        delta_br = ball_xy - robot_pos[:2]
        ball_to_robot = torch.norm(delta_br)

        touch_radius = 0.4  # 小于这个距离算“脚边/触球”
        touch_now = ball_to_robot < touch_radius

        prev_touched = getattr(self, "_has_touched_ball", False)
        first_touch = (not prev_touched) and bool(touch_now)
        if touch_now:
            self._has_touched_ball = True

        # 大额触球奖励
        r_touch = torch.tensor(
            20.0 if first_touch else 0.0,
            device=device, dtype=dtype
        )

        # ========== 5. 成功判定 ==========
        success_thresh = 0.60  # 球进入这个半径算成功
        success = bool((ball_dist < success_thresh).item())
        r_succ = torch.tensor(
            80.0 if success else 0.0,
            device=device, dtype=dtype
        )

        # ========== 6. 时间 & 摔倒惩罚 ==========
        time_penalty = torch.tensor(0.01, device=device, dtype=dtype)
        fallen_penalty = torch.tensor(
            10.0 if self.extras.get("fall", False) else 0.0,
            device=device,
            dtype=dtype,
        )

        # ========== 7. 合成总奖励 ==========
        # 机器人方向 shaping：仅在“尚未触球”阶段生效
        if not prev_touched:
            # robot_align_reward ∈ [0,1]，给很小的权重，避免形成“离谱高分失败 episode”
            r_robot = 0.3 * robot_align_reward
        else:
            r_robot = torch.zeros((), device=device, dtype=dtype)

        # 球速方向暂时不加到 reward，只记录日志
        r_ball = torch.zeros((), device=device, dtype=dtype)

        reward = (
            r_robot
            + r_ball
            + r_touch
            + r_succ
            - time_penalty
            - fallen_penalty
        )

        # ========== 8. logging ==========
        if "rew_terms" not in self.extras:
            self.extras["rew_terms"] = {}
        terms = self.extras["rew_terms"]

        # 几何信息
        terms["ball_dist"]         = ball_dist.detach()
        terms["ball_speed"]        = ball_speed.detach()
        terms["ball_align_cos"]    = ball_align_cos.detach()
        terms["ball_align_reward"] = ball_align_reward.detach()

        terms["robot_speed"]        = robot_speed.detach()
        terms["robot_align_cos"]    = robot_align_cos.detach()
        terms["robot_align_reward"] = robot_align_reward.detach()

        # 子 reward 分量
        terms["r_robot"] = r_robot.detach()
        terms["r_ball"]  = r_ball.detach()
        terms["r_touch"] = r_touch.detach()
        terms["r_succ"]  = r_succ.detach()
        terms["time_penalty"]   = (-time_penalty).detach()
        terms["fallen_penalty"] = (-fallen_penalty).detach()

        # 事件标记
        terms["touch_now"] = torch.tensor(
            1.0 if touch_now else 0.0,
            device=device, dtype=dtype
        )
        terms["first_touch"] = torch.tensor(
            1.0 if first_touch else 0.0,
            device=device, dtype=dtype
        )

        self.extras["success"] = success

        # ========== 9. 终端 debug 输出 ==========
        debug_flag = getattr(self, "debug_reward", True)
        if debug_flag:
            print(
                "[RewardDebug] "
                f"dist={float(ball_dist):.3f}, "
                f"robot_speed={float(robot_speed):.3f}, "
                f"robot_align={float(robot_align_reward):.3f}, "
                f"ball_speed={float(ball_speed):.3f}, "
                f"ball_align={float(ball_align_reward):.3f}, "
                f"r_robot={float(r_robot):.3f}, "
                f"r_touch={float(r_touch):.1f}, "
                f"r_succ={float(r_succ):.1f}, "
                f"fallen_penalty={float(fallen_penalty):.1f}, "
                f"total={float(reward):.3f}"
            )

        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        """
        高层观测，返回 shape = (1, 8)

        各分量（全部是世界坐标系下的量）：
        0: base_x        机器人基座 x
        1: base_y        机器人基座 y
        2: ball_x        球的 x
        3: ball_y        球的 y
        4: v_world_x     机器人在世界系下的 vx
        5: v_world_y     机器人在世界系下的 vy
        6: target_x      target 的 x
        7: target_y      target 的 y
        """
        device = self.base_pos.device
        dtype  = self.base_pos.dtype

        # ----- 机器人位置 -----
        base_pos = self.base_pos[0, :3]             # (3,)
        base_x = base_pos[0]
        base_y = base_pos[1]

        # ----- 球的位置 -----
        ball_pos, _, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]                      # (2,)
        ball_x  = ball_xy[0]
        ball_y  = ball_xy[1]

        # ----- 机器人速度（世界系） -----
        # 当前 base_lin_vel 是自车系，旋回世界系
        v_body  = self.base_lin_vel[0, :3]          # (3,)
        v_world = quat_rotate(self.base_quat[0:1],
                            v_body[None, :]).squeeze(0)  # (3,)
        v_world_x = v_world[0]
        v_world_y = v_world[1]

        # ----- target 位置 -----
        target_xy = self.target_xy                  # (2,)
        target_x  = target_xy[0]
        target_y  = target_xy[1]

        # ----- 拼观测向量 -----
        obs = torch.stack([
            base_x,
            base_y,
            ball_x,
            ball_y,
            v_world_x,
            v_world_y,
            target_x,
            target_y,
        ], dim=0).to(device=device, dtype=dtype)    # (8,)

        return obs.view(1, -1)


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
    
    def get_robot_to_ball_dist_xy(self, frame: str = "world"):
        """
        返回机器人与球的平面距离（单位：米）。
        frame:
        - "world": 直接用世界坐标 (默认)
        - "body" : 先把相对位移旋到自车系再取范数
        返回: Python float
        """
        device = self.controller.device
        dtype  = self.base_pos.dtype

        # 取机器人和球的位姿（世界系）
        robot_pos = self.base_pos[0, :3]  # (3,)

        num_robot_bodies = self.controller.num_bodies_robot
        ball_idx = num_robot_bodies
        ball_pos = self.body_states[0, ball_idx, 0:3]  # (3,)

        # 世界系下的相对位移：球 - 机器人
        delta_world = torch.zeros(3, device=device, dtype=dtype)
        delta_world[:2] = ball_pos[:2] - robot_pos[:2]

        if frame == "world":
            dist_xy = torch.norm(delta_world[:2])
        else:
            # 旋到机体系
            delta_body = quat_rotate_inverse(
                self.base_quat[0:1],
                delta_world[None, :]
            ).squeeze(0)
            dist_xy = torch.norm(delta_body[:2])

        # 护航，避免偶发 NaN 干扰日志
        if torch.isnan(dist_xy) or torch.isinf(dist_xy):
            dist_xy = torch.tensor(float("nan"), device=dist_xy.device)

        return float(dist_xy.item())
        
    
    def get_ball_to_target_dist_xy(self, frame: str = "world"):
        """
        返回球与 target_xy 的平面距离（单位：米）。
        frame:
        - "world": 直接用世界坐标 (默认)
        - "body" : 先把相对位移旋到自车系再取范数
        返回: Python float
        """
        device = self.controller.device
        dtype  = self.base_pos.dtype

        # 球的位置（世界系）
        num_robot_bodies = self.controller.num_bodies_robot
        ball_idx = num_robot_bodies
        ball_pos = self.body_states[0, ball_idx, 0:3]  # (3,)

        target_xy = self.target_xy.to(device)

        delta_world = torch.zeros(3, device=device, dtype=dtype)
        # 世界系：target - 球
        delta_world[:2] = target_xy - ball_pos[:2]

        if frame == "world":
            dist_xy = torch.norm(delta_world[:2])
        else:
            # 虽然距离跟坐标系无关，这里只是保持接口一致，旋到机体系
            delta_body = quat_rotate_inverse(
                self.base_quat[0:1],
                delta_world[None, :]
            ).squeeze(0)
            dist_xy = torch.norm(delta_body[:2])

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

