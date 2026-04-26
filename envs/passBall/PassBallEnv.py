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
        self._init_reward_stage_curriculum()

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
        # BaseTask.render() 会在 controller 上读取 root_states 来做相机跟随/录制
        # 这里显式透传，避免 headless 录像时报属性缺失
        self.controller.root_states = self.root_states
        
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

    def _init_reward_stage_curriculum(self):
        curr_cfg = self.controller.cfg.get("curriculum", {})
        stage_cfg = dict(curr_cfg.get("reward_stages", {}))
        self._reward_stage_cfg = {
            "enabled": True,
            "initial_stage": 0,
            "success_window": 80,
            "stage_min_episodes": [80, 100, 100, 0],
            "stage_success_thresholds": [0.75, 0.70, 0.70],
            "stage_names": ["close_ball", "align", "touch", "precise_pass"],
            "stage2_success_metric": "touch",
            "stage_mix_probs": [
                [1.00, 0.00, 0.00, 0.00],
                [0.20, 0.80, 0.00, 0.00],
                [0.05, 0.55, 0.40, 0.00],
                [0.05, 0.10, 0.15, 0.70],
            ],
            "stage_rollback_min_episodes": [0, 0, 120, 160],
            "stage_rollback_thresholds": [0.00, 0.00, 0.20, 0.18],
            "close_dist_thresh": 1.60,
            "align_hold_steps": 6,
            "align_line_thresh": 0.25,
            "align_approach_thresh": 0.35,
            "align_robot_cos_thresh": 0.50,
            "stage2_kick_dir_thresh": 0.45,
            "stage2_kick_vpara_thresh": 0.35,
            "stage2_kick_vratio_thresh": 0.80,
            "pre_touch_scale": [1.40, 1.10, 0.25, 0.10],
            "post_touch_scale": [0.00, 0.00, 0.60, 1.50],
            "touch_bonus_scale": [0.00, 0.00, 0.50, 0.10],
            "align_bonus_scale": [0.00, 1.00, 0.00, 0.00],
            "success_bonus_scale": [0.00, 0.00, 0.10, 1.50],
        }
        self._reward_stage_cfg.update(stage_cfg)

        self.reward_stage_enabled = bool(self._reward_stage_cfg.get("enabled", True))
        default_stage_names = ["close_ball", "align", "touch", "precise_pass"]
        self._num_reward_stages = len(self._reward_stage_cfg.get("stage_names", default_stage_names))
        self.max_reward_stage = int(self._reward_stage_cfg.get("initial_stage", 0))
        self.max_reward_stage = max(0, min(self.max_reward_stage, self._num_reward_stages - 1))
        self.reward_stage = self.max_reward_stage
        self._reward_stage_episode_count = 0
        self._success_hist = deque(maxlen=int(self._reward_stage_cfg.get("success_window", 80)))
        self._episode_close_success = False
        self._episode_align_success = False
        self._episode_touch_success = False
        self._episode_coarse_success = False
        self._episode_precise_success = False
        self._align_hold_counter = 0
        self._touch_count = 0
        self._touch_now_prev = False

    def _reward_stage_value(self, key: str) -> float:
        values = self._reward_stage_cfg.get(key, [1.0] * self._num_reward_stages)
        if isinstance(values, (list, tuple)) and len(values) >= self._num_reward_stages:
            return float(values[self.reward_stage])
        return float(values)

    def _reward_stage_name(self) -> str:
        names = self._reward_stage_cfg.get("stage_names", ["close_ball", "align", "touch", "precise_pass"])
        if isinstance(names, (list, tuple)) and len(names) >= self._num_reward_stages:
            return str(names[self.reward_stage])
        return str(self.reward_stage)

    def _sample_reward_stage_for_episode(self) -> int:
        if not self.reward_stage_enabled:
            return int(self.max_reward_stage)
        mix_cfg = self._reward_stage_cfg.get("stage_mix_probs", [])
        if (
            isinstance(mix_cfg, (list, tuple))
            and len(mix_cfg) > self.max_reward_stage
            and isinstance(mix_cfg[self.max_reward_stage], (list, tuple))
        ):
            probs = np.asarray(mix_cfg[self.max_reward_stage], dtype=np.float64)
        else:
            probs = np.zeros(self._num_reward_stages, dtype=np.float64)
            probs[: self.max_reward_stage + 1] = 1.0
        if probs.shape[0] < self._num_reward_stages:
            probs = np.pad(probs, (0, self._num_reward_stages - probs.shape[0]))
        probs[self.max_reward_stage + 1 :] = 0.0
        probs = np.clip(probs, 0.0, None)
        if float(probs.sum()) <= 0.0:
            probs = np.zeros(self._num_reward_stages, dtype=np.float64)
            probs[self.max_reward_stage] = 1.0
        probs = probs / probs.sum()
        return int(np.random.choice(np.arange(self._num_reward_stages), p=probs))

    def _stage_success_flag(self, stage_idx: int) -> bool:
        if stage_idx <= 0:
            return bool(self._episode_close_success)
        if stage_idx == 1:
            return bool(self._episode_align_success)
        if stage_idx == 2:
            stage2_metric = str(self._reward_stage_cfg.get("stage2_success_metric", "touch")).lower()
            if stage2_metric in ("touch", "valid_hit", "hit"):
                return bool(self._episode_touch_success)
            return bool(self._episode_coarse_success)
        return bool(self._episode_precise_success)

    def _reward_stage_snapshot(self) -> Dict:
        has_samples = len(self._success_hist) > 0
        success_rate = float(sum(self._success_hist) / len(self._success_hist)) if has_samples else float("nan")
        stage_names = self._reward_stage_cfg.get("stage_names", ["close_ball", "align", "touch", "precise_pass"])
        return {
            "reward_stage": float(self.reward_stage),
            "reward_stage_name": self._reward_stage_name(),
            "reward_stage_max": float(self.max_reward_stage),
            "reward_stage_max_name": str(stage_names[self.max_reward_stage]),
            "stage_success_rate": success_rate,
            "stage_success_rate_valid": float(1.0 if has_samples else 0.0),
            "episodes_in_reward_stage": float(self._reward_stage_episode_count),
        }

    def _update_reward_stage_curriculum(self, task_success: bool, episode_idx: int) -> Dict:
        frontier_stage = int(self.max_reward_stage)
        sampled_stage = int(self.reward_stage)
        frontier_success = bool(self._stage_success_flag(frontier_stage))
        active_stage_success = bool(self._stage_success_flag(sampled_stage))
        sampled_frontier = sampled_stage == frontier_stage

        changed = False
        reason = None
        prev_stage = self.max_reward_stage
        if sampled_frontier:
            self._success_hist.append(1 if frontier_success else 0)
            self._reward_stage_episode_count += 1

        success_rate = float(sum(self._success_hist) / len(self._success_hist)) if self._success_hist else 0.0
        if self.reward_stage_enabled:
            thresholds = list(self._reward_stage_cfg.get("stage_success_thresholds", [0.60] * (self._num_reward_stages - 1)))
            min_episodes = list(self._reward_stage_cfg.get("stage_min_episodes", [40] * self._num_reward_stages))
            rollback_min_episodes = list(self._reward_stage_cfg.get("stage_rollback_min_episodes", [0] * self._num_reward_stages))
            rollback_thresholds = list(self._reward_stage_cfg.get("stage_rollback_thresholds", [0.0] * self._num_reward_stages))
            if frontier_stage < self._num_reward_stages - 1:
                min_eps = int(min_episodes[min(frontier_stage, len(min_episodes) - 1)])
                thresh = float(thresholds[min(frontier_stage, len(thresholds) - 1)])
                if sampled_frontier and self._reward_stage_episode_count >= min_eps and success_rate >= thresh:
                    self.max_reward_stage = min(frontier_stage + 1, self._num_reward_stages - 1)
                    changed = True
                    reason = f"reward-stage-up:{frontier_stage}->{self.max_reward_stage} (stage_success_rate={success_rate:.3f})"
            if (not changed) and frontier_stage > 0:
                rollback_min_eps = int(rollback_min_episodes[min(frontier_stage, len(rollback_min_episodes) - 1)])
                rollback_thresh = float(rollback_thresholds[min(frontier_stage, len(rollback_thresholds) - 1)])
                if sampled_frontier and rollback_min_eps > 0 and self._reward_stage_episode_count >= rollback_min_eps and success_rate < rollback_thresh:
                    self.max_reward_stage = max(frontier_stage - 1, 0)
                    changed = True
                    reason = f"reward-stage-down:{frontier_stage}->{self.max_reward_stage} (stage_success_rate={success_rate:.3f})"

        if changed:
            self._reward_stage_episode_count = 0
            self._success_hist.clear()

        info = self._reward_stage_snapshot()
        info.update({
            "reward_stage_changed": bool(changed),
            "reward_stage_prev": float(prev_stage),
            "reward_stage_reason": reason,
            "episode_sampled_stage": float(sampled_stage),
            "episode_frontier_stage": float(frontier_stage),
            "episode_stage_success": float(active_stage_success),
            "episode_frontier_stage_success": float(frontier_success),
            "episode_task_success": float(bool(task_success)),
            "episode_touch": float(bool(self._episode_touch_success)),
            "episode_success": float(bool(task_success)),
        })
        return info
    # 根据权重调整采样策略
    def on_episode_end(self, success: bool, episode_idx: int) -> Tuple[float, float, bool, Dict]:
        """
        新课程学习入口：每个 episode 结束时调用。
        使用“同级验证 + 最小驻留 + 冷却 + 比例化步长”的策略更新难度。
        返回: (r_min, r_max, changed, info)
        - changed: 本次是否调整了 r_max
        - info: 包含 rate_global/rate_curr/episodes_at_level 等统计
        """
        stage_info = self._update_reward_stage_curriculum(success, episode_idx)
        r_min, r_max, changed, info = self.curriculum.update_on_episode_end(success, episode_idx)
        self.cur_r_min, self.cur_r_max = r_min, r_max
        merged_info = dict(info)
        merged_info.update(stage_info)
        self._last_curr_info = dict(merged_info)
        if changed:
            print(f"[Curriculum] ep#{episode_idx} r_max -> {self.cur_r_max:.2f} | {info.get('reason')}")
        if bool(stage_info.get("reward_stage_changed", False)):
            print(
                f"[RewardStage] ep#{episode_idx} -> {int(stage_info.get('reward_stage_max', self.max_reward_stage))} "
                f"({stage_info.get('reward_stage_reason')})"
            )
        return r_min, r_max, changed, merged_info

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
        stage_cfg = self.controller.cfg.get("curriculum", {}).get("reward_stages", {})
        ball_spawn_min = float(stage_cfg.get("ball_spawn_min_dist", 0.2))
        ball_spawn_max = float(stage_cfg.get("ball_spawn_max_dist", 0.6))
        theta_min_deg = float(stage_cfg.get("ball_spawn_theta_min_deg", -20.0))
        theta_max_deg = float(stage_cfg.get("ball_spawn_theta_max_deg", 20.0))
        fwd_local = torch.tensor([1.0, 0.0, 0.0], device=self.base_quat.device, dtype=self.base_quat.dtype)
        fwd_world = quat_rotate(self.base_quat[0:1], fwd_local[None, :]).squeeze(0)
        heading = float(torch.atan2(fwd_world[1], fwd_world[0]).item())

        self.ball_world.reset_pass_ball(           
            root_states=self.root_states,
            base_xy=base_xy,
            r_min=ball_spawn_min,
            r_max=ball_spawn_max,
            theta_range=(heading + np.deg2rad(theta_min_deg), heading + np.deg2rad(theta_max_deg)),
        )
        self.reward_stage = self._sample_reward_stage_for_episode()
        self._prev_dist_xy = None  # 重置进步奖励计算
        self._prev_ball_dist = None
        self._prev_ball_speed = None
        self._has_touched_ball = False
        self._touch_now_prev = False
        self._touch_count = 0
        self._align_hold_counter = 0
        self._episode_close_success = False
        self._episode_align_success = False
        self._episode_touch_success = False
        self._episode_coarse_success = False
        self._episode_precise_success = False
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
        
        if (
            getattr(self.controller, "viewer", None) is not None
            or bool(self.controller.cfg.get("viewer", {}).get("record_video", False))
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
        passBall 任务的高层奖励（重构版）：

        1) 触球前：重点鼓励“接近球 + 站到球后方 + 朝球接近”
        2) 首次有效触球：给中等事件奖励，避免“只要碰到就赚”
        3) 触球后：持续鼓励“球朝 target 飞 + 球离 target 更近”
        4) 成功（球进入 target 邻域）：给大额终止奖励
        5) 时间惩罚 + 摔倒惩罚
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

        # ========== 2. 机器人朝向/速度几何 ==========
        # 2.0 机器人线速度（世界系）
        # base_lin_vel 当前是自车系，先旋回世界系
        v_body = self.base_lin_vel[0, :3]  # (3,)
        v_world = quat_rotate(self.base_quat[0:1], v_body[None, :]).squeeze(0)  # (3,)
        v_world_xy = v_world[:2]

        # ========== 2.1 机器人-球-目标三点共线 + 面向球 + 朝球前进（用于反制绕球刷分） ==========
        delta_rb = ball_xy - robot_pos[:2]                  # 机器人 -> 球
        robot_to_ball_dist = torch.norm(delta_rb) + 1e-6
        dir_rb = delta_rb / robot_to_ball_dist

        # 身体前向是否“面对球”
        fwd_local = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        fwd_world = quat_rotate(self.base_quat[0:1], fwd_local[None, :]).squeeze(0)
        fwd_world_xy = fwd_world[:2]
        fwd_norm = torch.norm(fwd_world_xy)
        if fwd_norm > 1e-6:
            fwd_dir = fwd_world_xy / (fwd_norm + 1e-6)
            robot_align_cos = torch.clamp(torch.dot(fwd_dir, dir_rb), -1.0, 1.0)
            # 仅奖励真正“朝向球”，侧向/背向不给正奖励
            robot_align_reward = torch.clamp(robot_align_cos, 0.0, 1.0)
        else:
            robot_align_cos = torch.tensor(0.0, device=device, dtype=dtype)
            robot_align_reward = torch.tensor(0.0, device=device, dtype=dtype)

        line_cos = torch.clamp(torch.dot(dir_rb, dir_bt), -1.0, 1.0)
        # 只奖励“机器人在球后方且球在目标方向上”的几何关系（cos>0）
        line_reward = torch.clamp(line_cos, 0.0, 1.0)
        # 距离门控：离球越近，共线奖励越有效，避免远距离刷形状分
        line_dist_gate = torch.exp(-robot_to_ball_dist / 1.5)

        robot_speed = torch.norm(v_world_xy)
        robot_move_thresh = torch.tensor(0.1, device=device, dtype=dtype)  # 认为“在走”的最小速度
        if robot_speed > robot_move_thresh:
            v_dir = v_world_xy / (robot_speed + 1e-6)
            approach_cos = torch.clamp(torch.dot(v_dir, dir_rb), -1.0, 1.0)
            approach_reward = 0.5 * (approach_cos + 1.0)
            tangentiality = torch.sqrt(torch.clamp(1.0 - approach_cos * approach_cos, min=0.0, max=1.0))
        else:
            approach_cos = torch.tensor(0.0, device=device, dtype=dtype)
            approach_reward = torch.tensor(0.0, device=device, dtype=dtype)
            tangentiality = torch.tensor(0.0, device=device, dtype=dtype)

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

        # ========== 4. 触球检测 ==========
        touch_now = self.is_feet_contact_ball()
        prev_touch_now = bool(getattr(self, "_touch_now_prev", False))
        touch_event = bool(touch_now and (not prev_touch_now))
        if touch_event:
            self._touch_count = int(getattr(self, "_touch_count", 0)) + 1
        touch_count = int(getattr(self, "_touch_count", 0))
        prev_touched = getattr(self, "_has_touched_ball", False)

        prev_ball_speed = getattr(self, "_prev_ball_speed", None)
        if prev_ball_speed is None:
            prev_ball_speed = torch.zeros((), device=device, dtype=dtype)
        ball_speed_gain = torch.clamp(ball_speed - prev_ball_speed, min=0.0)
        hit_speed_thresh = torch.tensor(0.20, device=device, dtype=dtype)
        hit_gain_thresh = torch.tensor(0.08, device=device, dtype=dtype)
        valid_hit_now = bool(touch_now) and bool(
            ((ball_speed > hit_speed_thresh) or (ball_speed_gain > hit_gain_thresh)).item()
        )
        first_touch = (not prev_touched) and valid_hit_now
        if valid_hit_now:
            self._has_touched_ball = True

        # ========== 5. 阶段成功判定 ==========
        close_dist_thresh = torch.tensor(float(self._reward_stage_cfg.get("close_dist_thresh", 1.60)), device=device, dtype=dtype)
        align_line_thresh = torch.tensor(float(self._reward_stage_cfg.get("align_line_thresh", 0.25)), device=device, dtype=dtype)
        align_approach_thresh = torch.tensor(float(self._reward_stage_cfg.get("align_approach_thresh", 0.35)), device=device, dtype=dtype)
        align_robot_cos_thresh = torch.tensor(float(self._reward_stage_cfg.get("align_robot_cos_thresh", 0.50)), device=device, dtype=dtype)
        align_hold_steps = int(self._reward_stage_cfg.get("align_hold_steps", 6))

        close_success_now = bool((robot_to_ball_dist < close_dist_thresh).item())
        align_step_good = bool(
            (
                (robot_to_ball_dist < close_dist_thresh)
                & (line_reward > align_line_thresh)
                & (approach_reward > align_approach_thresh)
                & (robot_align_cos > align_robot_cos_thresh)
            ).item()
        )
        if align_step_good:
            self._align_hold_counter = int(getattr(self, "_align_hold_counter", 0)) + 1
        else:
            self._align_hold_counter = 0
        align_success_now = self._align_hold_counter >= max(1, align_hold_steps)
        touch_success_now = bool(self._has_touched_ball)

        v_para = torch.dot(ball_vel_xy, dir_bt)
        v_perp = torch.norm(ball_vel_xy - v_para * dir_bt)
        v_ratio = v_perp / (torch.abs(v_para) + 1e-6)
        stage2_kick_dir_thresh = torch.tensor(float(self._reward_stage_cfg.get("stage2_kick_dir_thresh", 0.45)), device=device, dtype=dtype)
        stage2_kick_vpara_thresh = torch.tensor(float(self._reward_stage_cfg.get("stage2_kick_vpara_thresh", 0.35)), device=device, dtype=dtype)
        stage2_kick_vratio_thresh = torch.tensor(float(self._reward_stage_cfg.get("stage2_kick_vratio_thresh", 0.80)), device=device, dtype=dtype)
        coarse_success_now = bool(touch_success_now) and bool(
            ((ball_align_cos > stage2_kick_dir_thresh) & (v_para > stage2_kick_vpara_thresh) & (v_ratio < stage2_kick_vratio_thresh)).item()
        )

        success_thresh = 0.60
        precise_success_now = bool((ball_dist < success_thresh).item())

        self._episode_close_success = self._episode_close_success or close_success_now
        self._episode_align_success = self._episode_align_success or align_success_now
        self._episode_touch_success = self._episode_touch_success or touch_success_now
        self._episode_coarse_success = self._episode_coarse_success or coarse_success_now
        self._episode_precise_success = self._episode_precise_success or precise_success_now

        stage_idx = int(self.reward_stage)
        pre_scale = torch.tensor(self._reward_stage_value("pre_touch_scale"), device=device, dtype=dtype)
        post_scale = torch.tensor(self._reward_stage_value("post_touch_scale"), device=device, dtype=dtype)
        touch_scale = torch.tensor(self._reward_stage_value("touch_bonus_scale"), device=device, dtype=dtype)
        align_scale = torch.tensor(self._reward_stage_value("align_bonus_scale"), device=device, dtype=dtype)
        success_scale = torch.tensor(self._reward_stage_value("success_bonus_scale"), device=device, dtype=dtype)

        # ========== 6. 分阶段奖励 ==========
        r_robot = torch.zeros((), device=device, dtype=dtype)
        r_approach = torch.zeros((), device=device, dtype=dtype)
        r_line = torch.zeros((), device=device, dtype=dtype)
        r_near_ball = torch.zeros((), device=device, dtype=dtype)
        r_ball = torch.zeros((), device=device, dtype=dtype)
        r_post_align = torch.zeros((), device=device, dtype=dtype)
        r_post_dist = torch.zeros((), device=device, dtype=dtype)
        r_progress = torch.zeros((), device=device, dtype=dtype)
        r_touch = torch.zeros((), device=device, dtype=dtype)
        r_succ = torch.zeros((), device=device, dtype=dtype)
        r_align_bonus = torch.zeros((), device=device, dtype=dtype)

        far_excess = torch.relu(robot_to_ball_dist - 3.0)
        far_penalty = 8.0 * (torch.exp(1.2 * far_excess) - 1.0)
        hack_gap = torch.relu(robot_align_reward - approach_reward)
        hack_penalty = 0.35 * hack_gap + 0.15 * tangentiality * robot_align_reward
        bad_touch_penalty = torch.zeros((), device=device, dtype=dtype)

        prev_ball_dist = getattr(self, "_prev_ball_dist", None)
        if prev_ball_dist is None:
            ball_progress = torch.zeros((), device=device, dtype=dtype)
        else:
            ball_progress = torch.clamp(prev_ball_dist - ball_dist, min=-0.30, max=0.30)

        if stage_idx == 0:
            # stage0: close_ball
            r_near_ball = pre_scale * 0.90 * torch.exp(-robot_to_ball_dist / 0.8)
            r_approach = pre_scale * 0.70 * approach_reward
            r_line = pre_scale * 0.15 * line_reward * line_dist_gate
            r_robot = pre_scale * 0.10 * robot_align_reward
            if first_touch:
                bad_touch_penalty = bad_touch_penalty + 0.40 * torch.relu(torch.tensor(0.55, device=device, dtype=dtype) - line_reward)
        elif stage_idx == 1:
            # stage1: align
            r_near_ball = pre_scale * 0.45 * torch.exp(-robot_to_ball_dist / 0.9)
            r_approach = pre_scale * 0.35 * approach_reward
            r_line = pre_scale * 0.80 * line_reward * line_dist_gate
            r_robot = pre_scale * 0.35 * robot_align_reward
            if align_success_now:
                r_align_bonus = 3.0 * align_scale
        elif stage_idx == 2:
            # stage2: touch
            if not self._has_touched_ball:
                r_near_ball = pre_scale * 0.45 * torch.exp(-robot_to_ball_dist / 0.9)
                r_approach = pre_scale * 0.35 * approach_reward
                r_line = pre_scale * 0.55 * line_reward * line_dist_gate
                r_robot = pre_scale * 0.25 * robot_align_reward
            if first_touch:
                r_touch = 8.0 * touch_scale
                kick_dir = torch.clamp(ball_align_cos, 0.0, 1.0)
                kick_speed = torch.relu(v_para)
                r_ball = 2.0 * kick_dir * kick_speed - 0.8 * v_perp
                bad_touch_penalty = bad_touch_penalty + 0.8 * torch.relu(stage2_kick_dir_thresh - ball_align_cos)
        else:
            # stage3: precise_pass
            speed_gate = torch.tanh(ball_speed / 1.5)
            r_post_align = post_scale * 1.20 * ball_align_reward * speed_gate
            r_post_dist = post_scale * 0.80 * torch.exp(-ball_dist / 0.9)
            r_progress = post_scale * 4.00 * ball_progress
            if first_touch:
                r_touch = 2.0 * touch_scale
            if precise_success_now:
                r_succ = 80.0 * success_scale

        # 通用成功奖励（仅最终成功）
        if precise_success_now and stage_idx < 3:
            r_succ = 20.0 * success_scale

        time_penalty = torch.tensor(0.01, device=device, dtype=dtype)
        fallen_penalty = torch.tensor(10.0 if self.extras.get("fall", False) else 0.0, device=device, dtype=dtype)

        reward = (
            r_robot
            + r_approach
            + r_line
            + r_near_ball
            + r_ball
            + r_post_align
            + r_post_dist
            + r_progress
            + r_touch
            + r_align_bonus
            + r_succ
            - far_penalty
            - hack_penalty
            - bad_touch_penalty
            - time_penalty
            - fallen_penalty
        )
        reward_raw = reward
        reward = torch.clamp(reward_raw, min=-100.0, max=100.0)

        # ========== 8. logging ==========
        if "rew_terms" not in self.extras:
            self.extras["rew_terms"] = {}
        terms = self.extras["rew_terms"]
        terms["ball_dist"] = ball_dist.detach()
        terms["ball_speed"] = ball_speed.detach()
        terms["ball_align_cos"] = ball_align_cos.detach()
        terms["ball_align_reward"] = ball_align_reward.detach()
        terms["robot_speed"] = robot_speed.detach()
        terms["robot_align_cos"] = robot_align_cos.detach()
        terms["robot_align_reward"] = robot_align_reward.detach()
        terms["robot_to_ball_dist"] = robot_to_ball_dist.detach()
        terms["line_cos"] = line_cos.detach()
        terms["line_reward"] = line_reward.detach()
        terms["line_dist_gate"] = line_dist_gate.detach()
        terms["approach_cos"] = approach_cos.detach()
        terms["approach_reward"] = approach_reward.detach()
        terms["v_para"] = v_para.detach()
        terms["v_perp"] = v_perp.detach()
        terms["v_ratio"] = v_ratio.detach()
        terms["align_step_good"] = torch.tensor(1.0 if align_step_good else 0.0, device=device, dtype=dtype)
        terms["align_hold_counter"] = torch.tensor(float(self._align_hold_counter), device=device, dtype=dtype)
        terms["align_hold_steps"] = torch.tensor(float(align_hold_steps), device=device, dtype=dtype)
        terms["touch_count"] = torch.tensor(float(touch_count), device=device, dtype=dtype)
        terms["touch_now"] = torch.tensor(1.0 if touch_now else 0.0, device=device, dtype=dtype)
        terms["first_touch"] = torch.tensor(1.0 if first_touch else 0.0, device=device, dtype=dtype)
        terms["r_robot"] = r_robot.detach()
        terms["r_approach"] = r_approach.detach()
        terms["r_line"] = r_line.detach()
        terms["r_near_ball"] = r_near_ball.detach()
        terms["r_ball"] = r_ball.detach()
        terms["r_post_align"] = r_post_align.detach()
        terms["r_post_dist"] = r_post_dist.detach()
        terms["r_progress"] = r_progress.detach()
        terms["r_touch"] = r_touch.detach()
        terms["r_align_bonus"] = r_align_bonus.detach()
        terms["r_succ"] = r_succ.detach()
        terms["far_penalty"] = (-far_penalty).detach()
        terms["hack_penalty"] = (-hack_penalty).detach()
        terms["side_hit_penalty"] = (-bad_touch_penalty).detach()
        terms["time_penalty"] = (-time_penalty).detach()
        terms["fallen_penalty"] = (-fallen_penalty).detach()
        terms["reward_raw"] = reward_raw.detach()
        terms["reward_clipped"] = reward.detach()
        terms["reward_stage"] = torch.tensor(float(self.reward_stage), device=device, dtype=dtype)
        terms["reward_stage_max"] = torch.tensor(float(self.max_reward_stage), device=device, dtype=dtype)
        terms["reward_stage_pre_scale"] = pre_scale.detach()
        terms["reward_stage_post_scale"] = post_scale.detach()
        terms["reward_stage_touch_scale"] = touch_scale.detach()
        terms["reward_stage_align_scale"] = align_scale.detach()
        terms["reward_stage_success_scale"] = success_scale.detach()
        terms["close_success"] = torch.tensor(1.0 if self._episode_close_success else 0.0, device=device, dtype=dtype)
        terms["align_success"] = torch.tensor(1.0 if self._episode_align_success else 0.0, device=device, dtype=dtype)
        terms["stage2_touch_success"] = torch.tensor(1.0 if self._episode_touch_success else 0.0, device=device, dtype=dtype)
        terms["coarse_success"] = torch.tensor(1.0 if self._episode_coarse_success else 0.0, device=device, dtype=dtype)
        terms["precise_success"] = torch.tensor(1.0 if self._episode_precise_success else 0.0, device=device, dtype=dtype)
        terms["stage_success_active"] = torch.tensor(1.0 if self._stage_success_flag(self.reward_stage) else 0.0, device=device, dtype=dtype)

        self.extras["success"] = bool(self._stage_success_flag(self.reward_stage))
        self.extras["final_success"] = bool(self._episode_precise_success)
        self.extras["hit"] = bool(first_touch)
        self.extras["reward_stage"] = int(self.reward_stage)
        self.extras["reward_stage_name"] = self._reward_stage_name()
        self.extras["reward_stage_max"] = int(self.max_reward_stage)
        self.extras["close_success"] = bool(self._episode_close_success)
        self.extras["align_success"] = bool(self._episode_align_success)
        self.extras["stage2_touch_success"] = bool(self._episode_touch_success)
        self.extras["coarse_success"] = bool(self._episode_coarse_success)
        self.extras["precise_success"] = bool(self._episode_precise_success)

        self._prev_ball_speed = ball_speed.detach()
        self._prev_ball_dist = ball_dist.detach()
        self._touch_now_prev = bool(touch_now)

        # ========== 9. 终端 debug 输出 ==========
        debug_flag = getattr(self, "debug_reward", False)
        if debug_flag:
            print(
                "[RewardDebug] "
                f"dist={float(ball_dist):.3f}, "
                f"robot_speed={float(robot_speed):.3f}, "
                f"robot_align={float(robot_align_reward):.3f}, "
                f"ball_speed={float(ball_speed):.3f}, "
                f"ball_align={float(ball_align_reward):.3f}, "
                f"r_robot={float(r_robot):.3f}, "
                f"r_approach={float(r_approach):.3f}, "
                f"r_line={float(r_line):.3f}, "
                f"r_near_ball={float(r_near_ball):.3f}, "
                f"r_post_align={float(r_post_align):.3f}, "
                f"r_post_dist={float(r_post_dist):.3f}, "
                f"r_progress={float(r_progress):.3f}, "
                f"far_penalty={float(far_penalty):.3f}, "
                f"hack_penalty={float(hack_penalty):.3f}, "
                f"r_touch={float(r_touch):.1f}, "
                f"r_succ={float(r_succ):.1f}, "
                f"fallen_penalty={float(fallen_penalty):.1f}, "
                f"raw_total={float(reward_raw):.3f}, "
                f"clipped_total={float(reward):.3f}"
            )

        return reward.view(1).to(device)

    def compute_midlevel_obs(self):
        """
        高层观测，返回 shape = (1, 11)

        各分量（全部是世界坐标系下的量）：
        0: base_x        机器人基座 x
        1: base_y        机器人基座 y
        2: ball_x        球的 x
        3: ball_y        球的 y
        4: v_world_x     机器人在世界系下的 vx
        5: v_world_y     机器人在世界系下的 vy
        6: target_x      target 的 x
        7: target_y      target 的 y
        8: rel_ball_x    球在机器人机体系下的相对 x
        9: rel_ball_y    球在机器人机体系下的相对 y
        10: rel_ball_ang 球在机器人机体系下的相对方位角 atan2(y, x)
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

        # ----- 球相对机器人（机体系） -----
        delta_world = torch.zeros(3, device=device, dtype=dtype)
        delta_world[0] = ball_x - base_x
        delta_world[1] = ball_y - base_y
        delta_body = quat_rotate_inverse(self.base_quat[0:1], delta_world[None, :]).squeeze(0)
        rel_ball_x = delta_body[0]
        rel_ball_y = delta_body[1]
        rel_ball_ang = torch.atan2(rel_ball_y, rel_ball_x + 1e-6)

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
            rel_ball_x,
            rel_ball_y,
            rel_ball_ang,
        ], dim=0).to(device=device, dtype=dtype)    # (11,)

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
        # 首个 episode 或重置边界时，initial_dist_xy 可能尚未写入。
        # 回退到当前实时距离，避免上层采样/日志崩溃。
        if self.initial_dist_xy is None:
            return float(self.get_dist_xy())
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

    def compute_midlevel_reward_with_goal(self, obs_high, goal_xy, include_success=False):
        """
        HER专用：根据给定的goal_xy（虚拟target），重新计算高层奖励。
        obs_high: shape (8,) 或 (1,8)
        goal_xy: shape (2,) numpy or torch
        """
        device = self.controller.device
        dtype = self.base_pos.dtype
        # 解析观测
        if obs_high.ndim == 2:
            obs_high = obs_high.squeeze(0)
        base_x, base_y = obs_high[0], obs_high[1]
        ball_x, ball_y = obs_high[2], obs_high[3]
        v_world_x, v_world_y = obs_high[4], obs_high[5]
        # 使用传入的goal替换target
        if isinstance(goal_xy, np.ndarray):
            goal_xy = torch.tensor(goal_xy, dtype=dtype, device=device)
        target_xy = goal_xy
        # 机器人/球位姿
        robot_pos = torch.tensor([base_x, base_y], dtype=dtype, device=device)
        ball_xy = torch.tensor([ball_x, ball_y], dtype=dtype, device=device)
        v_world_xy = torch.tensor([v_world_x, v_world_y], dtype=dtype, device=device)
        # 1. 球->target
        delta_bt = target_xy - ball_xy
        ball_dist = torch.norm(delta_bt) + 1e-6
        dir_bt = delta_bt / ball_dist
        # 2. 机器人朝向/速度几何（与主奖励保持一致）
        delta_rb = ball_xy - robot_pos
        robot_to_ball_dist = torch.norm(delta_rb) + 1e-6
        dir_rb = delta_rb / robot_to_ball_dist

        # 面向球（身体朝向）
        fwd_local = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        fwd_world = quat_rotate(self.base_quat[0:1], fwd_local[None, :]).squeeze(0)
        fwd_world_xy = fwd_world[:2]
        fwd_norm = torch.norm(fwd_world_xy)
        if fwd_norm > 1e-6:
            fwd_dir = fwd_world_xy / (fwd_norm + 1e-6)
            robot_align_cos = torch.clamp(torch.dot(fwd_dir, dir_rb), -1.0, 1.0)
            robot_align_reward = torch.clamp(robot_align_cos, 0.0, 1.0)
        else:
            robot_align_cos = torch.tensor(0.0, device=device, dtype=dtype)
            robot_align_reward = torch.tensor(0.0, device=device, dtype=dtype)

        # 三点共线 + 朝球前进
        line_cos = torch.clamp(torch.dot(dir_rb, dir_bt), -1.0, 1.0)
        line_reward = torch.clamp(line_cos, 0.0, 1.0)
        line_dist_gate = torch.exp(-robot_to_ball_dist / 1.5)

        robot_speed = torch.norm(v_world_xy)
        robot_move_thresh = torch.tensor(0.1, device=device, dtype=dtype)
        if robot_speed > robot_move_thresh:
            v_dir = v_world_xy / (robot_speed + 1e-6)
            approach_cos = torch.clamp(torch.dot(v_dir, dir_rb), -1.0, 1.0)
            approach_reward = 0.5 * (approach_cos + 1.0)
            tangentiality = torch.sqrt(torch.clamp(1.0 - approach_cos * approach_cos, min=0.0, max=1.0))
        else:
            approach_cos = torch.tensor(0.0, device=device, dtype=dtype)
            approach_reward = torch.tensor(0.0, device=device, dtype=dtype)
            tangentiality = torch.tensor(0.0, device=device, dtype=dtype)
        # 3. HER时无法可靠恢复触球事件，统一不给事件奖励
        r_touch = torch.tensor(0.0, device=device, dtype=dtype)
        # 4. 成功判定
        success_thresh = 0.60
        success = bool((ball_dist < success_thresh).item())
        # HER 默认不注入大额成功奖励，避免 replay 被高奖励虚拟样本主导
        r_succ = torch.tensor(
            80.0 if (include_success and success) else 0.0,
            device=device,
            dtype=dtype,
        )
        # 5. 时间/摔倒惩罚
        time_penalty = torch.tensor(0.01, device=device, dtype=dtype)
        fallen_penalty = torch.tensor(0.0, device=device, dtype=dtype)  # HER无法判断
        # 6. shaping：HER 与主 reward 保持同一倾向，但不依赖事件历史
        r_robot = 0.05 * robot_align_reward
        r_approach = 0.45 * approach_reward
        r_line = 0.45 * line_reward * line_dist_gate
        r_near_ball = 0.50 * torch.exp(-robot_to_ball_dist / 0.8)
        r_post_align = torch.zeros((), device=device, dtype=dtype)
        r_post_dist = 0.80 * torch.exp(-ball_dist / 0.9)
        r_progress = torch.zeros((), device=device, dtype=dtype)
        far_excess = torch.relu(robot_to_ball_dist - 3.0)
        far_penalty = 12.0 * (torch.exp(1.4 * far_excess) - 1.0)
        hack_gap = torch.relu(robot_align_reward - approach_reward)
        hack_penalty = 0.45 * hack_gap + 0.20 * tangentiality * robot_align_reward
        r_ball = torch.zeros((), device=device, dtype=dtype)
        reward = (
            r_robot
            + r_approach
            + r_line
            + r_near_ball
            + r_ball
            + r_post_align
            + r_post_dist
            + r_progress
            + r_touch
            + r_succ
            - far_penalty
            - hack_penalty
            - time_penalty
            - fallen_penalty
        )
        reward = torch.clamp(reward, min=-100.0, max=100.0)
        return float(reward.item())

    def should_early_stop(self):
        """
        判断是否可以早停：检测到球和机器人有碰撞（已触球）后，等待球运动一段时间再估算速度。
        返回: (should_stop: bool, will_succeed: bool)
        """
        # 只要 self._has_touched_ball 为 True 就认为已触球
        if getattr(self, "_has_touched_ball", False):
            # 预留一段球运动的时间，避免刚触球时速度估算不准
            if not hasattr(self, "_touch_step_counter"):
                self._touch_step_counter = self.common_step_counter
            steps_since_touch = self.common_step_counter - self._touch_step_counter
            min_steps = 100  # 例如至少等待20步
            if steps_since_touch < min_steps:
                return False, False
            will_succeed, _, _, _ = self._predict_ball_outcome_from_current_state()
            return True, will_succeed
        else:
            # 没有触球则重置计数器
            if hasattr(self, "_touch_step_counter"):
                del self._touch_step_counter
            # 用底层刚体碰撞检测方法判断是否刚刚触球
            just_touched = self.is_feet_contact_ball()  # 调用新的距离检测方法
            if just_touched:
                self._has_touched_ball = True
                self._touch_step_counter = self.common_step_counter
            return False, False

    def _predict_ball_outcome_from_current_state(self):
        """
        基于当前球位置/速度，按线性阻尼模型估算球的最大前向位移并预测是否会进入成功半径。
        返回: (will_succeed: bool, final_dist: float, s_max: float, v0: float)
        """
        ball_pos, ball_lin_vel, _ = self.ball_world.get_pose(self.root_states)
        ball_xy = ball_pos[:2]
        ball_vel_xy = ball_lin_vel[:2]
        target_xy = self.target_xy
        delta_bt = target_xy - ball_xy
        dir_bt = delta_bt / (torch.norm(delta_bt) + 1e-6)

        dyn = {}
        if hasattr(self.ball_world, "get_dynamics"):
            dyn = self.ball_world.get_dynamics()
        linear_damping = float(dyn.get("linear_damping", 0.015))
        ball_density = float(dyn.get("density", 80.0))
        ball_radius = float(dyn.get("radius", 0.11))

        ball_volume = (4.0 / 3.0) * np.pi * (ball_radius ** 3)
        ball_mass = ball_density * ball_volume
        v0 = torch.norm(ball_vel_xy).item()
        s_max = v0 * ball_mass / (linear_damping + 1e-6)
        ball_final_xy = ball_xy + dir_bt * s_max
        final_dist = torch.norm(target_xy - ball_final_xy).item()
        success_thresh = 0.60
        will_succeed = final_dist < success_thresh
        return bool(will_succeed), float(final_dist), float(s_max), float(v0)

    def predict_success_after_hit(self):
        """
        触球当下调用：不等待额外步数，直接用当前球速预测后续轨迹是否会进入成功区域。
        返回: (will_succeed: bool, final_dist: float, s_max: float, v0: float)
        """
        return self._predict_ball_outcome_from_current_state()

    def is_feet_contact_ball(self):
        """
        通过距离检测检查脚部是否与球接触。
        返回: bool
        """
        # 获取脚部的坐标
        foot_positions = self.body_states[:, self.controller.feet_indices, 0:3]

        # 获取球的位置，假设球是最后一个刚体
        ball_position = self.body_states[:, self.controller.num_bodies_robot, 0:3]  # 球的坐标，假设是最后一个刚体

        # 计算脚部与球的距离
        distances = torch.norm(foot_positions - ball_position, dim=2)

        # 设置一个阈值，判断是否接触
        contact_threshold = 0.25  # 你可以根据实际情况调整这个值

        # 检查是否有脚部与球的距离小于阈值
        if torch.any(distances < contact_threshold):
            return True
        
        return False
