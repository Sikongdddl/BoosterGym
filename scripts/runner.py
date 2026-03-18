import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import torch

from utils.model import ActorCritic
from envs import *
from core.agents.sac.agent import SACAgent
from core.utils.logger import TBLogger

class Runner:

    def __init__(self, test=False, task_name=None):
        self.test = test
        # CLI + CFG
        self._get_args()
        self._update_cfg_from_args(task_name)
        self._set_seed()
        # env
        task_class = eval(self.cfg["basic"]["task"])
        dummy_target = torch.zeros(2, device=self.cfg["basic"]["rl_device"], dtype=torch.float32)
        self.env = task_class(self.cfg, dummy_target)
        # device
        self.device = self.cfg["basic"]["rl_device"]
        # low_level locomotion model(already trained)
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--algo", type=str, choices=["dqn", "sac"], default="sac", help="High-level RL algorithm: dqn or sac.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self, task_name):
        cfg_file = os.path.join("envs", task_name,f"{self.args.task}.yaml")
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        self.cfg["basic"].setdefault("algo", "sac")
        self.cfg["basic"].setdefault("curriculum_window", 50)     # 最近 N 个 episode 统计成功率
        self.cfg["basic"].setdefault("eval_every_episodes", 100)  # 每 N 个 episode 评估
        self.cfg["basic"].setdefault("respawn_r_min", 2.0)
        self.cfg["basic"].setdefault("respawn_r_max", 8.0)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        ckpt = self.cfg["basic"].get("checkpoint",None)
        if not ckpt:
            return
        if ckpt == "-1" or ckpt == -1:
            all_ckpts = sorted(glob.glob(os.path.join("logs/low", "**/*.pth"), recursive=True), key=os.path.getmtime)
            if not all_ckpts:
                print("[WARN] No checkpoint found under logs/**.pth; skip loading.")
                return
            ckpt = all_ckpts[-1]
            self.cfg["basic"]["checkpoint"] = ckpt
        print(f"Loading low-level model from {ckpt}")
        model_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)

    def _set_target_from_ball(self):
        x,_,_ = self.env.ball_world.get_pose(self.env.root_states)
        self.env.target_xy = x[:2].to(self.device)

    def _set_target_for_pass(self):
        """
        为 passBall 采样一个新的 target_xy 并写进 env。
        采样规则：
        - 方向：前方小扇形内
        - 距离：使用 curriculum 当前窗口 [cur_r_min, cur_r_max]
        """
        device = self.device

        # 1) 取机器人当前位置 (世界系)
        base_pos = self.env.base_pos[0, :3]   # tensor
        base_x = float(base_pos[0].item())
        base_y = float(base_pos[1].item())

        # 2) 方向：先控制在前方 ±30° 内（后面可以做成 curriculum）
        # 这里假设 +x 是机器人“朝前”的世界系方向
        THETA_MAX = np.deg2rad(30.0)   # 30 度
        theta = np.random.uniform(-THETA_MAX, THETA_MAX)
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)

        # 3) 距离：按课程窗口采样，确保难度变化真实生效到任务生成
        r_min = float(getattr(self.env, "cur_r_min", 1.0))
        r_max = float(getattr(self.env, "cur_r_max", 1.5))
        if r_max < r_min:
            r_min, r_max = r_max, r_min
        R = np.random.uniform(r_min, r_max)

        tx = base_x + R * dir_x
        ty = base_y + R * dir_y

        target_xy = torch.tensor([tx, ty], dtype=self.env.base_pos.dtype, device=device)
        self.env.target_xy = target_xy

        print(
            f"[Target] R={R:.2f} in [{r_min:.2f}, {r_max:.2f}], "
            f"theta={theta:.2f} rad, target=({tx:.2f}, {ty:.2f})"
        )
        
    def _build_high_agent(self):
        """根据 algo 构建高层 agent。"""
        algo = self.cfg["basic"]["algo"].lower()
        obs_high = self.env.compute_midlevel_obs().to(self.device)
        state_dim = int(obs_high.shape[1])
        if algo == "sac":
            # 连续动作：这里用 3 维（vx, vy, yaw），步频固定；也可扩成 4 维把步频也学出来
            action_dim = 3
            # 从 cfg 读高层命令的物理范围；提供安全缺省
            cmd_cfg = self.env.controller.cfg.get("commands", {})
            vx_range = np.array([0.0, 0.6], dtype=np.float32)  # 允许只前进：min=0.0
            vy_range = np.array([-0.35, 0.35], dtype=np.float32)
            yaw_range = np.array([-1.0, 1.0], dtype=np.float32)
            act_low = np.array([vx_range[0], vy_range[0], yaw_range[0]], dtype=np.float32)
            act_high = np.array([vx_range[1], vy_range[1], yaw_range[1]], dtype=np.float32)

            agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device=self.device,
                action_low=act_low,
                action_high=act_high,
                buffer_capacity=200000,
                batch_size=256,
                lr=3e-4,
                gamma=0.90,
                tau=0.005,
                alpha=None,  # 自动温度
            )
            action_mode = "continuous"
        else:
            raise ValueError(f"Unknown algo: {algo}")

        return agent, action_mode

    def _apply_high_level_cmd(self, action_mode, agent, obs_high):
        """根据动作模式（离散/连续）生成并下发高层命令，返回（记录用的）动作表示。"""
        obs_high_np = obs_high.squeeze(0).cpu().numpy()

        if action_mode == "discrete":
            # DQN：选动作 id
            action_id = agent.select_action(obs_high_np)
            cmd = self.env.high_level_action_id_to_vector(action_id)
            # 固定步频或由表中带出
            self.env.apply_high_level_command(cmd)
            return ("discrete", action_id, cmd)

        else:
            # SAC：输出连续 [vx, vy, yaw]，步频固定
            a_cont = agent.select_action(obs_high_np, eval_mode=False)
            gait_freq = 1.5
            cmd = [float(a_cont[0]), float(a_cont[1]), float(a_cont[2]), gait_freq]
            self.env.apply_high_level_command(cmd, smooth=0.5)
            return ("continuous", a_cont, cmd)

    def chaseBall(self):
        # tensorboard logger
        run_name = f"{self.cfg['basic']['task']}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb = TBLogger(
            logdir="logs/tb",
            run_name=run_name
        )
        global_step = 0
        
        # ----- 可调参数 -----
        ACTION_REPEAT = 10  # <<< 高层动作重复次数，建议先试 5~10
        WARMUP = 5000   # <<< 低层模型 warmup 步数
        UPDATE_K = 1    # <<< 低层模型更新频率

        # init
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        # 更新目标位置
        self._set_target_from_ball()

        # build high-level agent
        agent, action_mode = self._build_high_agent()
        
        episode_step = 0
        episode_return = 0
        max_steps = 500
        episode_idx = 0
        EP_RATE_WIN = 200
        ep_success_window = []

        try:
            while True:
                # ---------- 高层观测 ----------
                obs_high = self.env.compute_midlevel_obs().to(self.device)

                # ---------- 下发高层命令 ----------
                mode, action_repr, action_cmd = self._apply_high_level_cmd(action_mode, agent, obs_high)

                # 若是 DQN，额外记录 Q 值（不影响选择逻辑）
                q_max = q_mean = q_selected = None
                if mode == "discrete" and hasattr(agent, "q_net"):
                    with torch.no_grad():
                        q_vals = agent.q_net(
                            torch.as_tensor(
                                obs_high.squeeze(0).cpu().numpy(), 
                                dtype=torch.float32, 
                                device=self.device
                                ).unsqueeze(0))
                        q_max = float(q_vals.max().item())
                        q_mean = float(q_vals.mean().item())
                        q_selected = float(q_vals[0, int(action_repr)].item())

                # ---------- 低层滚动（动作重复） ----------
                acc_rew_high = 0.0
                last_infos = infos
                success_happened = False
                fall_happened = False
                tb.set_step(global_step)
                tb.add_scalar("train/env_frames", global_step * ACTION_REPEAT)

                for _ in range(ACTION_REPEAT):
                    with torch.no_grad():
                        obs_mod = obs.clone()
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = (
                            action_cmd[0], action_cmd[1], action_cmd[2]
                        )
                        dist = self.model.act(obs_mod)
                        act = dist.loc
                        obs, rew, done, infos = self.env.step(act)
                        obs = obs.to(self.device)
                        last_infos = infos

                    step_rew_high = float(rew)
                    acc_rew_high += step_rew_high

                    if isinstance(infos, dict) and infos.get("fall", False):
                        fall_happened = True
                        tb.add_scalar("events/fallen", 1.0)
                        break

                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True
                        tb.add_scalar("events/success", 1.0)
                        break

                    if isinstance(infos, dict) and infos.get("hit", False):
                        hit_happened = True
                        tb.add_scalar("events/hit", 1.0)
                        break

                    if torch.any(done).item():
                        break
                # ---------- 高层一步的转移 ----------
                next_obs_high = self.env.compute_midlevel_obs().to(self.device)
                next_obs_high_np = next_obs_high.squeeze(0).cpu().numpy()
                rew_high = acc_rew_high / ACTION_REPEAT

                episode_step += 1
                episode_return += rew_high
                done_high = (episode_step > max_steps) or success_happened or fall_happened

                # 经验入池
                if mode == "discrete":
                    agent.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        int(action_repr),   # 动作 id
                        rew_high,
                        next_obs_high_np,
                        done_high
                    )
                else:
                    agent.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        np.asarray(action_repr, dtype=np.float32),  # 连续动作
                        rew_high,
                        next_obs_high_np,
                        done_high
                    )

                # ---------- TensorBoard ----------
                if mode == "discrete":
                    tb.add_scalar("dqn/epsilon", getattr(agent, "epsilon", 0.0))
                    tb.add_scalar("dqn/update_steps", getattr(agent, "step_count", 0))
                    tb.add_scalar("high/action_id", int(action_repr))
                    if q_max is not None: tb.add_scalar("dqn/q_max", q_max)
                    if q_mean is not None: tb.add_scalar("dqn/q_mean", q_mean)
                    if q_selected is not None: tb.add_scalar("dqn/q_selected", q_selected)
                else:
                    if hasattr(agent, "log_alpha"):
                        tb.add_scalar("sac/alpha", float(agent.log_alpha.exp().item()))
                    tb.add_scalar("high/action_vx", float(action_cmd[0]))
                    tb.add_scalar("high/action_vy", float(action_cmd[1]))
                    tb.add_scalar("high/action_yaw", float(action_cmd[2]))

                tb.add_scalar("train/replay_size", len(agent.replay_buffer))
                tb.add_scalar("high/reward", rew_high)

                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        if "dist_xy" in terms:
                            tb.add_scalar("rew/dist_xy", float(terms["dist_xy"]))
                        if "heading_cos" in terms:
                            tb.add_scalar("rew/heading_cos", float(terms["heading_cos"]))
                        if "progress_gain" in terms:
                            tb.add_scalar("rew/progress_gain", float(terms["progress_gain"]))
                        if "speed_toward" in terms:
                            tb.add_scalar("rew/speed_toward", float(terms["speed_toward"]))
                        if "speed_orth" in terms:
                            tb.add_scalar("rew/speed_orth", float(terms["speed_orth"]))
                        if "spin_penalty" in terms:
                            tb.add_scalar("rew/spin_penalty", float(terms["spin_penalty"]))
                global_step += 1

                # ---------- 定期保存模型 ----------
                if (global_step % 10000 == 0) and (len(agent.replay_buffer) >= WARMUP):
                    os.makedirs(os.path.join("logs", "ckpt"), exist_ok=True)
                    ckpt_path = os.path.join("logs", "ckpt", f"sac_agent_step_{global_step}.pt")
                    torch.save({"agent": agent, "global_step": global_step, "cfg": self.cfg}, ckpt_path)
                    print(f"[Save] SAC agent saved at step {global_step} -> {ckpt_path}")
                # ---------- 回合结束 ----------
                if done_high:
                    succ = 1.0 if (success_happened and not fall_happened) else 0.0
                    ep_success_window.append(succ)
                    if len(ep_success_window) > EP_RATE_WIN:
                        ep_success_window.pop(0)
                    succ_rate = float(sum(ep_success_window) / len(ep_success_window))
                    tb.add_scalar("episode/return", float(episode_return))
                    tb.add_scalar("episode/length", float(episode_step))
                    tb.add_scalar("episode/success", float(succ))
                    tb.add_scalar("episode/success_rate", succ_rate)
                    tb.add_scalar("episode/init_dist", float(self.env.get_initial_dist_xy()))
                    tb.add_scalar("episode/cur_r_max", float(self.env.cur_r_max))
                    print(f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                          f"Step: {episode_step} | Success: {bool(succ)} | "
                          f"InitDist: {self.env.get_initial_dist_xy():.2f}")

                    # === 新：基于“单回合结果”的课程更新（env 内部处理防连跳/冷却/驻留/比例步长） ===
                    try:
                        rmin, rmax, changed, info = self.env.on_episode_end(
                            success=(success_happened and not fall_happened),
                            episode_idx=episode_idx
                        )
                        # 记录关键课程指标（最少也把 r_max 记上）
                        tb.add_scalar("curr/r_max", float(rmax))
                        if isinstance(info, dict):
                            if "rate_global" in info: tb.add_scalar("curr/rate_global", float(info["rate_global"]))
                            if "rate_curr" in info: tb.add_scalar("curr/rate_curr", float(info["rate_curr"]))
                            tb.add_scalar("curr/changed", 1.0 if changed else 0.0)
                        if changed and isinstance(info, dict) and info.get("reason"):
                            print(f"[Curriculum] ep#{episode_idx} r_max -> {rmax:.2f} | {info['reason']}")
                    except Exception as e:
                        print("[Curriculum] update failed:", e)

                    episode_idx += 1
                    episode_step = 0
                    episode_return = 0.0
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)
                    # 更新目标位置
                    self._set_target_from_ball()
                    
                # ---------- 更新 ----------
                if len(agent.replay_buffer) >= WARMUP:
                    for _ in range(UPDATE_K):
                        did_update, q1_loss, q2_loss, pi_loss, alpha_loss, alpha = agent.update()
                        if did_update:
                            tb.add_scalar("train/q1_loss", q1_loss)
                            tb.add_scalar("train/q2_loss", q2_loss)
                            tb.add_scalar("train/policy_loss", pi_loss)
                            tb.add_scalar("train/alpha_loss", alpha_loss)
                            tb.add_scalar("train/alpha", alpha)

                # 可选：周期性 flush，防 TensorBoard 不刷盘
                if global_step % 200 == 0:
                    try: tb.flush()
                    except: pass

        finally:
            tb.close()
    
    def passBall(self):
        # tensorboard logger
        run_name = f"{self.cfg['basic']['task']}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb = TBLogger(
            logdir="logs/tb",
            run_name=run_name
        )
        global_step = 0

        # ----- 可调参数 -----
        ACTION_REPEAT = 10  # 高层动作重复次数
        WARMUP = 5000       # 高层 agent warmup 步数
        UPDATE_K = 1        # 每个高层 step 更新次数

        # ----- 性能计时（不会影响训练逻辑；仅在间隔步做 CUDA 同步计时）-----
        PROFILE_EVERY = 50  # 每 N 个 high-level step 记录一次更精确的 GPU 时间；其余步仅做轻量 CPU 计时
        _use_cuda_timing = torch.cuda.is_available() and (str(self.device).startswith("cuda"))
        def _cpu_ms(t0: float) -> float:
            return (time.perf_counter() - t0) * 1000.0

        def _cuda_ms(fn):
            """对 GPU-heavy 代码段计时：仅在 global_step%PROFILE_EVERY==0 时启用，避免频繁 synchronize 影响吞吐。"""
            if (not _use_cuda_timing) or (global_step % PROFILE_EVERY != 0):
                return fn(), None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            return out, float(start.elapsed_time(end))  # milliseconds


        # 简单滑动平均，避免日志噪声
        _ema = {}
        def _ema_update(key: str, value: float, beta: float = 0.95) -> float:
            if value is None:
                return _ema.get(key, None)
            if key not in _ema:
                _ema[key] = value
            else:
                _ema[key] = beta * _ema[key] + (1.0 - beta) * value
            return _ema[key]

        # 初始化目标位置 & 环境
        obs, infos = self.env.reset()
        obs = obs.to(self.device)

        self._set_target_for_pass()

        # build high-level agent
        agent, action_mode = self._build_high_agent()

        episode_step = 0
        episode_return = 0.0
        max_steps = 300
        episode_idx = 0
        hit_happened = False        # 记录是否已触球
        before_hit_indices = []     # 记录本回合所有 before hit 的 buffer 索引
        EP_RATE_WIN = 200
        ep_success_window = []

        try:
            while True:
                # ---------- 高层观测 ----------
                _t0 = time.perf_counter()
                obs_high = self.env.compute_midlevel_obs()
                # 避免不必要的 .to(device)（通常 obs_high 已在 rl_device 上）
                if torch.is_tensor(obs_high) and obs_high.device != torch.device(self.device):
                    obs_high = obs_high.to(self.device)
                obs_ms = _cpu_ms(_t0)
                _ema_update("high_obs_ms", obs_ms)
 
                # ---------- 下发高层命令 ----------
                mode, action_repr, action_cmd = self._apply_high_level_cmd(action_mode, agent, obs_high)

                # ---------- 低层滚动（动作重复） ----------
                _t_rollout = time.perf_counter()
                last_infos = infos
                success_happened = False
                fall_happened = False
                tb.set_step(global_step)
                tb.add_scalar("train/env_frames", global_step * ACTION_REPEAT)

                # 累计 reward：避免在循环内反复 float(rew) 触发同步
                acc_rew_high_t = None  # torch scalar on device
                acc_rew_high = 0.0     # python fallback

                # 避免每次都分配 obs_mod：预分配一次，然后 copy_ 覆盖（语义等价于 clone）
                obs_mod = obs.clone()


                for _ in range(ACTION_REPEAT):
                    with torch.no_grad():
                        # obs_mod <- obs，然后覆盖 command 维度
                        obs_mod.copy_(obs)
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = (
                            action_cmd[0], action_cmd[1], action_cmd[2]
                        )

                        # 低层 policy forward（可选 CUDA 计时）
                        def _low_act():
                            dist = self.model.act(obs_mod)
                            return dist.loc
                        act, low_act_ms = _cuda_ms(_low_act)
                        if low_act_ms is not None:
                            _ema_update("low_act_ms", low_act_ms)

                        # env.step（可选 CUDA 计时）
                        def _env_step():
                            return self.env.step(act)
                        (obs, rew, done, infos), env_step_ms = _cuda_ms(_env_step)
                        if env_step_ms is not None:
                            _ema_update("env_step_ms", env_step_ms)

                        if torch.is_tensor(obs) and obs.device != torch.device(self.device):
                            obs = obs.to(self.device)
                        last_infos = infos

                    # reward 累加（尽量留在 torch）
                    if torch.is_tensor(rew):
                        if acc_rew_high_t is None:
                            acc_rew_high_t = torch.zeros((), device=rew.device, dtype=rew.dtype)
                        acc_rew_high_t = acc_rew_high_t + rew.reshape(-1)[0]
                    else:
                        acc_rew_high += float(rew)

                    if isinstance(infos, dict) and infos.get("fall", False):
                        fall_happened = True
                        tb.add_scalar("events/fallen", 1.0)
                        break

                    # ✅ success 判定：完全依赖 env.extras["success"]
                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True
                        tb.add_scalar("events/success", 1.0)
                        break

                    # 新增：检测触球事件
                    if isinstance(infos, dict) and infos.get("hit", False):
                        hit_happened = True
                        tb.add_scalar("events/hit", 1.0)
                        # 触球即早停：基于“触球当下速度”预测后续轨迹是否会进入成功半径
                        if hasattr(self.env, "predict_success_after_hit"):
                            will_succeed, final_dist, s_max, v0 = self.env.predict_success_after_hit()
                            hit_pred_success = bool(will_succeed)
                            tb.add_scalar("events/hit_pred_success", float(hit_pred_success))
                            tb.add_scalar("diag/hit_pred_final_dist", float(final_dist))
                            tb.add_scalar("diag/hit_pred_s_max", float(s_max))
                            tb.add_scalar("diag/hit_pred_v0", float(v0))
                            if hit_pred_success:
                                success_happened = True
                                tb.add_scalar("events/success", 1.0)
                        break

                    if torch.any(done).item():
                        break


                # rollout 总耗时（CPU 粗计时；EMA）
                rollout_ms = _cpu_ms(_t_rollout)
                _ema_update("rollout_ms", rollout_ms)

                # rollout reward 汇总（只在这里触发一次 .item() 同步）
                if acc_rew_high_t is not None:
                    rew_high = float((acc_rew_high_t / ACTION_REPEAT).item())
                else:
                    rew_high = acc_rew_high / ACTION_REPEAT

                # ---------- 高层一步的转移 ----------
                _t1 = time.perf_counter()
                next_obs_high = self.env.compute_midlevel_obs()
                if torch.is_tensor(next_obs_high) and next_obs_high.device != torch.device(self.device):
                    next_obs_high = next_obs_high.to(self.device)
                next_obs_ms = _cpu_ms(_t1)
                _ema_update("next_high_obs_ms", next_obs_ms)

                # 为保持现有 replay_buffer 接口（numpy），这里仍转 numpy；同时记录开销
                _t_np = time.perf_counter()
                next_obs_high_np = next_obs_high.squeeze(0).detach().cpu().numpy()
                obs_high_np = obs_high.squeeze(0).detach().cpu().numpy()
                to_numpy_ms = _cpu_ms(_t_np)
                _ema_update("to_numpy_ms", to_numpy_ms)

                episode_step += 1
                episode_return += rew_high
                done_high = (episode_step > max_steps) or success_happened or fall_happened or hit_happened

                # 经验入池，带 note
                note = "after hit" if hit_happened else "before hit"
                _t_push = time.perf_counter()
                if mode == "discrete":
                    agent.replay_buffer.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        int(action_repr),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note=note,
                    )
                else:
                    agent.replay_buffer.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        np.asarray(action_repr, dtype=np.float32),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note=note,
                    )
                # before hit transition 单独记录，供“成功回溯加奖”和 HER 采样
                if note == "before hit":
                    buf_idx = len(agent.replay_buffer) - 1
                    before_hit_indices.append(buf_idx)

                push_ms = _cpu_ms(_t_push)
                _ema_update("push_ms", push_ms)                
                
                # ---------- TensorBoard：高层动作 & 算法状态 ----------

                if hasattr(agent, "log_alpha"):
                    tb.add_scalar("sac/alpha", float(agent.log_alpha.exp().item()))
                tb.add_scalar("high/action_vx", float(action_cmd[0]))
                tb.add_scalar("high/action_vy", float(action_cmd[1]))
                tb.add_scalar("high/action_yaw", float(action_cmd[2]))

                tb.add_scalar("train/replay_size", len(agent.replay_buffer))
                tb.add_scalar("high/reward", rew_high)

                # ---------- TensorBoard：reward 分解（与 PassBallEnv.rew_terms 保持一致） ----------
                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        # PassBallEnv 当前输出字段（reward 相关）
                        rew_keys = (
                            "ball_dist",
                            "ball_speed",
                            "ball_align_cos",
                            "ball_align_reward",
                            "robot_speed",
                            "robot_align_cos",
                            "robot_align_reward",
                            "robot_to_ball_dist",
                            "line_cos",
                            "line_reward",
                            "line_dist_gate",
                            "approach_cos",
                            "approach_reward",
                            "r_robot",
                            "r_line",
                            "r_near_ball",
                            "r_ball",
                            "r_touch",
                            "r_succ",
                            "far_penalty",
                            "hack_penalty",
                            "time_penalty",
                            "fallen_penalty",
                        )
                        for k in rew_keys:
                            if k in terms:
                                tb.add_scalar(f"rew/{k}", float(terms[k]))

                        # PassBallEnv 当前输出字段（事件相关）
                        event_keys = (
                            "touch_now",
                            "first_touch",
                        )
                        for k in event_keys:
                            if k in terms:
                                tb.add_scalar(f"events/{k}", float(terms[k]))
                global_step += 1

                # ---------- 时间剖析（EMA，减少日志开销） ----------
                if global_step % PROFILE_EVERY == 0:
                    v = _ema.get("high_obs_ms", None)
                    if v is not None: tb.add_scalar("time/high_obs_ms", v)
                    v = _ema.get("next_high_obs_ms", None)
                    if v is not None: tb.add_scalar("time/next_high_obs_ms", v)
                    v = _ema.get("rollout_ms", None)
                    if v is not None: tb.add_scalar("time/rollout_ms", v)
                    v = _ema.get("low_act_ms", None)
                    if v is not None: tb.add_scalar("time/low_act_ms", v)
                    v = _ema.get("env_step_ms", None)
                    if v is not None: tb.add_scalar("time/env_step_ms", v)
                    v = _ema.get("to_numpy_ms", None)
                    if v is not None: tb.add_scalar("time/to_numpy_ms", v)
                    v = _ema.get("push_ms", None)
                    if v is not None: tb.add_scalar("time/push_ms", v)
                    v = _ema.get("update_ms", None)
                    if v is not None: tb.add_scalar("time/update_ms", v)

                # ---------- ✅ 每 10k step dump 一份 replay ----------
                if global_step % 10000 == 0 and len(agent.replay_buffer) > 0:  # <<<
                    try:
                        agent.replay_buffer.save_to_disk(
                            save_dir="logs/replay",
                            filename=f"replay_step_{global_step}_N{len(agent.replay_buffer)}.npz"
                        )
                    except Exception as e:
                        print("[ReplayBuffer] save failed:", e)


                # ---------- 定期保存模型 ----------
                if (global_step % 10000 == 0) and (len(agent.replay_buffer) >= WARMUP):
                    ckpt_dir = os.path.join("logs", "ckpt", "passBall", "sac")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"sac_agent_step_{global_step}.pt")
                    torch.save({"agent": agent, "global_step": global_step, "cfg": self.cfg}, ckpt_path)
                    print(f"[Save] SAC agent saved at step {global_step} -> {ckpt_path}")

                # ---------- 回合结束 ----------
                if done_high:
                    succ = 1.0 if (success_happened and not fall_happened) else 0.0

                    ep_success_window.append(succ)
                    if len(ep_success_window) > EP_RATE_WIN:
                        ep_success_window.pop(0)
                    succ_rate = float(sum(ep_success_window) / len(ep_success_window))
                    tb.add_scalar("episode/return", float(episode_return))
                    tb.add_scalar("episode/length", float(episode_step))
                    tb.add_scalar("episode/success", float(succ))
                    tb.add_scalar("episode/success_rate", succ_rate)
                    print(f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                        f"Step: {episode_step} | Success: {bool(succ)}")

                    # 回溯式奖励：如果成功，给最后一个 before hit transition 加成功奖励
                    if succ == 1.0 and before_hit_indices:
                        last_idx = before_hit_indices[-1]
                        if 0 <= last_idx < len(agent.replay_buffer.buf):
                            transition = agent.replay_buffer.buf[last_idx]
                            # transition 结构: (s, a, r, s2, d, note)
                            old_r = transition[2]
                            new_r = old_r + 80.0  # r_succ=80.0
                            transition = transition[:2] + (new_r,) + transition[3:]
                            agent.replay_buffer.buf[last_idx] = transition
                            print(
                                f"[RewardBackProp] Add success reward to buffer idx {last_idx} "
                                f"(r: {old_r:.3f} -> {new_r:.3f})"
                            )
                        else:
                            print(f"[RewardBackProp] invalid idx {last_idx}, current buf size={len(agent.replay_buffer.buf)}")

                    # === HER: 如果 episode 没有成功但已触球，采样球的终止位置作为虚拟 goal ===
                    if succ == 0.0 and hit_happened and before_hit_indices:
                    # 采样球的终止位置（episode 终点）
                        ball_pos, _, _ = self.env.ball_world.get_pose(self.env.root_states)
                        her_goal = ball_pos[:2].detach().cpu().numpy()
                        print(f"[HER] Sampled virtual goal: {her_goal}")

                        # 随机选取 10% 的 before-hit transition 做 HER，避免虚拟样本占比过高
                        num_her = max(1, int(0.1 * len(before_hit_indices)))
                        her_indices = np.random.choice(before_hit_indices, num_her, replace=False)

                        added = 0
                        buf_size = len(agent.replay_buffer)

                        for buf_idx in her_indices:
                            # deque 在 maxlen 时会左侧弹出，因此老的 index 可能失效，这里做一次安全检查
                            if buf_idx < 0 or buf_idx >= buf_size:
                                print(f"[HER] skip invalid idx {buf_idx} (buf_size={buf_size})")
                                continue

                            # 结构: (s, a, r, s2, d, note)
                            obs, act, rew, next_obs, done, note = agent.replay_buffer.buf[buf_idx]

                            # 基于虚拟 goal 修改观测
                            obs_her = obs.copy()
                            next_obs_her = next_obs.copy()
                            # 这里假设 obs[6:8] 存的是 target/goal 的 xy
                            obs_her[6:8] = her_goal
                            next_obs_her[6:8] = her_goal

                            # 重新计算在新 goal 下的奖励
                            rew_her = float(
                                self.env.compute_midlevel_reward_with_goal(
                                    np.asarray(obs_her, dtype=np.float32),
                                    her_goal,
                                    include_success=False,
                                )
                            )
                            # HER 终止：若在 next_obs 下达到虚拟 goal，则标注终止
                            done_her = bool(done) or (float(np.linalg.norm(next_obs_her[2:4] - her_goal)) < 0.60)
                            note_her = "her"

                            # ✅ 统一直接走 replay_buffer.push，带 note
                            agent.replay_buffer.push(
                                obs_her,
                                act,
                                rew_her,
                                next_obs_her,
                                done_her,
                                note=note_her,
                            )
                            added += 1

                        print(f"[HER] Added {added} HER transitions (sampled {num_her}).")
                    # === 课程更新：目前仍调用 env.on_episode_end，供你以后接入“target curriculum” ===
                    try:
                        rmin, rmax, changed, info = self.env.on_episode_end(
                            success=(success_happened and not fall_happened),
                            episode_idx=episode_idx
                        )
                        # 暂时只记录 r_max，虽然当前 target 采样没有用到它，
                        # 但以后如果你改成“距离课程”可以直接复用。
                        tb.add_scalar("curr/r_max", float(rmax))
                        if isinstance(info, dict):
                            if "rate_global" in info:
                                tb.add_scalar("curr/rate_global", float(info["rate_global"]))
                            if "rate_curr" in info:
                                tb.add_scalar("curr/rate_curr", float(info["rate_curr"]))
                            tb.add_scalar("curr/changed", 1.0 if changed else 0.0)
                        if changed and isinstance(info, dict) and info.get("reason"):
                            print(f"[Curriculum] ep#{episode_idx} r_max -> {rmax:.2f} | {info['reason']}")
                    except Exception as e:
                        print("[Curriculum] update failed:", e)

                    # 新回合
                    episode_idx += 1
                    episode_step = 0
                    episode_return = 0.0
                    
                    hit_happened = False
                    before_hit_indices = []
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)
                    self._set_target_for_pass()

                # ---------- 高层更新 ----------
                if len(agent.replay_buffer) >= WARMUP:
                    _t_upd = time.perf_counter()

                    def _do_updates():
                        for _ in range(UPDATE_K):
                            did_update, q1_loss, q2_loss, pi_loss, alpha_loss, alpha = agent.update()
                            if did_update:
                                tb.add_scalar("train/q1_loss", q1_loss)
                                tb.add_scalar("train/q2_loss", q2_loss)
                                tb.add_scalar("train/policy_loss", pi_loss)
                                tb.add_scalar("train/alpha_loss", alpha_loss)
                                tb.add_scalar("train/alpha", alpha)

                    _, upd_cuda_ms = _cuda_ms(_do_updates)
                    upd_cpu_ms = _cpu_ms(_t_upd)
                    _ema_update("update_ms", upd_cuda_ms if upd_cuda_ms is not None else upd_cpu_ms)

                # 周期性 flush，防止 TensorBoard 不刷盘
                if global_step % 200 == 0:
                    try:
                        tb.flush()
                    except:
                        pass

        finally:
            tb.close()

    def trapBall(self):
        run_name = f"{self.cfg['basic']['task']}_{time.strftime('%Y%m%d-%H%M%S')}"
        tb = TBLogger(
            logdir="logs/tb",
            run_name=run_name
        )
        global_step = 0

        ACTION_REPEAT = 10
        WARMUP = 5000
        UPDATE_K = 1
        PROFILE_EVERY = 50
        _use_cuda_timing = torch.cuda.is_available() and (str(self.device).startswith("cuda"))

        def _cpu_ms(t0: float) -> float:
            return (time.perf_counter() - t0) * 1000.0

        def _cuda_ms(fn):
            if (not _use_cuda_timing) or (global_step % PROFILE_EVERY != 0):
                return fn(), None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            return out, float(start.elapsed_time(end))

        _ema = {}

        def _ema_update(key: str, value: float, beta: float = 0.95) -> float:
            if value is None:
                return _ema.get(key, None)
            if key not in _ema:
                _ema[key] = value
            else:
                _ema[key] = beta * _ema[key] + (1.0 - beta) * value
            return _ema[key]

        obs, infos = self.env.reset()
        obs = obs.to(self.device)

        agent, action_mode = self._build_high_agent()

        episode_step = 0
        episode_return = 0.0
        max_steps = 300
        episode_idx = 0
        EP_RATE_WIN = 200
        ep_success_window = []

        try:
            while True:
                _t0 = time.perf_counter()
                obs_high = self.env.compute_midlevel_obs()
                if torch.is_tensor(obs_high) and obs_high.device != torch.device(self.device):
                    obs_high = obs_high.to(self.device)
                _ema_update("high_obs_ms", _cpu_ms(_t0))

                mode, action_repr, action_cmd = self._apply_high_level_cmd(action_mode, agent, obs_high)

                _t_rollout = time.perf_counter()
                last_infos = infos
                success_happened = False
                fail_happened = False
                fall_happened = False
                tb.set_step(global_step)
                tb.add_scalar("train/env_frames", global_step * ACTION_REPEAT)

                acc_rew_high_t = None
                acc_rew_high = 0.0
                obs_mod = obs.clone()

                for _ in range(ACTION_REPEAT):
                    with torch.no_grad():
                        obs_mod.copy_(obs)
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = (
                            action_cmd[0], action_cmd[1], action_cmd[2]
                        )

                        def _low_act():
                            dist = self.model.act(obs_mod)
                            return dist.loc

                        act, low_act_ms = _cuda_ms(_low_act)
                        if low_act_ms is not None:
                            _ema_update("low_act_ms", low_act_ms)

                        def _env_step():
                            return self.env.step(act)

                        (obs, rew, done, infos), env_step_ms = _cuda_ms(_env_step)
                        if env_step_ms is not None:
                            _ema_update("env_step_ms", env_step_ms)

                        if torch.is_tensor(obs) and obs.device != torch.device(self.device):
                            obs = obs.to(self.device)
                        last_infos = infos

                    if torch.is_tensor(rew):
                        if acc_rew_high_t is None:
                            acc_rew_high_t = torch.zeros((), device=rew.device, dtype=rew.dtype)
                        acc_rew_high_t = acc_rew_high_t + rew.reshape(-1)[0]
                    else:
                        acc_rew_high += float(rew)

                    if isinstance(infos, dict) and infos.get("fall", False):
                        fall_happened = True
                        tb.add_scalar("events/fallen", 1.0)
                        break

                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True
                        tb.add_scalar("events/success", 1.0)
                        break

                    if isinstance(infos, dict) and infos.get("fail", False):
                        fail_happened = True
                        tb.add_scalar("events/fail", 1.0)
                        break

                    if torch.any(done).item():
                        break

                _ema_update("rollout_ms", _cpu_ms(_t_rollout))

                if acc_rew_high_t is not None:
                    rew_high = float((acc_rew_high_t / ACTION_REPEAT).item())
                else:
                    rew_high = acc_rew_high / ACTION_REPEAT

                _t1 = time.perf_counter()
                next_obs_high = self.env.compute_midlevel_obs()
                if torch.is_tensor(next_obs_high) and next_obs_high.device != torch.device(self.device):
                    next_obs_high = next_obs_high.to(self.device)
                _ema_update("next_high_obs_ms", _cpu_ms(_t1))

                _t_np = time.perf_counter()
                next_obs_high_np = next_obs_high.squeeze(0).detach().cpu().numpy()
                obs_high_np = obs_high.squeeze(0).detach().cpu().numpy()
                _ema_update("to_numpy_ms", _cpu_ms(_t_np))

                episode_step += 1
                episode_return += rew_high
                done_high = (episode_step > max_steps) or success_happened or fail_happened or fall_happened

                _t_push = time.perf_counter()
                if mode == "discrete":
                    agent.replay_buffer.push(
                        obs_high_np,
                        int(action_repr),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note="trap",
                    )
                else:
                    agent.replay_buffer.push(
                        obs_high_np,
                        np.asarray(action_repr, dtype=np.float32),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note="trap",
                    )
                _ema_update("push_ms", _cpu_ms(_t_push))

                if hasattr(agent, "log_alpha"):
                    tb.add_scalar("sac/alpha", float(agent.log_alpha.exp().item()))
                tb.add_scalar("high/action_vx", float(action_cmd[0]))
                tb.add_scalar("high/action_vy", float(action_cmd[1]))
                tb.add_scalar("high/action_yaw", float(action_cmd[2]))
                tb.add_scalar("high/reward", rew_high)
                tb.add_scalar("train/replay_size", len(agent.replay_buffer))

                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        for k, v in terms.items():
                            try:
                                tb.add_scalar(f"rew/{k}", float(v))
                            except Exception:
                                pass
                global_step += 1

                if global_step % PROFILE_EVERY == 0:
                    for key in (
                        "high_obs_ms",
                        "next_high_obs_ms",
                        "rollout_ms",
                        "low_act_ms",
                        "env_step_ms",
                        "to_numpy_ms",
                        "push_ms",
                        "update_ms",
                    ):
                        v = _ema.get(key, None)
                        if v is not None:
                            tb.add_scalar(f"time/{key}", v)

                if global_step % 10000 == 0 and len(agent.replay_buffer) > 0:
                    try:
                        agent.replay_buffer.save_to_disk(
                            save_dir="logs/replay",
                            filename=f"trap_replay_step_{global_step}_N{len(agent.replay_buffer)}.npz"
                        )
                    except Exception as e:
                        print("[ReplayBuffer] save failed:", e)

                if (global_step % 10000 == 0) and (len(agent.replay_buffer) >= WARMUP):
                    ckpt_dir = os.path.join("logs", "ckpt", "trapBall", "sac")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"sac_agent_step_{global_step}.pt")
                    torch.save({"agent": agent, "global_step": global_step, "cfg": self.cfg}, ckpt_path)
                    print(f"[Save] SAC agent saved at step {global_step} -> {ckpt_path}")

                if done_high:
                    succ = 1.0 if (success_happened and not fall_happened and not fail_happened) else 0.0
                    ep_success_window.append(succ)
                    if len(ep_success_window) > EP_RATE_WIN:
                        ep_success_window.pop(0)
                    succ_rate = float(sum(ep_success_window) / len(ep_success_window))
                    tb.add_scalar("episode/return", float(episode_return))
                    tb.add_scalar("episode/length", float(episode_step))
                    tb.add_scalar("episode/success", float(succ))
                    tb.add_scalar("episode/success_rate", succ_rate)
                    tb.add_scalar("episode/init_dist", float(self.env.get_initial_dist_xy()))
                    print(
                        f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                        f"Step: {episode_step} | Success: {bool(succ)} | "
                        f"Fail: {bool(fail_happened)} | Fall: {bool(fall_happened)} | "
                        f"InitDist: {self.env.get_initial_dist_xy():.2f}"
                    )

                    episode_idx += 1
                    episode_step = 0
                    episode_return = 0.0
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)

                if len(agent.replay_buffer) >= WARMUP:
                    _t_upd = time.perf_counter()

                    def _do_updates():
                        for _ in range(UPDATE_K):
                            did_update, q1_loss, q2_loss, pi_loss, alpha_loss, alpha = agent.update()
                            if did_update:
                                tb.add_scalar("train/q1_loss", q1_loss)
                                tb.add_scalar("train/q2_loss", q2_loss)
                                tb.add_scalar("train/policy_loss", pi_loss)
                                tb.add_scalar("train/alpha_loss", alpha_loss)
                                tb.add_scalar("train/alpha", alpha)

                    _, upd_cuda_ms = _cuda_ms(_do_updates)
                    upd_cpu_ms = _cpu_ms(_t_upd)
                    _ema_update("update_ms", upd_cuda_ms if upd_cuda_ms is not None else upd_cpu_ms)

                if global_step % 200 == 0:
                    try:
                        tb.flush()
                    except Exception:
                        pass

        finally:
            tb.close()

    def boosterT12v2(self):
        """
        Minimal inference-first runner for the new 2v2 Booster T1 IsaacGym env.
        It reuses the existing single-robot low-level policy in batched mode.
        """
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        zero_cmd = self.env.get_high_level_action_space()
        self.env.apply_high_level_command(zero_cmd)
        max_steps = int(self.cfg.get("game", {}).get("episode_length_steps", 1000))

        for step_idx in range(max_steps):
            with torch.no_grad():
                obs_mod = obs.clone()
                obs_mod[:, 6] = self.env.commands[:, 0]
                obs_mod[:, 7] = self.env.commands[:, 1]
                obs_mod[:, 8] = self.env.commands[:, 2]
                dist = self.model.act(obs_mod)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs = obs.to(self.device)

            if step_idx % 100 == 0:
                ball = infos["ball_state"]["pos"]
                print(
                    f"[boosterT12v2] step={step_idx} "
                    f"ball=({float(ball[0]):.2f}, {float(ball[1]):.2f}, {float(ball[2]):.2f})"
                )

            if torch.any(done).item():
                print(f"[boosterT12v2] episode finished at step {step_idx}")
                break

    def boosterT12v2Locomotion(self):
        """
        Smoke-test the pretrained low-level locomotion model in the 2v2 env
        before any high-level task logic is introduced.
        """
        obs, infos = self.env.reset()
        obs = obs.to(self.device)

        gait_freq = 0.5 * (
            self.cfg["commands"]["gait_frequency"][0] + self.cfg["commands"]["gait_frequency"][1]
        )
        phases = [
            {"steps": 120, "home": [0.45, 0.0, 0.0, gait_freq], "away": [-0.45, 0.0, 0.0, gait_freq], "label": "forward/backward"},
            {"steps": 120, "home": [0.0, 0.25, 0.0, gait_freq], "away": [0.0, -0.25, 0.0, gait_freq], "label": "lateral"},
            {"steps": 120, "home": [0.2, 0.0, 0.6, gait_freq], "away": [-0.2, 0.0, -0.6, gait_freq], "label": "turning"},
            {"steps": 120, "home": [0.0, 0.0, 0.0, gait_freq], "away": [0.0, 0.0, 0.0, gait_freq], "label": "settle"},
        ]

        print("[boosterT12v2Locomotion] starting low-level locomotion smoke test")
        global_step = 0
        current_phase_end = 0

        for phase_idx, phase in enumerate(phases):
            current_phase_end += phase["steps"]
            self.env.set_team_command("home", phase["home"])
            self.env.set_team_command("away", phase["away"])
            print(f"[boosterT12v2Locomotion] phase={phase_idx} label={phase['label']} steps={phase['steps']}")

            for _ in range(phase["steps"]):
                with torch.no_grad():
                    obs_mod = obs.clone()
                    obs_mod[:, 6] = self.env.commands[:, 0]
                    obs_mod[:, 7] = self.env.commands[:, 1]
                    obs_mod[:, 8] = self.env.commands[:, 2]
                    dist = self.model.act(obs_mod)
                    act = dist.loc
                    obs, rew, done, infos = self.env.step(act)
                    obs = obs.to(self.device)
                global_step += 1

                if global_step % 40 == 0 or global_step == current_phase_end:
                    motion = infos["motion_metrics"]
                    mean_xy_err = motion["tracking_xy_error"].abs().mean(dim=0)
                    mean_yaw_err = motion["tracking_yaw_error"].abs().mean()
                    home_center = self.env.base_pos[self.env.team_indices["home"], :2].mean(dim=0)
                    away_center = self.env.base_pos[self.env.team_indices["away"], :2].mean(dim=0)
                    print(
                        "[boosterT12v2Locomotion] "
                        f"step={global_step} "
                        f"home_center=({float(home_center[0]):.2f}, {float(home_center[1]):.2f}) "
                        f"away_center=({float(away_center[0]):.2f}, {float(away_center[1]):.2f}) "
                        f"mean_xy_err=({float(mean_xy_err[0]):.3f}, {float(mean_xy_err[1]):.3f}) "
                        f"mean_yaw_err={float(mean_yaw_err):.3f}"
                    )

                if torch.any(done).item():
                    print(f"[boosterT12v2Locomotion] env terminated early at step {global_step}")
                    return

        print("[boosterT12v2Locomotion] completed")
