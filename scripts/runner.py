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
        版本1：任务难度暂时不走 curriculum，
        直接把 target 固定在一个“看起来像真·传球”的距离上，
        方向在前方一个小扇形内。
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

        # 3) 距离：直接固定在 4~6 米之间，看起来比较像传球
        TARGET_DIST_MIN = 4.0
        TARGET_DIST_MAX = 6.0
        R = np.random.uniform(TARGET_DIST_MIN, TARGET_DIST_MAX)

        tx = base_x + R * dir_x
        ty = base_y + R * dir_y

        target_xy = torch.tensor([tx, ty], dtype=self.env.base_pos.dtype, device=device)
        self.env.target_xy = target_xy

        print(f"[Target] R={R:.2f}, theta={theta:.2f} rad, target=({tx:.2f}, {ty:.2f})")
        
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
                        # 写入期望速度到低层观察
                        obs_mod = obs.clone()
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = action_cmd[0], action_cmd[1], action_cmd[2]
                        dist = self.model.act(obs_mod)
                        act = dist.loc
                        obs, rew, done, infos = self.env.step(act)
                        obs = obs.to(self.device)
                        last_infos = infos

                    step_rew_high = float(rew) 
                    acc_rew_high += step_rew_high

                    if isinstance(infos, dict) and infos.get("fall", False):
                        fall_happened = True   
                        tb.add_scalar("events/fallen", 1.0)  # 触发就写 1   
                        break

                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True     
                        tb.add_scalar("events/success", 1.0)  # 触发就写 1
                        break
                    if torch.any(done).item():
                        break
                if not success_happened:
                    tb.add_scalar("events/success", 0.0)
                if not fall_happened:
                    tb.add_scalar("events/fallen", 0.0)
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
                    tb.add_scalars("high/episode", {
                        "return": episode_return,
                        "length": episode_step,
                        "success": succ,
                        "init_dist": self.env.get_initial_dist_xy(),
                        "cur_r_max": self.env.cur_r_max,
                    })
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

        # 初始化目标位置 & 环境
        obs, infos = self.env.reset()
        obs = obs.to(self.device)

        self._set_target_for_pass()

        # build high-level agent
        agent, action_mode = self._build_high_agent()

        episode_step = 0
        episode_return = 0.0
        max_steps = 500
        episode_idx = 0

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
                            ).unsqueeze(0)
                        )
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
                        # 写入期望速度到低层观察
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
                        tb.add_scalar("events/fallen", 1.0)  # 触发就写 1
                        break

                    # ✅ success 判定：完全依赖 env.extras["success"]
                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True
                        tb.add_scalar("events/success", 1.0)
                        break

                    if torch.any(done).item():
                        break

                if not success_happened:
                    tb.add_scalar("events/success", 0.0)
                if not fall_happened:
                    tb.add_scalar("events/fallen", 0.0)

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

                # ---------- TensorBoard：高层动作 & 算法状态 ----------
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

                # ---------- TensorBoard：新 reward 分解 ----------
                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        # 来自新 reward：
                        # ball_dist, ball_progress, ball_speed, align_cos, align_reward, touch_now, first_touch
                        if "ball_dist" in terms:
                            tb.add_scalar("rew/ball_dist", float(terms["ball_dist"]))
                        if "ball_progress" in terms:
                            tb.add_scalar("rew/ball_progress", float(terms["ball_progress"]))
                        if "ball_speed" in terms:
                            tb.add_scalar("rew/ball_speed", float(terms["ball_speed"]))
                        if "align_cos" in terms:
                            tb.add_scalar("rew/align_cos", float(terms["align_cos"]))
                        if "align_reward" in terms:
                            tb.add_scalar("rew/align_reward", float(terms["align_reward"]))
                        if "touch_now" in terms:
                            tb.add_scalar("events/touch_now", float(terms["touch_now"]))
                        if "first_touch" in terms:
                            tb.add_scalar("events/first_touch", float(terms["first_touch"]))
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

                    # 这里只保留基本 episode 指标；几何指标以后可以按需要加
                    tb.add_scalars("high/episode", {
                        "return": episode_return,
                        "length": episode_step,
                        "success": succ,
                    })
                    print(f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                        f"Step: {episode_step} | Success: {bool(succ)}")

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

                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)
                    self._set_target_for_pass()

                # ---------- 高层更新 ----------
                if len(agent.replay_buffer) >= WARMUP:
                    for _ in range(UPDATE_K):
                        did_update, q1_loss, q2_loss, pi_loss, alpha_loss, alpha = agent.update()
                        if did_update:
                            tb.add_scalar("train/q1_loss", q1_loss)
                            tb.add_scalar("train/q2_loss", q2_loss)
                            tb.add_scalar("train/policy_loss", pi_loss)
                            tb.add_scalar("train/alpha_loss", alpha_loss)
                            tb.add_scalar("train/alpha", alpha)

                # 周期性 flush，防止 TensorBoard 不刷盘
                if global_step % 200 == 0:
                    try:
                        tb.flush()
                    except:
                        pass

        finally:
            tb.close()
