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
from core.agents.dqn.agent import DQNAgent
from core.agents.sac.agent import SACAgent
from core.utils.logger import TBLogger
from eval.chaseBallEvaluator import ChaseBallEvaluator

class Runner:

    def __init__(self, test=False):
        self.test = test
        # CLI + CFG
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        # env
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        # device
        self.device = self.cfg["basic"]["rl_device"]
        # low_level locomotion model(already trained)
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--algo", type=str, choices=["dqn", "sac"], default="dqn", help="High-level RL algorithm: dqn or sac.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "passBall",f"{self.args.task}.yaml")
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        self.cfg["basic"].setdefault("algo", "dqn")
        self.cfg["basic"].setdefault("curriculum_window", 50)     # 最近 N 个 episode 统计成功率
        self.cfg["basic"].setdefault("eval_every_episodes", 100)  # 每 N 个 episode 评估
        self.cfg["basic"].setdefault("use_respawn_on_success", False)  # True=成功后不结束、刷远球继续追
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

    def _build_high_agent(self):
        """根据 algo 构建高层 agent。"""
        algo = self.cfg["basic"]["algo"].lower()
        obs_high = self.env.compute_high_level_obs().to(self.device)
        state_dim = int(obs_high.shape[1])

        if algo == "dqn":
            # 离散动作数量（来自 env 的查表）
            # 这里假定你的 env.high_level_action_id_to_vector 支持 id ∈ [0, N-1]
            # 你当前实现是 6 动（0..5）
            action_dim = 6
            agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=self.device)
            action_mode = "discrete"
        elif algo == "sac":
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

    def passBall(self):
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
        # build high-level agent
        agent, action_mode = self._build_high_agent()

        self.evaluator = evaluator = ChaseBallEvaluator(
            max_steps=500, 
            success_dist_thresh=0.6,
            tb_prefix="eval",
            save_dir = os.path.join("logs", "ckpt"),
            save_best_by = "success_rate",
            higher_is_better = True,
            save_every_eval = False,
        )
        
        episode_step = 0
        episode_return = 0
        max_steps = 500
        episode_idx = 0
        CURR_N = int(self.cfg["basic"]["curriculum_window"])
        succ_history = []  # 最近 N 个 episode 的 0/1
        use_respawn = bool(self.cfg["basic"]["use_respawn_on_success"])
        respawn_r_min = float(self.cfg["basic"]["respawn_r_min"])
        respawn_r_max = float(self.cfg["basic"]["respawn_r_max"])

        try:
            while True:
                # ---------- 高层观测 ----------
                obs_high = self.env.compute_high_level_obs().to(self.device)

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
                        if use_respawn:
                            try:
                                self.env.respawn_ball_far(r_min=respawn_r_min, r_max=respawn_r_max)
                            except Exception:
                                pass
                            # 连击模式下不 break，本高层步继续滚动（你也可以选择 break）
                        else:
                            break
                    if torch.any(done).item():
                        break
                if not success_happened:
                    tb.add_scalar("events/success", 0.0)
                if not fall_happened:
                    tb.add_scalar("events/fallen", 0.0)
                # ---------- 高层一步的转移 ----------
                next_obs_high = self.env.compute_high_level_obs().to(self.device)
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

                # ---------- 定期评估 ----------
                EVAL_EVERY = int(self.cfg["basic"]["eval_every_episodes"])
                if (episode_idx % EVAL_EVERY == 0) and (episode_idx != 0):
                    metrics = self.evaluator.evaluate(
                        env=self.env, low_model=self.model, high_agent=agent,
                        device=self.device, episodes=5, tb=tb, global_step=global_step
                    )
                    print(f"[Eval @ episode {episode_idx} | step {global_step}] {metrics}")
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

                    # 课程统计
                    succ_history.append(succ)
                    if len(succ_history) > CURR_N:
                        succ_history.pop(0)
                    # 每个 episode 都可以用最近窗口调整一次课程窗口
                    if episode_idx > 20 :
                        try:
                            self.env.set_curriculum_window(succ_count=int(sum(succ_history)), epi_count=len(succ_history))
                        except Exception:
                            pass

                    episode_idx += 1
                    episode_step = 0
                    episode_return = 0.0
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)

                # ---------- 更新 ----------
                if len(agent.replay_buffer) >= WARMUP:
                    for _ in range(UPDATE_K):
                        did_update, loss_val = agent.update()
                        if did_update and (loss_val is not None):
                            tb.add_scalar("train/loss", loss_val)

                # 可选：周期性 flush，防 TensorBoard 不刷盘
                if global_step % 200 == 0:
                    try: tb.flush()
                    except: pass

        finally:
            tb.close()
    
    if __name__ == "__main__":
        runner = Runner(test=False)
        runner.chaseBall()