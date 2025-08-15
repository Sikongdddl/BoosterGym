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
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "chaseBall",f"{self.args.task}.yaml")
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
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
            all_ckpts = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)
            if not all_ckpts:
                print("[WARN] No checkpoint found under logs/**.pth; skip loading.")
                return
            ckpt = all_ckpts[-1]
            self.cfg["basic"]["checkpoint"] = ckpt
        print(f"Loading low-level model from {ckpt}")
        model_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)

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
        UPDATE_k = 1    # <<< 低层模型更新频率

        # init
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        gait_freq = 1.5
        obs_high = self.env.compute_high_level_obs().to(self.device)
        action_high = [0.0, 0.0, 0.0, gait_freq]
        agent = DQNAgent(state_dim=obs_high.shape[1], action_dim=4, device=self.device)
        
        episode_step = 0
        episode_return = 0
        max_steps = 500
        episode_idx = 0

        self.evaluator = evaluator = ChaseBallEvaluator(
            max_steps=500, 
            success_dist_thresh=0.5,
            tb_prefix="eval")

        try:
            while True:
                # high level observation and action
                obs_high = self.env.compute_high_level_obs().to(self.device)
                obs_high_np = obs_high.squeeze(0).cpu().numpy()
                
                # 选动作前先算一眼Q用于记录（不影响选择的epsilon逻辑）
                with torch.no_grad():
                    q_vals = agent.q_net(torch.as_tensor(obs_high_np, dtype=torch.float32, device=self.device).unsqueeze(0))
                    q_max  = float(q_vals.max().item())
                    q_mean = float(q_vals.mean().item())
                
                action_high_id = agent.select_action(obs_high_np)
                q_selected = float(q_vals[0, action_high_id].item())
                action_high = self.env.high_level_action_id_to_vector(action_high_id)
                self.env.apply_high_level_command(action_high)
                
                # low level inference
                acc_rew_high = 0.0
                last_infos = infos
                success_happened = False

                for _ in range(ACTION_REPEAT):
                    with torch.no_grad():
                        #low level step first
                        obs_mod = obs.clone()
                        obs_mod[:, 6] = action_high[0]   # 期望x方向线速度
                        obs_mod[:, 7] = action_high[1]   # 期望y方向线速度
                        obs_mod[:, 8] = action_high[2]   # 期望角速度（绕z轴）
                        
                        dist = self.model.act(obs_mod)
                        act = dist.loc
                        obs, rew, done, infos = self.env.step(act)
                        obs = obs.to(self.device)
                        last_infos = infos

                    step_rew_high = self.env.compute_high_level_reward().item()
                    acc_rew_high += step_rew_high

                    if isinstance(infos, dict) and infos.get("success", False):
                        success_happened = True
                        break

                    if done:
                        break
                
                # high level step
                next_obs_high = self.env.compute_high_level_obs().to(self.device)
                next_obs_high_np = next_obs_high.squeeze(0).cpu().numpy()
                rew_high = acc_rew_high if success_happened else (acc_rew_high / ACTION_REPEAT)

                # statistics
                episode_step += 1
                episode_return += rew_high
                done_high = episode_step > max_steps or success_happened
                # save transitions to buffer
                agent.push(
                    obs_high_np,
                    action_high_id,
                    rew_high,
                    next_obs_high_np,
                    done_high
                )

                # tensorboard logging
                tb.set_step(global_step)
                env_frames = global_step * ACTION_REPEAT
                tb.add_scalar("train/env_frames", env_frames)
                tb.add_scalar("train/action_repeat", ACTION_REPEAT)
                tb.add_scalar("train/replay_size", len(agent.replay_buffer))
                tb.add_scalar("train/epsilon", agent.epsilon)
                tb.add_scalar("train/update_steps", agent.step_count)

                # 高层行为/Q值
                tb.add_scalar("high/action_id", action_high_id)
                tb.add_scalar("high/reward", rew_high)
                tb.add_scalar("high/q_max", q_max)
                tb.add_scalar("high/q_mean", q_mean)
                tb.add_scalar("high/q_selected", q_selected)
                tb.add_scalar("high/success", float(success_happened))

                # 奖励分解项（来自 env.extras）
                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        if "dist_xy" in terms:
                            tb.add_scalar("rew/dist_xy", float(terms["dist_xy"]))
                        if "heading_cos" in terms:
                            tb.add_scalar("rew/heading_cos", float(terms["heading_cos"]))
                        if "progress" in terms:
                            tb.add_scalar("rew/progress", float(terms["progress"]))
                        if "progress_norm" in terms:
                            tb.add_scalar("rew/progress_norm", float(terms["progress_norm"]))
                        if "speed_toward" in terms:
                            tb.add_scalar("rew/speed_toward", float(terms["speed_toward"]))
                        if "speed_orth" in terms:
                            tb.add_scalar("rew/speed_orth", float(terms["speed_orth"]))
                        if "spin_penalty" in terms:
                            tb.add_scalar("rew/spin_penalty", float(terms["spin_penalty"]))

                # evaluate every 100 episodes
                if episode_idx % 100 == 0 and episode_idx != 0:
                    metrics = self.evaluator.evaluate(
                        env = self.env,
                        low_model = self.model,
                        high_agent = agent,
                        device = self.device,
                        episodes = 5,
                        tb = tb
                    )
                    print(f"[Eval @ step {episode_idx}] {metrics}")

                # episode end
                if done_high:
                    tb.add_scalars("high/episode", {
                        "return": episode_return,
                        "length": episode_step,
                        "success": float(success_happened),
                    })
                    print(f"[Episode End] Return: {episode_return:.2f}, Step: {episode_step}")
                    episode_idx += 1
                    episode_step = 0
                    episode_return = 0
                    obs, infos = self.env.reset()
                    obs = obs.to(self.device)

                # optimize model 
                if len(agent.replay_buffer) >= WARMUP:
                    for _ in range(UPDATE_K):
                        did_update, loss_val = agent.update()
                        if did_update and loss_val is not None:
                            tb.add_scalar("train/loss", loss_val)
        finally:
            tb.close()
