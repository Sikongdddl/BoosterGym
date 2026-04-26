import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import imageio
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from envs import *
import torch
from isaacgym.torch_utils import quat_rotate
from utils.model import ActorCritic
from core.agents.sac.agent import SACAgent
from core.utils.logger import TBLogger

class Runner:

    def __init__(self, test=False, task_name=None):
        self.test = test
        self.task_name_dir = task_name
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
        self.low_policy_kind = "actor_critic"
        self.low_clip_actions = float(self.cfg.get("normalization", {}).get("clip_actions", 1.0))
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--algo", type=str, choices=["dqn", "sac"], default="sac", help="High-level RL algorithm: dqn or sac.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--high_checkpoint", type=str, help="Path of the high-level SAC checkpoint to resume from. Use -1 to load latest task checkpoint.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        self.args = parser.parse_args()

    def _resolve_task_dir(self, task_name):
        if task_name:
            return task_name
        mapping = {
            "ChaseBallEnv": "chaseBall",
            "DribbleBallEnv": "dribbleBall",
            "PassBallEnv": "passBall",
            "TrapBallEnv": "trapBall",
        }
        if self.args.task in mapping:
            return mapping[self.args.task]
        raise ValueError(
            f"Cannot infer task directory for task '{self.args.task}'. "
            "Please pass task_name explicitly when constructing Runner."
        )

    # Override config file with args if needed
    def _update_cfg_from_args(self, task_name):
        self.task_name_dir = self._resolve_task_dir(task_name)
        cfg_file = os.path.join("envs", self.task_name_dir, f"{self.args.task}.yaml")
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
                print("[WARN] No low-level checkpoint found under logs/low/**/*.pth; skip loading low-level model.")
                return
            ckpt = all_ckpts[-1]
            self.cfg["basic"]["checkpoint"] = ckpt
        print(f"Loading low-level model from {ckpt}")
        ckpt = str(ckpt)

        # 支持 TorchScript 低层策略（例如 deploy/models/T1.pt）
        if ckpt.lower().endswith(".pt"):
            try:
                model = torch.jit.load(ckpt, map_location=self.device)
                model.eval()
                test_out = model(torch.zeros(1, self.env.num_obs, device=self.device, dtype=torch.float32))
                if int(test_out.shape[-1]) != int(self.env.num_actions):
                    raise ValueError(
                        f"Low-level TorchScript output dim mismatch: got {tuple(test_out.shape)}, "
                        f"expected (*, {self.env.num_actions})"
                    )
                self.model = model
                self.low_policy_kind = "jit"
                print(f"[Load] low-level policy mode: {self.low_policy_kind}")
                return
            except Exception as jit_err:
                # 回退识别：若该 .pt 实际是 pickle/checkpoint，给出更明确错误
                try:
                    model_obj = torch.load(ckpt, map_location=self.device, weights_only=True)
                except Exception:
                    model_obj = torch.load(ckpt, map_location=self.device, weights_only=False)

                if isinstance(model_obj, dict) and "model" in model_obj:
                    self.model.load_state_dict(model_obj["model"], strict=False)
                    self.low_policy_kind = "actor_critic"
                    print(f"[Load] low-level policy mode: {self.low_policy_kind}")
                    return

                if isinstance(model_obj, dict) and "agent" in model_obj:
                    raise ValueError(
                        f"Invalid --checkpoint: {ckpt}\n"
                        "This file is a high-level SAC checkpoint (contains key 'agent').\n"
                        "Use it with --high_checkpoint, and provide a low-level locomotion .pth/.pt for --checkpoint."
                    ) from jit_err

                keys_preview = list(model_obj.keys())[:10] if isinstance(model_obj, dict) else None
                raise ValueError(
                    f"Unsupported low-level .pt format: {ckpt}\n"
                    f"TorchScript load error: {repr(jit_err)}\n"
                    f"Detected type: {type(model_obj)}"
                    + (f", keys preview: {keys_preview}" if keys_preview is not None else "")
                ) from jit_err

        try:
            model_obj = torch.load(ckpt, map_location=self.device, weights_only=True)
        except Exception:
            # 兼容旧格式/包含自定义类的 checkpoint（仅限本地可信文件）
            model_obj = torch.load(ckpt, map_location=self.device, weights_only=False)

        if isinstance(model_obj, dict) and "model" in model_obj:
            self.model.load_state_dict(model_obj["model"], strict=False)
            self.low_policy_kind = "actor_critic"
            print(f"[Load] low-level policy mode: {self.low_policy_kind}")
            return

        if isinstance(model_obj, dict) and "agent" in model_obj:
            raise ValueError(
                f"Invalid --checkpoint: {ckpt}\n"
                "This file is a high-level SAC checkpoint (contains key 'agent').\n"
                "Use it with --high_checkpoint, and provide a low-level locomotion .pth "
                "that contains key 'model' for --checkpoint."
            )

        keys_preview = list(model_obj.keys())[:10] if isinstance(model_obj, dict) else None
        raise ValueError(
            f"Unsupported low-level checkpoint format: {ckpt}\n"
            "Expected a dict with key 'model' (low-level ActorCritic weights).\n"
            f"Detected type: {type(model_obj)}"
            + (f", keys preview: {keys_preview}" if keys_preview is not None else "")
        )

    def _low_policy_action(self, obs_mod):
        if self.low_policy_kind == "jit":
            act = self.model(obs_mod)
            act = torch.as_tensor(act, device=obs_mod.device, dtype=obs_mod.dtype)
            return torch.clamp(act, -self.low_clip_actions, self.low_clip_actions)
        dist = self.model.act(obs_mod)
        return dist.loc

    @staticmethod
    def _optimizer_to_device(optimizer, device):
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    @staticmethod
    def _move_loaded_sac_agent_to_device(agent, device):
        agent.device = device
        for module_name in ("policy", "q1", "q2", "q1_target", "q2_target"):
            module = getattr(agent, module_name, None)
            if module is not None:
                module.to(device)
        if hasattr(agent, "action_low") and torch.is_tensor(agent.action_low):
            agent.action_low = agent.action_low.to(device)
        if hasattr(agent, "action_high") and torch.is_tensor(agent.action_high):
            agent.action_high = agent.action_high.to(device)
        if hasattr(agent, "log_alpha") and torch.is_tensor(agent.log_alpha):
            agent.log_alpha = agent.log_alpha.to(device)
            agent.log_alpha.requires_grad_(True)

        Runner._optimizer_to_device(getattr(agent, "pi_optim", None), device)
        Runner._optimizer_to_device(getattr(agent, "q1_optim", None), device)
        Runner._optimizer_to_device(getattr(agent, "q2_optim", None), device)
        Runner._optimizer_to_device(getattr(agent, "alpha_optim", None), device)
        return agent

    def _resolve_high_ckpt_path(self, high_ckpt):
        if not high_ckpt:
            return None
        if high_ckpt != "-1" and high_ckpt != -1:
            return high_ckpt

        task_dir = self.task_name_dir if self.task_name_dir else "passBall"
        # 兼容两种目录结构：
        # 1) logs/ckpt/<task>/sac/sac_agent_step_*.pt
        # 2) logs/ckpt/<task>/sac/<run_name>/sac_agent_step_*.pt
        pattern_flat = os.path.join("logs", "ckpt", task_dir, "sac", "sac_agent_step_*.pt")
        pattern_nested = os.path.join("logs", "ckpt", task_dir, "sac", "**", "sac_agent_step_*.pt")
        all_ckpts = list(glob.glob(pattern_flat))
        all_ckpts.extend(glob.glob(pattern_nested, recursive=True))
        all_ckpts = sorted(set(all_ckpts), key=os.path.getmtime)
        if not all_ckpts:
            print(
                "[WARN] No high-level checkpoint found under "
                f"{os.path.join('logs', 'ckpt', task_dir, 'sac')}; skip resume."
            )
            return None
        latest = all_ckpts[-1]
        print(f"[Resume] Auto-selected latest high-level checkpoint: {latest}")
        return latest

    @staticmethod
    def _sac_io_dims(agent):
        state_dim = None
        action_dim = None
        try:
            first = agent.policy.net[0]
            state_dim = int(first.in_features)
        except Exception:
            pass
        try:
            action_dim = int(agent.policy.mu.out_features)
        except Exception:
            pass
        return state_dim, action_dim

    def _resume_high_level_if_needed(self, agent):
        high_ckpt = self.cfg["basic"].get("high_checkpoint", None)
        high_ckpt = self._resolve_high_ckpt_path(high_ckpt)
        if not high_ckpt:
            return agent, 0

        print(f"[Resume] Loading high-level checkpoint from {high_ckpt}")
        payload = torch.load(high_ckpt, map_location=self.device, weights_only=False)

        resumed_step = 0
        resumed_agent = agent

        if isinstance(payload, dict) and "agent" in payload:
            loaded_agent = payload["agent"]
            if not isinstance(loaded_agent, SACAgent):
                raise TypeError(
                    f"Unsupported checkpoint format in {high_ckpt}: payload['agent'] is {type(loaded_agent)}"
                )
            resumed_agent = self._move_loaded_sac_agent_to_device(loaded_agent, self.device)
            resumed_step = int(payload.get("global_step", 0))
        elif isinstance(payload, SACAgent):
            resumed_agent = self._move_loaded_sac_agent_to_device(payload, self.device)
            resumed_step = 0
        elif isinstance(payload, dict) and "policy" in payload:
            resumed_agent.load(high_ckpt)
            resumed_step = int(payload.get("global_step", 0))
            resumed_agent = self._move_loaded_sac_agent_to_device(resumed_agent, self.device)
        else:
            raise ValueError(
                f"Unsupported high-level checkpoint format: {high_ckpt}. "
                "Expected SACAgent, {'agent': SACAgent, ...}, or SACAgent.save() weights."
            )

        exp_s, exp_a = self._sac_io_dims(agent)
        got_s, got_a = self._sac_io_dims(resumed_agent)
        if (exp_s is not None and got_s is not None and exp_s != got_s) or (
            exp_a is not None and got_a is not None and exp_a != got_a
        ):
            raise ValueError(
                "High-level checkpoint incompatible with current task model shape: "
                f"expected (state_dim={exp_s}, action_dim={exp_a}), "
                f"got (state_dim={got_s}, action_dim={got_a}) from {high_ckpt}. "
                "This usually means cross-task resume (e.g., chaseBall ckpt -> trapBall env). "
                "Use a checkpoint from the same task, or omit --high_checkpoint to train from scratch."
            )

        print(f"[Resume] High-level training will continue from global_step={resumed_step}")
        return resumed_agent, resumed_step

    def _set_target_from_ball(self):
        x,_,_ = self.env.ball_world.get_pose(self.env.root_states)
        self.env.target_xy = x[:2].to(self.device)

    def _get_robot_heading(self) -> float:
        """
        返回机器人当前朝向（世界系 yaw，弧度）。
        使用 base_quat 旋转局部前向向量，避免四元数分量顺序歧义。
        """
        fwd_local = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.env.base_quat.dtype)
        fwd_world = quat_rotate(self.env.base_quat[0:1], fwd_local[None, :]).squeeze(0)
        return float(torch.atan2(fwd_world[1], fwd_world[0]).item())

    def _set_target_for_pass(self):
        """
        为 passBall 采样一个新的 target_xy 并写进 env。
        采样规则：
        - 方向：机器人前方扇形（相对机器人朝向）
        - 距离：按 curriculum.pass_target_r_min / pass_target_r_max
        """
        device = self.device
        curr_cfg = self.env.controller.cfg.get("curriculum", {})

        # 1) 取机器人当前位置 (世界系)
        base_pos = self.env.base_pos[0, :3]   # tensor
        base_x = float(base_pos[0].item())
        base_y = float(base_pos[1].item())

        # 2) 方向：相对机器人当前朝向的前方总宽 60°（±30°），连续采样
        heading = self._get_robot_heading()
        theta_half_deg = float(curr_cfg.get("pass_target_theta_half_deg", 30.0))
        theta_min_deg = -theta_half_deg
        theta_max_deg = theta_half_deg
        theta = heading + np.random.uniform(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg))
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)

        # 3) 距离：由 curriculum 配置驱动
        r_min = float(curr_cfg.get("pass_target_r_min", 1.3))
        r_max = float(curr_cfg.get("pass_target_r_max", 1.6))
        if r_max < r_min:
            r_min, r_max = r_max, r_min
        R = np.random.uniform(r_min, r_max)

        tx = base_x + R * dir_x
        ty = base_y + R * dir_y

        target_xy = torch.tensor([tx, ty], dtype=self.env.base_pos.dtype, device=device)
        self.env.target_xy = target_xy

        print(
            f"[Target] R={R:.2f} in [{r_min:.2f}, {r_max:.2f}] "
            f"theta={theta:.2f} rad (heading±{theta_half_deg:.1f}deg, heading={heading:.2f}), target=({tx:.2f}, {ty:.2f})"
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
            curr_cfg = self.env.controller.cfg.get("curriculum", {})
            task_name = str(self.cfg["basic"].get("task", ""))
            if task_name == "TrapBallEnv":
                vx_min = float(curr_cfg.get("trap_vx_min", -0.25))
                vx_max = float(curr_cfg.get("trap_vx_max", 0.6))
                if vx_max < vx_min:
                    vx_min, vx_max = vx_max, vx_min
                vx_range = np.array([vx_min, vx_max], dtype=np.float32)
            elif task_name == "DribbleBallEnv":
                vx_min = float(curr_cfg.get("dribble_vx_min", -0.10))
                vx_max = float(curr_cfg.get("dribble_vx_max", 0.60))
                if vx_max < vx_min:
                    vx_min, vx_max = vx_max, vx_min
                vx_range = np.array([vx_min, vx_max], dtype=np.float32)
            else:
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
                sample_sigma=float(curr_cfg.get("sample_sigma", 0.2)),
                sample_epsilon=float(curr_cfg.get("sample_epsilon", 0.1)),
                sample_success_bonus=float(curr_cfg.get("sample_success_bonus", 1.2)),
                sample_hard_focus=float(curr_cfg.get("sample_hard_focus", 0.0)),
                sample_rmax_ema_beta=float(curr_cfg.get("sample_rmax_ema_beta", 0.9)),
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
            curr_cfg = self.env.controller.cfg.get("curriculum", {})
            task_name = str(self.cfg["basic"].get("task", ""))
            if task_name == "TrapBallEnv":
                default_smooth = 0.2
                smooth_alpha = float(curr_cfg.get("trap_cmd_smooth", default_smooth))
            elif task_name == "DribbleBallEnv":
                default_smooth = 0.35
                smooth_alpha = float(curr_cfg.get("dribble_cmd_smooth", default_smooth))
            else:
                default_smooth = 0.5
                smooth_alpha = default_smooth
            self.env.apply_high_level_command(cmd, smooth=smooth_alpha)
            return ("continuous", a_cont, cmd)

    def _update_window_rate(self, window, value: float, window_size: int) -> float:
        """维护滑窗并返回当前均值。"""
        window.append(float(value))  # 最近窗口的成功标记序列
        if len(window) > int(window_size):
            window.pop(0)
        return float(sum(window) / max(1, len(window)))

    def _tb_log_episode_metrics(
        self,
        tb,
        episode_return: float,
        episode_step: int,
        succ: float,
        succ_rate: float,
        final_succ: float = None,
        final_succ_rate: float = None,
        touch: float = None,
        touch_rate: float = None,
        init_dist: float = None,
        cur_r_max: float = None,
    ) -> None:
        """统一记录回合级指标，避免不同任务分支重复打点。"""
        tb.add_scalar("episode/return", float(episode_return))
        tb.add_scalar("episode/length", float(episode_step))
        tb.add_scalar("episode/success", float(succ))
        tb.add_scalar("episode/success_rate", float(succ_rate))
        if final_succ is not None:
            tb.add_scalar("episode/final_success", float(final_succ))
        if final_succ_rate is not None:
            tb.add_scalar("episode/final_success_rate", float(final_succ_rate))
        if touch is not None:
            tb.add_scalar("episode/touch", float(touch))
        if touch_rate is not None:
            tb.add_scalar("episode/touch_rate", float(touch_rate))
        if init_dist is not None:
            tb.add_scalar("episode/init_dist", float(init_dist))
        if cur_r_max is not None:
            tb.add_scalar("episode/cur_r_max", float(cur_r_max))

    def _tb_log_curriculum_metrics(self, tb, rmax: float, changed: bool, info: dict) -> None:
        """统一记录课程学习指标；stage 相关字段按存在性自动记录。"""
        tb.add_scalar("curr/r_max", float(rmax))
        if isinstance(info, dict):
            curr_map = {  # 课程主指标映射（info_key -> tb_tag）
                "rate_global": "curr/rate_global",
                "rate_global_raw": "curr/rate_global_raw",
                "rate_curr": "curr/rate_curr",
                "rate_curr_raw": "curr/rate_curr_raw",
                "reward_stage": "curr/reward_stage",
                "reward_stage_max": "curr/reward_stage_max",
                "stage_success_rate": "curr/stage_success_rate",
                "stage_success_rate_valid": "curr/stage_success_rate_valid",
                "episodes_in_reward_stage": "curr/episodes_in_reward_stage",
                "episode_stage_success": "curr/episode_stage_success",
                "episode_frontier_stage_success": "curr/episode_frontier_stage_success",
                "target_dist_min_curr": "curr/target_dist_min_curr",
                "target_dist_max_curr": "curr/target_dist_max_curr",
                "target_angle_abs_min_deg_curr": "curr/target_angle_abs_min_deg_curr",
                "target_angle_abs_max_deg_curr": "curr/target_angle_abs_max_deg_curr",
                "ball_spawn_dist_min_curr": "curr/ball_spawn_dist_min_curr",
                "ball_spawn_dist_max_curr": "curr/ball_spawn_dist_max_curr",
                "ball_spawn_lateral_abs_curr": "curr/ball_spawn_lateral_abs_curr",
                "episode_target_dist": "curr/episode_target_dist",
                "episode_target_angle_deg": "curr/episode_target_angle_deg",
                "episode_ball_spawn_dist": "curr/episode_ball_spawn_dist",
                "episode_ball_spawn_lateral": "curr/episode_ball_spawn_lateral",
                "max_control_streak": "curr/max_control_streak",
                "control_streak_score": "curr/control_streak_score",
                "control_streak_target": "curr/control_streak_target",
            }
            for k, tag in curr_map.items():
                if k in info:
                    v = float(info[k])
                    # 空样本时 stage_success_rate 可能为 NaN；跳过 NaN，避免图上出现“假 0”
                    if not np.isfinite(v):
                        continue
                    tb.add_scalar(tag, v)
            if "reward_stage_changed" in info:
                tb.add_scalar("curr/reward_stage_changed", 1.0 if info["reward_stage_changed"] else 0.0)
        tb.add_scalar("curr/changed", 1.0 if changed else 0.0)

    def _tb_log_update_losses(self, tb, q1_loss: float, q2_loss: float, pi_loss: float, alpha_loss: float, alpha: float) -> None:
        """统一记录 actor-critic 训练损失。"""
        tb.add_scalar("train/q1_loss", q1_loss)
        tb.add_scalar("train/q2_loss", q2_loss)
        tb.add_scalar("train/policy_loss", pi_loss)
        tb.add_scalar("train/alpha_loss", alpha_loss)
        tb.add_scalar("train/alpha", alpha)

    def _tb_log_passball_cn_notes(self, tb) -> None:
        """在 TensorBoard 中补充 PassBall 常用指标中文说明。"""
        tb.add_text(
            "中文注释/PassBall指标说明",
            (
                "### PassBall 常看指标\n"
                "- `high/reward`：高层每步平均奖励\n"
                "- `episode/success_rate`：最近窗口阶段成功率\n"
                "- `episode/final_success_rate`：最近窗口最终成功率\n"
                "- `episode/touch_rate`：最近窗口触球率\n"
                "- `curr/r_max`：课程当前采样半径上限\n"
                "- `train/replay_size`：经验池样本数\n"
                "\n"
                "同时提供了 `中文注释/...` 的同值曲线，便于直接按中文查看。"
            ),
        )

    def _save_camera_frames_as_video(self, frames, out_path: str, fps: int = 50) -> int:
        """将 Isaac Gym 相机帧列表保存成 mp4，返回写入帧数。"""
        if not frames:
            return 0
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        video_frames = []
        for fr in frames:
            arr = np.asarray(fr)
            if arr.ndim != 3 or arr.shape[-1] < 3:
                continue
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            video_frames.append(rgb)
        if not video_frames:
            return 0
        imageio.mimsave(out_path, video_frames, fps=int(fps), macro_block_size=1)
        return len(video_frames)

    def _record_passball_eval_video(
        self,
        agent,
        action_mode: str,
        global_step: int,
        seconds: float = 12.0,
        action_repeat: int = 10,
        gait_freq: float = 1.5,
        smooth: float = 0.5,
        fps: int = 50,
    ) -> dict:
        """
        在当前训练过程中录制一段固定时长评估视频（默认 12s）。
        为避免打断训练语义，仅用于观测当前策略效果，不写入 replay。
        """
        prev_record_video = bool(self.cfg["viewer"].get("record_video", False))
        self.cfg["viewer"]["record_video"] = True
        # controller 与 runner 共享同一 cfg，但这里显式写回，避免后续引用不同对象
        self.env.controller.cfg["viewer"]["record_video"] = True

        try:
            if hasattr(self.env.controller, "camera_frames"):
                self.env.controller.camera_frames = []

            obs, infos = self.env.reset()
            obs = obs.to(self.device)
            self._set_target_for_pass()

            low_dt = float(self.env.dt)
            high_dt = low_dt * int(action_repeat)
            n_high_steps = max(1, int(np.ceil(float(seconds) / max(1e-6, high_dt))))

            success = False
            fall = False
            with torch.no_grad():
                for _ in range(n_high_steps):
                    obs_high = self.env.compute_midlevel_obs()
                    if torch.is_tensor(obs_high) and obs_high.device != torch.device(self.device):
                        obs_high = obs_high.to(self.device)
                    obs_high_np = obs_high.squeeze(0).detach().cpu().numpy()

                    if action_mode == "discrete":
                        action_id = agent.select_action(obs_high_np)
                        cmd = self.env.high_level_action_id_to_vector(action_id)
                        self.env.apply_high_level_command(cmd)
                    else:
                        a_cont = agent.select_action(obs_high_np, eval_mode=True)
                        cmd = [float(a_cont[0]), float(a_cont[1]), float(a_cont[2]), float(gait_freq)]
                        self.env.apply_high_level_command(cmd, smooth=float(smooth))

                    for _ in range(int(action_repeat)):
                        obs_mod = obs.clone()
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
                        act = self._low_policy_action(obs_mod)
                        obs, rew, done, infos = self.env.step(act)
                        if torch.is_tensor(obs) and obs.device != torch.device(self.device):
                            obs = obs.to(self.device)
                        if isinstance(infos, dict):
                            success = success or bool(infos.get("success", False))
                            fall = fall or bool(infos.get("fall", False))

            out_dir = os.path.join("logs", "videos", "passBall")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"train_eval_step_{int(global_step)}.mp4")
            frames = getattr(self.env.controller, "camera_frames", None)
            frame_count = self._save_camera_frames_as_video(frames, out_path=out_path, fps=int(fps))
            return {
                "path": out_path,
                "frames": int(frame_count),
                "success": bool(success),
                "fall": bool(fall),
                "seconds": float(seconds),
            }
        finally:
            self.cfg["viewer"]["record_video"] = prev_record_video
            self.env.controller.cfg["viewer"]["record_video"] = prev_record_video
            if hasattr(self.env.controller, "camera_frames"):
                self.env.controller.camera_frames = []

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
        agent, global_step = self._resume_high_level_if_needed(agent)
        
        episode_step = 0
        episode_return = 0
        max_steps = 500
        episode_idx = 0
        EP_RATE_WIN = 200  # 回合成功率滑窗长度
        ep_success_window = []  # 阶段成功率滑窗（episode/success_rate）

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
                final_success_happened = False
                fall_happened = False
                tb.set_step(global_step)
                tb.add_scalar("train/env_frames", global_step * ACTION_REPEAT)

                for _ in range(ACTION_REPEAT):
                    with torch.no_grad():
                        obs_mod = obs.clone()
                        obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = (
                            action_cmd[0], action_cmd[1], action_cmd[2]
                        )
                        act = self._low_policy_action(obs_mod)
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
                    if isinstance(infos, dict) and infos.get("final_success", False):
                        final_success_happened = True
                        tb.add_scalar("events/final_success", 1.0)
                    if success_happened:
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
                    succ_rate = self._update_window_rate(ep_success_window, succ, EP_RATE_WIN)
                    self._tb_log_episode_metrics(
                        tb=tb,
                        episode_return=episode_return,
                        episode_step=episode_step,
                        succ=succ,
                        succ_rate=succ_rate,
                        init_dist=float(self.env.get_initial_dist_xy()),
                        cur_r_max=float(self.env.cur_r_max),
                    )
                    print(f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                          f"Step: {episode_step} | Success: {bool(succ)} | "
                          f"InitDist: {self.env.get_initial_dist_xy():.2f}")

                    # === 新：基于“单回合结果”的课程更新（env 内部处理防连跳/冷却/驻留/比例步长） ===
                    try:
                        rmin, rmax, changed, info = self.env.on_episode_end(
                            success=(success_happened and not fall_happened),
                            episode_idx=episode_idx
                        )
                        self._tb_log_curriculum_metrics(tb=tb, rmax=rmax, changed=changed, info=info)
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
                            self._tb_log_update_losses(tb, q1_loss, q2_loss, pi_loss, alpha_loss, alpha)

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
        ckpt_dir = os.path.join("logs", "ckpt", "passBall", "sac", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[CKPT] passBall checkpoints will be saved to: {ckpt_dir}")
        self._tb_log_passball_cn_notes(tb)
        global_step = 0

        # ----- 可调参数 -----
        ACTION_REPEAT = 5  # 高层动作重复次数
        WARMUP = 5000       # 高层 agent warmup 步数
        UPDATE_K = 1        # 每个高层 step 更新次数
        VIDEO_EVERY = 10000  # 每多少个高层 step 导出一次评估视频
        VIDEO_SECONDS = 12.0
        VIDEO_FPS = 50
        VIDEO_GAIT_FREQ = 1.
        VIDEO_SMOOTH = 0.5

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
        agent, global_step = self._resume_high_level_if_needed(agent)

        episode_step = 0
        episode_return = 0.0
        max_steps = 300
        episode_idx = 0
        hit_happened = False        # 记录是否已触球
        before_hit_indices = []     # 记录本回合所有 before hit 的 buffer 索引
        EP_RATE_WIN = 200  # 回合成功率滑窗长度
        ep_success_window = []  # 阶段成功率滑窗（当前采样阶段 success）
        ep_final_success_window = []  # 最终成功率滑窗（precise success）
        ep_touch_window = []  # 触球率滑窗（episode/touch_rate）
        video_pending_step = None

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
                final_success_happened = False
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
                            return self._low_policy_action(obs_mod)
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
                        if not fall_happened:
                            fall_happened = True
                            tb.add_scalar("events/fallen", 1.0)
                        break

                    # ✅ success 判定：默认看当前阶段成功，同时单独记录最终成功
                    if isinstance(infos, dict) and infos.get("success", False) and (not success_happened):
                        success_happened = True
                        tb.add_scalar("events/success", 1.0)
                    if isinstance(infos, dict) and infos.get("final_success", False) and (not final_success_happened):
                        final_success_happened = True
                        tb.add_scalar("events/final_success", 1.0)
                    if success_happened:
                        break

                    # 新增：检测触球事件
                    if isinstance(infos, dict) and infos.get("hit", False) and (not hit_happened):
                        hit_happened = True
                        tb.add_scalar("events/hit", 1.0)
                        # 触球后的轨迹预测仅做诊断，不再作为成功判定或提前结束条件
                        if hasattr(self.env, "predict_success_after_hit"):
                            will_succeed, final_dist, s_max, v0 = self.env.predict_success_after_hit()
                            hit_pred_success = bool(will_succeed)
                            tb.add_scalar("events/hit_pred_success", float(hit_pred_success))
                            tb.add_scalar("diag/hit_pred_final_dist", float(final_dist))
                            tb.add_scalar("diag/hit_pred_s_max", float(s_max))
                            tb.add_scalar("diag/hit_pred_v0", float(v0))

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
                done_high = (episode_step > max_steps) or success_happened or fall_happened

                # 经验入池，带 note
                note = "after hit" if hit_happened else "before hit"
                curr_difficulty = float(self.env.get_initial_dist_xy())
                curr_success = bool(success_happened and not fall_happened)
                _t_push = time.perf_counter()
                if mode == "discrete":
                    agent.replay_buffer.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        int(action_repr),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note=note,
                        difficulty=curr_difficulty,
                        success=curr_success,
                    )
                else:
                    agent.replay_buffer.push(
                        obs_high.squeeze(0).cpu().numpy(),
                        np.asarray(action_repr, dtype=np.float32),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note=note,
                        difficulty=curr_difficulty,
                        success=curr_success,
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
                tb.add_scalar("中文注释/高层奖励(high/reward)", float(rew_high))
                tb.add_scalar("中文注释/经验池大小(train/replay_size)", float(len(agent.replay_buffer)))

                # ---------- TensorBoard：reward 分解（与 PassBallEnv.rew_terms 保持一致） ----------
                if isinstance(last_infos, dict):
                    terms = last_infos.get("rew_terms", {})
                    if isinstance(terms, dict):
                        # PassBallEnv 当前输出字段（reward 相关）
                        rew_keys = tuple(dict.fromkeys((  # 去重后再打点，避免同名指标重复写入
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
                            "orbit_cos",
                            "orbit_reward",
                            "orbit_radius_reward",
                            "orbit_tangent_cos",
                            "orbit_tangent_reward",
                            "orbit_need_turn_gate",
                            "orbit_radius_in_band",
                            "radial_speed",
                            "approach_cos",
                            "approach_reward",
                            "pass_seg_dist",
                            "touch_count",
                            "touch_event",
                            "pass_target_now",
                            "single_touch_ok",
                            "align_hold_counter",
                            "align_hold_steps",
                            "stage2_stop_now",
                            "stage2_stop_ready",
                            "stage2_stop_hold_counter",
                            "stage2_stop_hold_steps",
                            "stop_gate",
                            "stop_ready_gate",
                            "prep_gate_quality",
                            "r_robot",
                            "r_approach",
                            "r_line",
                            "r_near_ball",
                            "r_ball",
                            "r_post_align",
                            "r_post_dist",
                            "r_progress",
                            "r_touch_soft",
                            "r_touch_hard",
                            "r_touch",
                            "r_release_bonus",
                            "r_stage01_touch",
                            "r_stage2_prep",
                            "r_stage2_stop",
                            "r_orbit",
                            "r_long_kick",
                            "r_right_foot",
                            "r_align_bonus",
                            "r_touch_stage_bonus",
                            "r_succ",
                            "far_penalty",
                            "hack_penalty",
                            "side_hit_penalty",
                            "time_penalty",
                            "fallen_penalty",
                            "reward_stage",
                            "reward_stage_max",
                            "reward_stage_pre_scale",
                            "reward_stage_post_scale",
                            "reward_stage_touch_scale",
                            "reward_stage_touch_stage_scale",
                            "reward_stage_align_scale",
                            "reward_stage_success_scale",
                            "align_success",
                            "stage0_align_success",
                            "stage1_close_success",
                            "coarse_success",
                            "stage1_align_success",
                            "stage2_touch_success",
                            "precise_success",
                            "stage_success_active",
                            "kick_dir_gate",
                            "left_foot_ball_dist",
                            "right_foot_ball_dist",
                        )))
                        for k in rew_keys:
                            if k in terms:
                                tb.add_scalar(f"rew/{k}", float(terms[k]))

                        # PassBallEnv 当前输出字段（事件相关）
                        event_keys = (
                            "touch_now",
                            "touch_dir_good",
                            "left_touch_now",
                            "right_touch_now",
                            "first_touch",
                        )
                        for k in event_keys:
                            if k in terms:
                                tb.add_scalar(f"events/{k}", float(terms[k]))
                global_step += 1
                if global_step % VIDEO_EVERY == 0:
                    video_pending_step = int(global_step)

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
                    ckpt_path = os.path.join(ckpt_dir, f"sac_agent_step_{global_step}.pt")
                    torch.save({"agent": agent, "global_step": global_step, "cfg": self.cfg}, ckpt_path)
                    print(f"[Save] SAC agent saved at step {global_step} -> {ckpt_path}")

                # ---------- 回合结束 ----------
                if done_high:
                    succ = 1.0 if (success_happened and not fall_happened) else 0.0
                    final_succ = 1.0 if (final_success_happened and not fall_happened) else 0.0
                    touch = 1.0 if hit_happened else 0.0
                    if isinstance(last_infos, dict):
                        terms = last_infos.get("rew_terms", {})
                        if isinstance(terms, dict) and ("touch_count" in terms):
                            try:
                                touch_count_ep = float(terms["touch_count"])
                                touch = 1.0 if touch_count_ep > 0.0 else touch
                            except Exception:
                                pass

                    succ_rate = self._update_window_rate(ep_success_window, succ, EP_RATE_WIN)
                    final_succ_rate = self._update_window_rate(ep_final_success_window, final_succ, EP_RATE_WIN)
                    touch_rate = self._update_window_rate(ep_touch_window, touch, EP_RATE_WIN)
                    self._tb_log_episode_metrics(
                        tb=tb,
                        episode_return=episode_return,
                        episode_step=episode_step,
                        succ=succ,
                        succ_rate=succ_rate,
                        final_succ=final_succ,
                        final_succ_rate=final_succ_rate,
                        touch=touch,
                        touch_rate=touch_rate,
                    )
                    tb.add_scalar("中文注释/阶段成功率(episode/success_rate)", float(succ_rate))
                    tb.add_scalar("中文注释/最终成功率(episode/final_success_rate)", float(final_succ_rate))
                    tb.add_scalar("中文注释/触球率(episode/touch_rate)", float(touch_rate))
                    tb.add_scalar("中文注释/回合回报(episode/return)", float(episode_return))
                    print(f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                        f"Step: {episode_step} | Success: {bool(succ)}")

                    # 回溯式奖励：如果成功，给最后一个 before hit transition 加成功奖励
                    if succ == 1.0 and before_hit_indices:
                        last_idx = before_hit_indices[-1]
                        if 0 <= last_idx < len(agent.replay_buffer.buf):
                            transition = agent.replay_buffer.buf[last_idx]
                            # transition 结构: (s, a, r, s2, d, note, difficulty, success)
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

                            # 结构: (s, a, r, s2, d, note, difficulty, success)
                            obs, act, rew, next_obs, done, note, difficulty, success = agent.replay_buffer.buf[buf_idx]

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
                                difficulty=difficulty,
                                success=False,
                            )
                            added += 1

                        print(f"[HER] Added {added} HER transitions (sampled {num_her}).")
                    # === 课程更新：r_max 采样范围只由最终成功率驱动，避免阶段成功把难度推高 ===
                    try:
                        rmin, rmax, changed, info = self.env.on_episode_end(
                            success=(final_success_happened and not fall_happened),
                            episode_idx=episode_idx
                        )
                        self._tb_log_curriculum_metrics(tb=tb, rmax=rmax, changed=changed, info=info)
                        tb.add_scalar("中文注释/课程半径上限(curr/r_max)", float(rmax))
                        if changed and isinstance(info, dict) and info.get("reason"):
                            print(f"[Curriculum] ep#{episode_idx} r_max -> {rmax:.2f} | {info['reason']}")
                    except Exception as e:
                        print("[Curriculum] update failed:", e)

                    # ---------- 每 10k step 录制一段评估视频（12s） ----------
                    if video_pending_step is not None:
                        try:
                            video_info = self._record_passball_eval_video(
                                agent=agent,
                                action_mode=action_mode,
                                global_step=video_pending_step,
                                seconds=VIDEO_SECONDS,
                                action_repeat=ACTION_REPEAT,
                                gait_freq=VIDEO_GAIT_FREQ,
                                smooth=VIDEO_SMOOTH,
                                fps=VIDEO_FPS,
                            )
                            tb.set_step(global_step)
                            tb.add_scalar("eval_video/frames", float(video_info["frames"]))
                            tb.add_scalar("eval_video/success", 1.0 if video_info["success"] else 0.0)
                            tb.add_scalar("eval_video/fall", 1.0 if video_info["fall"] else 0.0)
                            tb.add_scalar("中文注释/评估视频帧数(eval_video/frames)", float(video_info["frames"]))
                            if int(video_info["frames"]) > 0:
                                print(
                                    f"[EvalVideo] step={video_pending_step} | "
                                    f"frames={video_info['frames']} | path={os.path.abspath(video_info['path'])}"
                                )
                            else:
                                print(
                                    f"[EvalVideo] step={video_pending_step} | "
                                    "frames=0 (请检查图形环境/离屏渲染配置)"
                                )
                        except Exception as e:
                            print(f"[EvalVideo] export failed at step {video_pending_step}: {e}")
                        finally:
                            video_pending_step = None

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
                                self._tb_log_update_losses(tb, q1_loss, q2_loss, pi_loss, alpha_loss, alpha)

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
        task_name = str(self.cfg["basic"].get("task", ""))
        task_log_dir = self.task_name_dir if self.task_name_dir else ("trapBall" if task_name == "TrapBallEnv" else "dribbleBall")
        replay_note = "trap" if task_name == "TrapBallEnv" else "dribble"
        action_repeat_key = "trap_action_repeat" if task_name == "TrapBallEnv" else "dribble_action_repeat"
        default_action_repeat = 4 if task_name == "TrapBallEnv" else 5

        ckpt_dir = os.path.join("logs", "ckpt", task_log_dir, "sac", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[CKPT] {task_log_dir} checkpoints will be saved to: {ckpt_dir}")
        global_step = 0

        curr_cfg = self.env.controller.cfg.get("curriculum", {})
        ACTION_REPEAT = max(1, int(curr_cfg.get(action_repeat_key, default_action_repeat)))
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
        agent, global_step = self._resume_high_level_if_needed(agent)

        episode_step = 0
        episode_return = 0.0
        max_steps = 300
        episode_idx = 0
        EP_RATE_WIN = 200  # 回合成功率滑窗长度
        ep_success_window = []  # 任务成功率滑窗

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
                            return self._low_policy_action(obs_mod)

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
                        note=replay_note,
                    )
                else:
                    agent.replay_buffer.push(
                        obs_high_np,
                        np.asarray(action_repr, dtype=np.float32),
                        rew_high,
                        next_obs_high_np,
                        done_high,
                        note=replay_note,
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
                            filename=f"{replay_note}_replay_step_{global_step}_N{len(agent.replay_buffer)}.npz"
                        )
                    except Exception as e:
                        print("[ReplayBuffer] save failed:", e)

                if (global_step % 10000 == 0) and (len(agent.replay_buffer) >= WARMUP):
                    ckpt_path = os.path.join(ckpt_dir, f"sac_agent_step_{global_step}.pt")
                    torch.save({"agent": agent, "global_step": global_step, "cfg": self.cfg}, ckpt_path)
                    print(f"[Save] SAC agent saved at step {global_step} -> {ckpt_path}")

                if done_high:
                    succ = 1.0 if (success_happened and not fall_happened and not fail_happened) else 0.0
                    succ_rate = self._update_window_rate(ep_success_window, succ, EP_RATE_WIN)
                    self._tb_log_episode_metrics(
                        tb=tb,
                        episode_return=episode_return,
                        episode_step=episode_step,
                        succ=succ,
                        succ_rate=succ_rate,
                        init_dist=float(self.env.get_initial_dist_xy()),
                    )
                    print(
                        f"[Episode End] ep#{episode_idx} | Return: {episode_return:.2f} | "
                        f"Step: {episode_step} | Success: {bool(succ)} | "
                        f"Fail: {bool(fail_happened)} | Fall: {bool(fall_happened)} | "
                        f"InitDist: {self.env.get_initial_dist_xy():.2f}"
                    )

                    # 课程更新：按回合成功率更新任务难度（由各环境自行实现）
                    try:
                        rmin, rmax, changed, info = self.env.on_episode_end(
                            success=(success_happened and not fail_happened and not fall_happened),
                            episode_idx=episode_idx,
                        )
                        self._tb_log_curriculum_metrics(tb=tb, rmax=rmax, changed=changed, info=info)
                        if changed and isinstance(info, dict) and info.get("reason"):
                            print(f"[Curriculum] ep#{episode_idx} level -> {rmax:.2f} | {info['reason']}")
                    except Exception as e:
                        print("[Curriculum] update failed:", e)

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
                                self._tb_log_update_losses(tb, q1_loss, q2_loss, pi_loss, alpha_loss, alpha)

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

    def dribbleBall(self):
        return self.trapBall()
