import isaacgym  # noqa: F401
import os
import glob
import time
import argparse
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
import yaml

from envs import TrapBallEnv
from utils.model import ActorCritic
from core.agents.sac.agent import SACAgent


def _resolve_low_ckpt(low_ckpt_arg: Optional[str], cfg: dict) -> Optional[str]:
    ckpt = low_ckpt_arg if low_ckpt_arg is not None else cfg["basic"].get("checkpoint", None)
    if ckpt in (None, "", "null"):
        return None
    if ckpt in ("-1", -1):
        candidates = sorted(glob.glob(os.path.join("logs", "low", "**", "*.pth"), recursive=True), key=os.path.getmtime)
        if not candidates:
            # 兼容旧目录
            candidates = sorted(glob.glob(os.path.join("logs", "**", "*.pth"), recursive=True), key=os.path.getmtime)
        if not candidates:
            return None
        return candidates[-1]
    return str(ckpt)


def _resolve_high_ckpt(high_ckpt_arg: str) -> str:
    if high_ckpt_arg not in ("-1", -1):
        if not os.path.isfile(high_ckpt_arg):
            raise FileNotFoundError(f"High-level checkpoint not found: {high_ckpt_arg}")
        return str(high_ckpt_arg)

    # 兼容平铺与 run_name 子目录
    cands = list(glob.glob(os.path.join("logs", "ckpt", "trapBall", "sac", "sac_agent_step_*.pt")))
    cands.extend(glob.glob(os.path.join("logs", "ckpt", "trapBall", "sac", "**", "sac_agent_step_*.pt"), recursive=True))
    cands = [p for p in cands if os.path.getsize(p) > 0]
    cands = sorted(set(cands), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError("No non-empty high-level checkpoints found under logs/ckpt/trapBall/sac/")
    return cands[-1]


def _build_sac_agent_for_trap(env: TrapBallEnv, device: str) -> SACAgent:
    obs_high = env.compute_midlevel_obs().to(device)
    state_dim = int(obs_high.shape[1])
    action_dim = 3

    curr_cfg = env.controller.cfg.get("curriculum", {})
    vx_range = np.array([0.0, 0.6], dtype=np.float32)
    vy_range = np.array([-0.35, 0.35], dtype=np.float32)
    yaw_range = np.array([-1.0, 1.0], dtype=np.float32)
    act_low = np.array([vx_range[0], vy_range[0], yaw_range[0]], dtype=np.float32)
    act_high = np.array([vx_range[1], vy_range[1], yaw_range[1]], dtype=np.float32)

    return SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        action_low=act_low,
        action_high=act_high,
        buffer_capacity=200000,
        batch_size=256,
        lr=3e-4,
        gamma=0.90,
        tau=0.005,
        alpha=None,
        sample_sigma=float(curr_cfg.get("sample_sigma", 0.2)),
        sample_epsilon=float(curr_cfg.get("sample_epsilon", 0.1)),
        sample_success_bonus=float(curr_cfg.get("sample_success_bonus", 1.2)),
        sample_hard_focus=float(curr_cfg.get("sample_hard_focus", 0.0)),
        sample_rmax_ema_beta=float(curr_cfg.get("sample_rmax_ema_beta", 0.9)),
    )


def _sac_io_dims(agent: SACAgent) -> Tuple[Optional[int], Optional[int]]:
    state_dim = None
    action_dim = None
    try:
        state_dim = int(agent.policy.net[0].in_features)
    except Exception:
        pass
    try:
        action_dim = int(agent.policy.mu.out_features)
    except Exception:
        pass
    return state_dim, action_dim


def _load_high_agent(high_ckpt: str, env: TrapBallEnv, device: str) -> SACAgent:
    pkg = torch.load(high_ckpt, map_location=device, weights_only=False)

    if isinstance(pkg, dict) and "agent" in pkg:
        agent = pkg["agent"]
        agent.device = device
        for m in ("policy", "q1", "q2", "q1_target", "q2_target"):
            if hasattr(agent, m):
                getattr(agent, m).to(device)
        if hasattr(agent, "action_low") and torch.is_tensor(agent.action_low):
            agent.action_low = agent.action_low.to(device)
        if hasattr(agent, "action_high") and torch.is_tensor(agent.action_high):
            agent.action_high = agent.action_high.to(device)
        if hasattr(agent, "policy"):
            agent.policy.eval()
    elif isinstance(pkg, dict) and "policy" in pkg:
        agent = _build_sac_agent_for_trap(env, device)
        agent.load(high_ckpt)
        agent.policy.eval()
    else:
        raise ValueError(f"Unsupported high-level checkpoint format: {high_ckpt}")

    exp_s = int(env.compute_midlevel_obs().shape[1])
    got_s, got_a = _sac_io_dims(agent)
    if got_s is not None and got_s != exp_s:
        raise ValueError(
            f"High-level checkpoint obs dim mismatch: env={exp_s}, ckpt={got_s}. "
            f"Checkpoint: {high_ckpt}"
        )
    if got_a is not None and got_a != 3:
        raise ValueError(f"High-level checkpoint action dim mismatch: expect 3, got {got_a}")
    return agent


def _load_low_policy(low_ckpt: Optional[str], env: TrapBallEnv, device: str):
    clip_actions = float(env.controller.cfg["normalization"]["clip_actions"])

    if low_ckpt is None:
        print("[WARN] No low-level checkpoint provided/found. Low-level policy uses random init.")
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        model.eval()
        return {"kind": "actor_critic", "module": model, "clip_actions": clip_actions, "source": "random_init"}

    low_ckpt = str(low_ckpt)
    if low_ckpt.lower().endswith(".pth"):
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        model.eval()
        print(f"[Load] low-level .pth ckpt: {low_ckpt}")
        low_dict = torch.load(low_ckpt, map_location=device, weights_only=True)
        if not (isinstance(low_dict, dict) and "model" in low_dict):
            raise ValueError(f"Invalid low-level .pth format (expect dict['model']): {low_ckpt}")
        model.load_state_dict(low_dict["model"], strict=False)
        return {"kind": "actor_critic", "module": model, "clip_actions": clip_actions, "source": low_ckpt}

    if low_ckpt.lower().endswith(".pt"):
        print(f"[Load] low-level TorchScript .pt ckpt: {low_ckpt}")
        try:
            model = torch.jit.load(low_ckpt, map_location=device)
            model.eval()
            test_out = model(torch.zeros(1, env.num_obs, device=device, dtype=torch.float32))
            if int(test_out.shape[-1]) != int(env.num_actions):
                raise ValueError(
                    f"Low-level TorchScript output dim mismatch: got {tuple(test_out.shape)}, "
                    f"expected (*, {env.num_actions})"
                )
            return {"kind": "jit", "module": model, "clip_actions": clip_actions, "source": low_ckpt}
        except Exception as jit_err:
            # 某些 .pt 不是 TorchScript（例如高层 SAC 包），这里回退做格式识别并给出明确提示
            try:
                low_obj = torch.load(low_ckpt, map_location=device, weights_only=True)
            except Exception:
                low_obj = torch.load(low_ckpt, map_location=device, weights_only=False)

            if isinstance(low_obj, dict) and "model" in low_obj:
                model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
                model.eval()
                model.load_state_dict(low_obj["model"], strict=False)
                return {"kind": "actor_critic", "module": model, "clip_actions": clip_actions, "source": low_ckpt}

            if isinstance(low_obj, dict) and "agent" in low_obj:
                raise ValueError(
                    f"Invalid --low_ckpt: {low_ckpt}\n"
                    "This file is a high-level SAC checkpoint (contains key 'agent'), not a low-level policy.\n"
                    "Use it with --high_ckpt, and provide a low-level locomotion checkpoint for --low_ckpt.\n"
                    "For example: --low_ckpt deploy/models/T1.pt"
                ) from jit_err

            keys_preview = list(low_obj.keys())[:10] if isinstance(low_obj, dict) else None
            raise ValueError(
                f"Unsupported low-level .pt format: {low_ckpt}\n"
                f"TorchScript load error: {repr(jit_err)}\n"
                f"Detected object type: {type(low_obj)}"
                + (f", keys preview: {keys_preview}" if keys_preview is not None else "")
            ) from jit_err

    raise ValueError(f"Unsupported low-level checkpoint suffix (need .pth or .pt): {low_ckpt}")


def _low_policy_act(low_policy: dict, obs_mod: torch.Tensor) -> torch.Tensor:
    kind = str(low_policy["kind"])
    model = low_policy["module"]
    clip_actions = float(low_policy["clip_actions"])

    if kind == "actor_critic":
        dist = model.act(obs_mod)
        return dist.loc

    if kind == "jit":
        act = model(obs_mod)
        act = torch.as_tensor(act, device=obs_mod.device, dtype=obs_mod.dtype)
        return torch.clip(act, -clip_actions, clip_actions)

    raise ValueError(f"Unknown low policy kind: {kind}")


def _save_video_from_camera_frames(env: TrapBallEnv, out_path: str, fps: int) -> int:
    frames = getattr(env.controller, "camera_frames", None)
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


def main():
    parser = argparse.ArgumentParser(description="TrapBall inference in Isaac Gym with video export")
    parser.add_argument("--task", type=str, default="TrapBallEnv", help="Task class name (default: TrapBallEnv)")
    parser.add_argument("--cfg", type=str, default="envs/trapBall/TrapBallEnv.yaml", help="Path to trapBall yaml")
    parser.add_argument(
        "--low_ckpt",
        type=str,
        default=None,
        help="Low-level checkpoint: .pth (ActorCritic) or .pt (TorchScript, e.g. deploy/models/T1.pt). Use -1 for latest logs/low/**/*.pth",
    )
    parser.add_argument("--high_ckpt", type=str, required=True, help="High-level SAC checkpoint (.pt). Use -1 for latest non-empty")
    parser.add_argument("--sim_device", type=str, default=None, help="Override sim device, e.g. cuda:0")
    parser.add_argument("--rl_device", type=str, default=None, help="Override RL device, e.g. cuda:0")
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument("--seconds", type=float, default=12.0, help="Inference wall-clock in sim-time seconds")
    parser.add_argument("--action_repeat", type=int, default=10, help="High-level action repeat")
    parser.add_argument("--gait_freq", type=float, default=1.5, help="Fixed gait frequency")
    parser.add_argument("--smooth", type=float, default=0.5, help="High-level command smoothing alpha")
    parser.add_argument("--spawn_dist_min", type=float, default=None, help="Ball spawn distance min (m) for this infer run")
    parser.add_argument("--spawn_dist_max", type=float, default=None, help="Ball spawn distance max (m) for this infer run")
    parser.add_argument("--spawn_lateral_abs", type=float, default=None, help="Ball lateral spawn abs range (m), i.e. [-x, x]")
    parser.add_argument("--ball_speed_min", type=float, default=None, help="Ball initial speed min (m/s) for this infer run")
    parser.add_argument("--ball_speed_max", type=float, default=None, help="Ball initial speed max (m/s) for this infer run")
    parser.add_argument(
        "--angle_levels_deg",
        type=str,
        default=None,
        help="Override curriculum angle levels, e.g. '12,16,20,24,30,36,42'",
    )
    parser.add_argument(
        "--angle_initial_level",
        type=int,
        default=None,
        help="Override curriculum initial angle level index",
    )
    parser.add_argument(
        "--angle_max_deg",
        type=float,
        default=None,
        help="Pin infer to a single curriculum angle level (equivalent to angle_levels_deg=[x], angle_initial_level=0)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS")
    parser.add_argument(
        "--stop_on_event",
        action="store_true",
        help="If set, stop rollout early when success/fail/fall/done appears. Default: run full seconds.",
    )
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args.sim_device is not None:
        cfg["basic"]["sim_device"] = args.sim_device
    if args.rl_device is not None:
        cfg["basic"]["rl_device"] = args.rl_device
    cfg["basic"]["headless"] = bool(args.headless)
    cfg["viewer"]["record_video"] = True
    cfg.setdefault("curriculum", {})
    if args.spawn_dist_min is not None:
        cfg["curriculum"]["trap_spawn_dist_min"] = float(args.spawn_dist_min)
    if args.spawn_dist_max is not None:
        cfg["curriculum"]["trap_spawn_dist_max"] = float(args.spawn_dist_max)
    if args.spawn_lateral_abs is not None:
        cfg["curriculum"]["trap_spawn_lateral_abs"] = float(args.spawn_lateral_abs)
    if args.ball_speed_min is not None:
        cfg["curriculum"]["trap_ball_speed_min"] = float(args.ball_speed_min)
    if args.ball_speed_max is not None:
        cfg["curriculum"]["trap_ball_speed_max"] = float(args.ball_speed_max)
    if args.angle_max_deg is not None:
        # 固定单一角度平台，仍复用课程学习同一采样逻辑 delta ~ U[-cur_r_max, cur_r_max]
        cfg["curriculum"]["angle_levels_deg"] = [float(args.angle_max_deg)]
        cfg["curriculum"]["angle_initial_level"] = 0
    else:
        if args.angle_levels_deg is not None:
            raw = [x.strip() for x in str(args.angle_levels_deg).split(",")]
            vals = [float(x) for x in raw if x != ""]
            if len(vals) == 0:
                raise ValueError("--angle_levels_deg is empty after parsing")
            cfg["curriculum"]["angle_levels_deg"] = vals
        if args.angle_initial_level is not None:
            cfg["curriculum"]["angle_initial_level"] = int(args.angle_initial_level)

    low_ckpt = _resolve_low_ckpt(args.low_ckpt, cfg)
    if low_ckpt is not None:
        cfg["basic"]["checkpoint"] = low_ckpt

    device = cfg["basic"]["rl_device"]
    target_xy = torch.zeros(2, device=device, dtype=torch.float32)

    task_class = eval(args.task)
    env: TrapBallEnv = task_class(cfg, target_xy)

    low_policy = _load_low_policy(low_ckpt, env, device)

    high_ckpt = _resolve_high_ckpt(args.high_ckpt)
    print(f"[Load] high-level ckpt: {high_ckpt}")
    high_agent = _load_high_agent(high_ckpt, env, device)

    obs, infos = env.reset()
    obs = obs.to(device)
    curr_cfg = env.controller.cfg.get("curriculum", {})
    print(
        "[SpawnAngle] "
        f"angle_levels_deg={curr_cfg.get('angle_levels_deg', None)}, "
        f"angle_initial_level={curr_cfg.get('angle_initial_level', None)}, "
        f"cur_r_max_deg={float(getattr(env, 'cur_r_max', float('nan'))):.2f}, "
        f"sampled_init_angle_deg={float(getattr(env, '_episode_init_angle_deg', float('nan'))):.2f}"
    )

    low_dt = float(env.dt)
    high_dt = low_dt * int(args.action_repeat)
    n_high_steps = max(1, int(np.ceil(float(args.seconds) / max(1e-6, high_dt))))

    print(
        f"[Run] seconds={args.seconds:.2f}, low_dt={low_dt:.4f}, action_repeat={args.action_repeat}, "
        f"high_steps={n_high_steps}"
    )

    success = False
    fail = False
    fall = False
    with torch.no_grad():
        for _ in range(n_high_steps):
            obs_high = env.compute_midlevel_obs().to(device)
            action = high_agent.select_action(obs_high.squeeze(0).detach().cpu().numpy(), eval_mode=True)
            cmd = [float(action[0]), float(action[1]), float(action[2]), float(args.gait_freq)]
            env.apply_high_level_command(cmd, smooth=float(args.smooth))

            for _ in range(int(args.action_repeat)):
                obs_mod = obs.clone()
                obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
                act = _low_policy_act(low_policy, obs_mod)
                obs, rew, done, infos = env.step(act)
                obs = obs.to(device)

                if isinstance(infos, dict):
                    success = success or bool(infos.get("success", False))
                    fail = fail or bool(infos.get("fail", False))
                    fall = fall or bool(infos.get("fall", False))
                should_stop = success or fail or fall or bool(torch.any(done).item())
                if bool(args.stop_on_event) and should_stop:
                    break
            if bool(args.stop_on_event) and (success or fail or fall):
                break

    if args.out is None:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        args.out = os.path.join("videos", f"trapball_infer_{ts}.mp4")

    frame_count = _save_video_from_camera_frames(env, args.out, fps=args.fps)

    print(f"[Done] success={success}, fail={fail}, fall={fall}, frames={frame_count}")
    if frame_count == 0:
        print("[WARN] No frames were captured. Check graphics availability and viewer.record_video settings.")
    else:
        print(f"[Video] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
