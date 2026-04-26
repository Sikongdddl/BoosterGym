import isaacgym  # noqa: F401
import os
import glob
import time
import argparse
from typing import Optional

import imageio
import numpy as np
import torch
import yaml
from isaacgym.torch_utils import quat_rotate

from envs import PassBallEnv
from utils.model import ActorCritic
from core.agents.sac.agent import SACAgent


def _resolve_low_ckpt(low_ckpt_arg: Optional[str], cfg: dict) -> Optional[str]:
    ckpt = low_ckpt_arg if low_ckpt_arg is not None else cfg["basic"].get("checkpoint", None)
    if ckpt in (None, "", "null"):
        return None
    if ckpt in ("-1", -1):
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

    pattern_flat = os.path.join("logs", "ckpt", "passBall", "sac", "sac_agent_step_*.pt")
    pattern_nested = os.path.join("logs", "ckpt", "passBall", "sac", "**", "sac_agent_step_*.pt")
    candidates = list(glob.glob(pattern_flat))
    candidates.extend(glob.glob(pattern_nested, recursive=True))
    candidates = sorted(set(candidates), key=os.path.getmtime)
    candidates = [p for p in candidates if os.path.getsize(p) > 0]
    if not candidates:
        raise FileNotFoundError("No non-empty high-level checkpoints found under logs/ckpt/passBall/sac/")
    return candidates[-1]


def _build_sac_agent_for_passball(env: PassBallEnv, device: str) -> SACAgent:
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


def _load_high_agent(high_ckpt: str, env: PassBallEnv, device: str):
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
        return agent

    # 兼容 weights-only 格式（SACAgent.save 产物）
    if isinstance(pkg, dict) and "policy" in pkg:
        agent = _build_sac_agent_for_passball(env, device)
        agent.load(high_ckpt)
        agent.policy.eval()
        return agent

    raise ValueError(f"Unsupported high-level checkpoint format: {high_ckpt}")


def _load_low_policy(low_ckpt: Optional[str], env: PassBallEnv, device: str):
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
        model = torch.jit.load(low_ckpt, map_location=device)
        model.eval()
        test_out = model(torch.zeros(1, env.num_obs, device=device, dtype=torch.float32))
        if int(test_out.shape[-1]) != int(env.num_actions):
            raise ValueError(
                f"Low-level TorchScript output dim mismatch: got {tuple(test_out.shape)}, "
                f"expected (*, {env.num_actions})"
            )
        return {"kind": "jit", "module": model, "clip_actions": clip_actions, "source": low_ckpt}

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


def _save_video_from_camera_frames(env: PassBallEnv, out_path: str, fps: int) -> int:
    frames = getattr(env.controller, "camera_frames", None)
    if not frames:
        return 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    video_frames = []
    for fr in frames:
        arr = np.asarray(fr)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            continue
        # Isaac Gym IMAGE_COLOR 通常是 RGBA/BGRA；写视频时统一取前 3 通道
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        video_frames.append(rgb)

    if not video_frames:
        return 0

    imageio.mimsave(out_path, video_frames, fps=int(fps), macro_block_size=1)
    return len(video_frames)


def _get_robot_heading(env: PassBallEnv) -> float:
    fwd_local = torch.tensor([1.0, 0.0, 0.0], device=env.base_quat.device, dtype=env.base_quat.dtype)
    fwd_world = quat_rotate(env.base_quat[0:1], fwd_local[None, :]).squeeze(0)
    return float(torch.atan2(fwd_world[1], fwd_world[0]).item())


def _set_infer_target(env: PassBallEnv, device: str, args):
    """
    设置 infer rollout 的 target：
    - train(default): 严格复用训练时 target 采样（前方锥形 + [cur_r_min, cur_r_max]）
    - random: 机器人前方扇形 + 课程半径窗口（可通过参数覆盖）
    - fixed: 世界坐标固定点
    - polar: 相对机器人(base)的极坐标
    - ball: 直接使用当前球位置
    """
    base_pos = env.base_pos[0, :3]
    base_x = float(base_pos[0].item())
    base_y = float(base_pos[1].item())

    mode = str(args.target_mode).lower()
    # target 半径下限（用于 train/random）
    curr_cfg = env.controller.cfg.get("curriculum", {})
    if args.target_r_floor is None:
        target_r_floor = float(curr_cfg.get("pass_target_r_min", 1.30))
    else:
        target_r_floor = float(args.target_r_floor)
    theta_half_deg_default = float(curr_cfg.get("pass_target_theta_half_deg", 30.0))
    heading = _get_robot_heading(env)
    if mode == "fixed":
        if args.target_x is None or args.target_y is None:
            raise ValueError("target_mode=fixed requires --target_x and --target_y")
        tx = float(args.target_x)
        ty = float(args.target_y)
        info = f"mode=fixed(world), target=({tx:.2f},{ty:.2f})"
    elif mode == "polar":
        if args.target_r is None:
            raise ValueError("target_mode=polar requires --target_r")
        theta_deg = 0.0 if args.target_theta_deg is None else float(args.target_theta_deg)
        theta = heading + np.deg2rad(theta_deg)
        radius = float(args.target_r)
        tx = float(base_x + radius * np.cos(theta))
        ty = float(base_y + radius * np.sin(theta))
        info = (
            f"mode=polar(base), base=({base_x:.2f},{base_y:.2f}), "
            f"r={radius:.2f}, heading={heading:.2f}, theta_deg={theta_deg:.1f}, target=({tx:.2f},{ty:.2f})"
        )
    elif mode == "ball":
        ball_pos, _, _ = env.ball_world.get_pose(env.root_states)
        tx = float(ball_pos[0].item())
        ty = float(ball_pos[1].item())
        info = f"mode=ball(world), target=({tx:.2f},{ty:.2f})"
    elif mode == "random":
        # 可配置随机：前方扇形 + 课程窗口（支持命令行覆盖）
        theta_min_deg = -theta_half_deg_default if args.target_theta_min_deg is None else float(args.target_theta_min_deg)
        theta_max_deg = theta_half_deg_default if args.target_theta_max_deg is None else float(args.target_theta_max_deg)
        if theta_max_deg < theta_min_deg:
            theta_min_deg, theta_max_deg = theta_max_deg, theta_min_deg
        theta = float(heading + np.random.uniform(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg)))

        r_min_raw = float(curr_cfg.get("pass_target_r_min", getattr(env, "cur_r_min", 1.3))) if args.target_r_min is None else float(args.target_r_min)
        r_max_raw = float(curr_cfg.get("pass_target_r_max", getattr(env, "cur_r_max", 1.6))) if args.target_r_max is None else float(args.target_r_max)
        r_min = max(r_min_raw, target_r_floor)
        r_max = max(r_max_raw, r_min)
        radius = float(np.random.uniform(r_min, r_max))

        tx = float(base_x + radius * np.cos(theta))
        ty = float(base_y + radius * np.sin(theta))
        info = (
            f"mode=random(base), base=({base_x:.2f},{base_y:.2f}), "
            f"R={radius:.2f} in [{r_min:.2f},{r_max:.2f}], "
            f"heading={heading:.2f}, theta_deg={np.rad2deg(theta):.1f}, target=({tx:.2f},{ty:.2f})"
        )
    else:
        # train/default: 与 runner._set_target_for_pass 保持一致
        theta = float(heading + np.random.uniform(-np.deg2rad(theta_half_deg_default), np.deg2rad(theta_half_deg_default)))
        r_min_raw = float(curr_cfg.get("pass_target_r_min", 1.3))
        r_max_raw = float(curr_cfg.get("pass_target_r_max", 1.6))
        r_min = max(r_min_raw, target_r_floor)
        r_max = max(r_max_raw, r_min)
        radius = float(np.random.uniform(r_min, r_max))
        tx = float(base_x + radius * np.cos(theta))
        ty = float(base_y + radius * np.sin(theta))
        info = (
            f"mode=train(base), base=({base_x:.2f},{base_y:.2f}), "
            f"R={radius:.2f} in [{r_min:.2f},{r_max:.2f}], "
            f"heading={heading:.2f}, theta_deg={np.rad2deg(theta):.1f} (half={theta_half_deg_default:.1f}), target=({tx:.2f},{ty:.2f})"
        )

    env.target_xy = torch.tensor([tx, ty], device=device, dtype=env.base_pos.dtype)
    return tx, ty, info


def _set_infer_ball_spawn(env: PassBallEnv, args):
    """
    可选覆盖 infer 的初始球位置：
    - 默认不覆盖（沿用 env.reset() 内 reset_pass_ball 的逻辑）
    - 传 --ball_r 时，按极坐标在给定半径/角度扇区内重设球位置
    - 否则若传 --ball_dist 时，按机器人基座前方距离重设球位置
    """
    base_pos = env.base_pos[0, :3]
    base_x = float(base_pos[0].item())
    base_y = float(base_pos[1].item())
    z = float(env.ball_world.default_z) if args.ball_z is None else float(args.ball_z)

    heading = _get_robot_heading(env)
    stage_cfg = env.controller.cfg.get("curriculum", {}).get("reward_stages", {})

    if args.ball_r is not None:
        radius = float(args.ball_r)
        theta_min_deg = float(stage_cfg.get("ball_spawn_theta_min_deg", -20.0)) if args.ball_theta_min_deg is None else float(args.ball_theta_min_deg)
        theta_max_deg = float(stage_cfg.get("ball_spawn_theta_max_deg", 20.0)) if args.ball_theta_max_deg is None else float(args.ball_theta_max_deg)
        if theta_max_deg < theta_min_deg:
            theta_min_deg, theta_max_deg = theta_max_deg, theta_min_deg

        theta_deg = float(np.random.uniform(theta_min_deg, theta_max_deg))
        theta = heading + np.deg2rad(theta_deg)
        x = base_x + radius * np.cos(theta)
        y = base_y + radius * np.sin(theta)
        env.ball_world.set_pose(env.root_states, (x, y, z), zero_velocity=True)

        return (
            f"override=True, mode=polar, base=({base_x:.2f},{base_y:.2f}), "
            f"ball=({x:.2f},{y:.2f},{z:.2f}), "
            f"r={radius:.2f}, heading={heading:.2f}, theta_deg={theta_deg:.1f}, "
            f"theta_range_deg=[{theta_min_deg:.1f}, {theta_max_deg:.1f}]"
        )

    if args.ball_dist is None:
        return None

    dist = float(args.ball_dist)
    lateral = 0.0 if args.ball_lateral is None else float(args.ball_lateral)

    x = base_x + dist * np.cos(heading) - lateral * np.sin(heading)
    y = base_y + dist * np.sin(heading) + lateral * np.cos(heading)
    env.ball_world.set_pose(env.root_states, (x, y, z), zero_velocity=True)

    planar_dist = float(np.sqrt((x - base_x) ** 2 + (y - base_y) ** 2))
    return (
        f"override=True, base=({base_x:.2f},{base_y:.2f}), "
        f"ball=({x:.2f},{y:.2f},{z:.2f}), "
        f"heading={heading:.2f}, front_dist={dist:.2f}, lateral={lateral:.2f}, planar_dist={planar_dist:.2f}"
    )


def _resolve_train_like_dynamics(env: PassBallEnv, args):
    """
    让 infer 默认与 passBall 训练时高层控制节奏一致：
    - action_repeat: 训练中固定为 10
    - gait_freq: 训练中固定为 1.5
    - smooth: passBall 训练中固定为 0.5（若 curriculum 提供 pass_cmd_smooth，则优先）
    """
    curr_cfg = env.controller.cfg.get("curriculum", {})
    train_action_repeat = 10
    train_gait_freq = 1.5
    train_smooth = float(curr_cfg.get("pass_cmd_smooth", 0.5))

    action_repeat = int(train_action_repeat if args.action_repeat is None else args.action_repeat)
    gait_freq = float(train_gait_freq if args.gait_freq is None else args.gait_freq)
    smooth = float(train_smooth if args.smooth is None else args.smooth)
    return action_repeat, gait_freq, smooth, train_action_repeat, train_gait_freq, train_smooth


def main():
    parser = argparse.ArgumentParser(description="PassBall inference in Isaac Gym with video export")
    parser.add_argument("--task", type=str, default="PassBallEnv", help="Task class name (default: PassBallEnv)")
    parser.add_argument("--cfg", type=str, default="envs/passBall/PassBallEnv.yaml", help="Path to passBall yaml")
    parser.add_argument(
        "--low_ckpt",
        type=str,
        default=None,
        help="Low-level checkpoint: .pth (ActorCritic) or .pt (TorchScript, e.g. deploy/models/T1.pt). Use -1 for latest logs/**/*.pth",
    )
    parser.add_argument("--high_ckpt", type=str, required=True, help="High-level SAC checkpoint (.pt). Use -1 for latest non-empty")
    parser.add_argument("--sim_device", type=str, default=None, help="Override sim device, e.g. cuda:0")
    parser.add_argument("--rl_device", type=str, default=None, help="Override RL device, e.g. cuda:0")
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument("--seconds", type=float, default=12.0, help="Inference wall-clock in sim-time seconds")
    parser.add_argument("--action_repeat", type=int, default=None, help="High-level action repeat. Default: follow train value (10)")
    parser.add_argument("--gait_freq", type=float, default=None, help="Fixed gait frequency. Default: follow train value (1.5)")
    parser.add_argument("--smooth", type=float, default=None, help="High-level command smoothing alpha. Default: follow train value (pass=0.5)")
    parser.add_argument("--out", type=str, default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="train",
        choices=["train", "random", "fixed", "polar", "ball"],
        help="Target generation mode for infer video",
    )
    parser.add_argument("--target_x", type=float, default=None, help="Used when target_mode=fixed")
    parser.add_argument("--target_y", type=float, default=None, help="Used when target_mode=fixed")
    parser.add_argument("--target_r", type=float, default=1.45, help="Used when target_mode=polar")
    parser.add_argument("--target_theta_deg", type=float, default=0.0, help="Used when target_mode=polar")
    parser.add_argument("--target_r_min", type=float, default=None, help="random mode radius min override")
    parser.add_argument("--target_r_max", type=float, default=None, help="random mode radius max override")
    parser.add_argument(
        "--target_r_floor",
        type=float,
        default=None,
        help="Minimum target radius floor (meters) for train/random target sampling. Default: curriculum.pass_target_r_floor",
    )
    parser.add_argument("--target_theta_min_deg", type=float, default=None, help="random mode theta min (deg) override")
    parser.add_argument("--target_theta_max_deg", type=float, default=None, help="random mode theta max (deg) override")
    parser.add_argument("--ball_r", type=float, default=None, help="Override initial ball radius (meters) from robot base")
    parser.add_argument("--ball_theta_min_deg", type=float, default=None, help="Ball spawn theta min (deg) when using --ball_r")
    parser.add_argument("--ball_theta_max_deg", type=float, default=None, help="Ball spawn theta max (deg) when using --ball_r")
    parser.add_argument("--ball_dist", type=float, default=None, help="Override initial ball front distance (meters) from robot base")
    parser.add_argument("--ball_lateral", type=float, default=0.0, help="Initial ball lateral offset (meters) when using --ball_dist")
    parser.add_argument("--ball_z", type=float, default=None, help="Initial ball height override when using --ball_dist")
    parser.add_argument(
        "--stop_on_event",
        action="store_true",
        help="If set, stop rollout early when success/fall/done appears. Default: run full seconds.",
    )
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args.sim_device is not None:
        cfg["basic"]["sim_device"] = args.sim_device
    if args.rl_device is not None:
        cfg["basic"]["rl_device"] = args.rl_device
    has_display = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
    force_headless = bool(args.headless) or (not has_display)
    if (not bool(args.headless)) and (not has_display):
        print("[Info] DISPLAY is unavailable; forcing headless mode for offscreen video capture.")
    cfg["basic"]["headless"] = force_headless
    cfg["viewer"]["record_video"] = True

    low_ckpt = _resolve_low_ckpt(args.low_ckpt, cfg)
    if low_ckpt is not None:
        cfg["basic"]["checkpoint"] = low_ckpt

    device = cfg["basic"]["rl_device"]
    target_xy = torch.zeros(2, device=device, dtype=torch.float32)

    task_class = eval(args.task)
    env: PassBallEnv = task_class(cfg, target_xy)
    graphics_id = int(getattr(env.controller, "graphics_device_id", -1))
    if graphics_id < 0:
        raise RuntimeError(
            "graphics_device_id=-1: video capture is unavailable. "
            "Run infer without --headless or create sim with viewer.record_video=true before env init."
        )

    low_policy = _load_low_policy(low_ckpt, env, device)

    high_ckpt = _resolve_high_ckpt(args.high_ckpt)
    print(f"[Load] high-level ckpt: {high_ckpt}")
    high_agent = _load_high_agent(high_ckpt, env, device)

    obs, infos = env.reset()
    obs = obs.to(device)

    (
        action_repeat,
        gait_freq,
        smooth,
        train_action_repeat,
        train_gait_freq,
        train_smooth,
    ) = _resolve_train_like_dynamics(env, args)

    ball_info = _set_infer_ball_spawn(env, args)
    tx, ty, target_info = _set_infer_target(env, device, args)

    low_dt = float(env.dt)  # 每次 env.step 的仿真秒
    high_dt = low_dt * int(action_repeat)
    n_high_steps = max(1, int(np.ceil(float(args.seconds) / max(1e-6, high_dt))))

    print(
        f"[Run] seconds={args.seconds:.2f}, low_dt={low_dt:.4f}, action_repeat={action_repeat}, "
        f"high_steps={n_high_steps}, target=({tx:.2f},{ty:.2f})"
    )
    print(
        f"[Dynamics] infer(action_repeat={action_repeat}, gait_freq={gait_freq:.2f}, smooth={smooth:.2f}) | "
        f"train_default(action_repeat={train_action_repeat}, gait_freq={train_gait_freq:.2f}, smooth={train_smooth:.2f})"
    )
    if ball_info is not None:
        print(f"[BallSpawn] {ball_info}")
    print(f"[Target] {target_info}")

    success = False
    fall = False
    with torch.no_grad():
        for _ in range(n_high_steps):
            obs_high = env.compute_midlevel_obs().to(device)
            action = high_agent.select_action(obs_high.squeeze(0).detach().cpu().numpy(), eval_mode=True)
            cmd = [float(action[0]), float(action[1]), float(action[2]), float(gait_freq)]
            env.apply_high_level_command(cmd, smooth=float(smooth))

            for _ in range(int(action_repeat)):
                obs_mod = obs.clone()
                obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
                act = _low_policy_act(low_policy, obs_mod)
                obs, rew, done, infos = env.step(act)
                obs = obs.to(device)

                if isinstance(infos, dict):
                    # infer 只按最终传球成功统计
                    if bool(infos.get("final_success", False)):
                        success = True
                    if bool(infos.get("fall", False)):
                        fall = True
                should_stop = success or fall or bool(torch.any(done).item())
                if bool(args.stop_on_event) and should_stop:
                    break
            if bool(args.stop_on_event) and (success or fall):
                break

    if args.out is None:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        args.out = os.path.join("videos", f"passball_infer_{ts}.mp4")

    frame_count = _save_video_from_camera_frames(env, args.out, fps=args.fps)

    print(f"[Done] success={success}, fall={fall}, frames={frame_count}")
    if frame_count == 0:
        print("[WARN] No frames were captured. Check graphics availability and viewer.record_video settings.")
    else:
        print(f"[Video] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
