#!/usr/bin/env python3
"""
Compact PoC: IsaacGym RGB capture -> (placeholder) vision policy -> high-level command -> low-level rollout.

Proves the pipeline for feeding simulation images into a high-level policy. Replace ``vision_high_level_command``
with a VLM or CNN when ready.

Usage (from repo root, with IsaacGym available):
  python scripts/isaac_vision_policy_poc.py --task passBall --task-class PassBallEnv --episodes 3

Requires in YAML (or overridden below):
  basic.headless: true
  viewer.capture_for_policy: true
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym
from envs.hyperGym.isaac_wrapper import isaac_booster_multi_tensors_to_hyper_state
from scripts.vlm_policy_poc import (
    DEFAULT_VLM_BASE_URL,
    DEFAULT_VLM_MODEL,
    OpenAICompatibleVisionVLM,
    _parse_team_decision,
    _get_vlm_api_key,
    _state_text_summary,
    _to_plain,
)
import torch


def vision_high_level_command(rgb: np.ndarray, mode: str) -> tuple[float, float, float]:
    """
    Placeholder vision -> [vx, vy, yaw]. Swap for VLM / encoder output.

    Modes:
      trivial: small forward velocity
      pixel_hash: deterministic pseudo-random command from image mean (sanity check that pixels change)
    """
    del rgb  # unused in trivial
    if mode == "trivial":
        return 0.2, 0.0, 0.0
    if mode == "pixel_hash":
        m = float(np.mean(rgb)) if rgb.size else 0.0
        vx = 0.1 + 0.15 * (m % 32) / 32.0
        vy = 0.05 * np.sin(m)
        yaw = 0.3 * np.cos(m * 0.1)
        return float(vx), float(vy), float(yaw)
    raise ValueError(f"Unknown vision mode: {mode}")


def build_vlm(model: str, base_url: str) -> OpenAICompatibleVisionVLM:
    return OpenAICompatibleVisionVLM(
        model=model,
        api_key=_get_vlm_api_key(),
        base_url=base_url,
        timeout_seconds=float(os.environ.get("ISAAC_VLM_TIMEOUT", "90")),
    )


def _estimate_ball_owner_id_for_booster(env, possession_radius: float = 0.6) -> str | None:
    ball_xy = env.root_states[env.ball_actor_index, 0:2]
    dists = torch.norm(env.base_pos[:, :2] - ball_xy.unsqueeze(0), dim=-1)
    nearest_idx = int(torch.argmin(dists).item())
    if float(dists[nearest_idx].item()) <= possession_radius:
        return env.player_names[nearest_idx]
    return None


def _booster_state_for_vlm(env, step_idx: int) -> dict:
    field_cfg = env.cfg.get("game", {}).get("field", {})
    field_size = (
        float(field_cfg.get("length", 14.0)),
        float(field_cfg.get("width", 9.0)),
    )
    world_bounds = (
        (-0.5 * field_size[0], 0.5 * field_size[0]),
        (-0.5 * field_size[1], 0.5 * field_size[1]),
    )
    return isaac_booster_multi_tensors_to_hyper_state(
        env.root_states,
        num_players=env.num_players,
        ball_actor_index=env.ball_actor_index,
        player_layout=env.controller.player_layout,
        field_size=field_size,
        world_bounds=world_bounds,
        step=step_idx,
        ball_owner_id=_estimate_ball_owner_id_for_booster(env),
    )


def _vlm_policy_actions_for_booster(env, rgb: np.ndarray, step_idx: int, args) -> dict:
    state = _booster_state_for_vlm(env, step_idx)
    state_text = _state_text_summary(state, recent_events=[])
    image = Image.fromarray(rgb)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        frame_path = Path(f.name)
    try:
        image.save(frame_path)
        raw = args._vlm.decide(
            state=state,
            state_text=state_text if args.vision_mode == "vlm_image_with_state_text" else "",
            image_path=frame_path,
            vision_view="global",
        )
        team_decision = _parse_team_decision(
            raw if isinstance(raw, dict) else {},
            state=state,
            fallback_reason="vlm_schema_invalid",
        )
        actions = {
            player_id: {
                "policy_id": decision.policy_id,
                "target": [float(decision.target[0]), float(decision.target[1])],
            }
            for player_id, decision in team_decision.players.items()
        }
        if getattr(args, "artifact_dir", ""):
            artifact_dir = Path(args.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = f"ep{args._current_episode:03d}_step{step_idx:03d}"
            image.save(artifact_dir / f"{stem}.png")
            import json

            with open(artifact_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
                json.dump(
                    _to_plain({
                        "state": state,
                        "state_text": state_text,
                        "raw_decision": raw,
                        "policy_actions": actions,
                    }),
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        return actions
    finally:
        frame_path.unlink(missing_ok=True)


def load_cfg(task_folder: str, task_class: str) -> dict:
    import yaml

    cfg_path = ROOT / "envs" / task_folder / f"{task_class}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg.setdefault("basic", {})
    cfg["basic"]["task"] = task_class
    cfg["basic"]["headless"] = True
    cfg.setdefault("viewer", {})
    cfg["viewer"]["capture_for_policy"] = True
    cfg["viewer"].setdefault("record_video", False)
    return cfg


def set_seed(cfg: dict) -> None:
    import random

    seed = int(cfg["basic"].get("seed", 0))
    if seed < 0:
        seed = int(np.random.randint(0, 10000))
        cfg["basic"]["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[PoC] seed={seed}")


def cfg_get_checkpoint(cfg) -> str | None:
    from glob import glob

    if cfg is None:
        return None
    ckpt = cfg.get("basic", {}).get("checkpoint")
    if ckpt in (None, "", False):
        return None
    if ckpt in ("-1", -1):
        hits = sorted(glob(str(ROOT / "logs" / "low" / "**" / "*.pth"), recursive=True), key=os.path.getmtime)
        return hits[-1] if hits else None
    return str(ckpt)


def run_pass_like_env(args, env, device: str) -> None:
    from utils.model import ActorCritic

    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    ckpt = cfg_get_checkpoint(env.controller.cfg)
    if ckpt:
        print(f"[PoC] Loading low-level weights: {ckpt}")
        blob = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(blob["model"], strict=False)
    else:
        print("[PoC] WARN: No low-level checkpoint.")

    ACTION_REPEAT = args.action_repeat
    max_steps = args.max_high_steps
    episodes = args.episodes
    gait_freq = 1.5

    def set_target():
        if args.task_folder == "passBall":
            # mirror runner._set_target_for_pass
            import numpy as np

            base_pos = env.base_pos[0, :3]
            base_x = float(base_pos[0].item())
            base_y = float(base_pos[1].item())
            THETA_MAX = np.deg2rad(30.0)
            theta = np.random.uniform(-THETA_MAX, THETA_MAX)
            dir_x = np.cos(theta)
            dir_y = np.sin(theta)
            r_min = float(getattr(env, "cur_r_min", 1.0))
            r_max = float(getattr(env, "cur_r_max", 1.5))
            if r_max < r_min:
                r_min, r_max = r_max, r_min
            r = np.random.uniform(r_min, r_max)
            tx = base_x + r * dir_x
            ty = base_y + r * dir_y
            env.target_xy = torch.tensor([tx, ty], dtype=env.base_pos.dtype, device=device)
        elif args.task_folder == "chaseBall":
            x, _, _ = env.ball_world.get_pose(env.root_states)
            env.target_xy = x[:2].to(device)
        # trapBall: target is internal

    total_frames = 0
    successes = 0

    for ep in range(episodes):
        obs, infos = env.reset()
        obs = obs.to(device)
        set_target()

        episode_step = 0
        success_happened = False
        fall_happened = False
        done_high = False

        while not done_high and episode_step < max_steps:
            env.controller.gym.refresh_actor_root_state_tensor(env.controller.sim)
            rgb = env.controller.capture_policy_frame(
                env.root_states,
                env_idx=0,
                follow_actor_index=0,
                width=args.cam_width,
                height=args.cam_height,
            )
            total_frames += 1
            if ep == 0 and episode_step == 0 and args.save_first_frame:
                out = Path(args.save_first_frame)
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image

                    Image.fromarray(rgb).save(out)
                    print(f"[PoC] Saved first frame to {out}")
                except Exception as exc:
                    print(f"[PoC] Could not save frame: {exc}")

            vx, vy, yaw = vision_high_level_command(rgb, args.vision_mode)
            env.apply_high_level_command([vx, vy, yaw, gait_freq], smooth=0.5)

            for _ in range(ACTION_REPEAT):
                with torch.no_grad():
                    obs_mod = obs.clone()
                    obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = vx, vy, yaw
                    dist = model.act(obs_mod)
                    act = dist.loc
                    obs, rew, done, infos = env.step(act)
                    obs = obs.to(device)
                if isinstance(infos, dict):
                    if infos.get("fall", False):
                        fall_happened = True
                        break
                    if infos.get("success", False):
                        success_happened = True
                        break
                if torch.any(done).item():
                    break

            episode_step += 1
            done_high = success_happened or fall_happened or (episode_step >= max_steps)

        succ = bool(success_happened and not fall_happened)
        successes += int(succ)
        print(
            f"[PoC] episode={ep} high_steps={episode_step} success={succ} "
            f"fall={fall_happened} frames_captured={total_frames}"
        )

    rate = successes / max(1, episodes)
    print(f"\n=== Isaac vision PoC ===\nepisodes={episodes} successes={successes} rate={rate:.4f}\nframes={total_frames}")


def run_booster_env(args, env, device: str) -> None:
    from utils.model import ActorCritic

    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    ckpt = cfg_get_checkpoint(env.cfg)
    if ckpt:
        print(f"[PoC] Loading low-level weights: {ckpt}")
        blob = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(blob["model"], strict=False)
    else:
        print("[PoC] WARN: No low-level checkpoint.")

    ACTION_REPEAT = args.action_repeat
    max_steps = args.max_high_steps
    episodes = args.episodes
    gait = 0.5 * (env.cfg["commands"]["gait_frequency"][0] + env.cfg["commands"]["gait_frequency"][1])

    total_frames = 0
    for ep in range(episodes):
        args._current_episode = ep
        env.reset()
        episode_step = 0
        done_high = False
        while not done_high and episode_step < max_steps:
            env.controller.gym.refresh_actor_root_state_tensor(env.controller.sim)
            rgb = env.controller.capture_policy_frame(
                env.root_states,
                env_idx=0,
                follow_actor_index=0,
                width=args.cam_width,
                height=args.cam_height,
            )
            total_frames += 1
            if ep == 0 and episode_step == 0 and args.save_first_frame:
                out = Path(args.save_first_frame)
                out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image

                    Image.fromarray(rgb).save(out)
                    print(f"[PoC] Saved first frame to {out}")
                except Exception as exc:
                    print(f"[PoC] Could not save frame: {exc}")

            if args.vision_mode in {"vlm_image_only", "vlm_image_with_state_text"}:
                policy_actions = _vlm_policy_actions_for_booster(env, rgb, episode_step, args)
                env.apply_policy_command(policy_actions, smooth=0.5, eval_mode=True)
            else:
                vx, vy, yaw = vision_high_level_command(rgb, args.vision_mode)
                cmd = {name: [vx, vy, yaw, float(gait)] for name in env.player_names}
                env.apply_high_level_command(cmd, smooth=0.5)

            for _ in range(ACTION_REPEAT):
                with torch.no_grad():
                    actions = torch.zeros(env.num_players, env.num_actions, device=device)
                    for p in range(env.num_players):
                        obs_p = env.obs_buf[p : p + 1]
                        obs_mod = obs_p.clone()
                        obs_mod[0, 6] = env.commands[p, 0]
                        obs_mod[0, 7] = env.commands[p, 1]
                        obs_mod[0, 8] = env.commands[p, 2]
                        dist = model.act(obs_mod)
                        actions[p] = dist.loc.squeeze(0)
                    env.step(actions)

            episode_step += 1
            if bool(env.reset_buf.item()):
                done_high = True

        print(f"[PoC] booster episode={ep} high_steps={episode_step} frames={total_frames}")

    print(f"\n=== Isaac vision PoC (BoosterT12v2) ===\nepisodes={episodes}\nframes={total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IsaacGym RGB -> high-level policy PoC")
    parser.add_argument("--task-folder", type=str, default="passBall", help="Subfolder under envs/ (yaml name)")
    parser.add_argument(
        "--task-class",
        type=str,
        default="PassBallEnv",
        help="Python class name: PassBallEnv, ChaseBallEnv, TrapBallEnv, BoosterT12v2Env",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-high-steps", type=int, default=50, help="High-level decision steps per episode")
    parser.add_argument("--action-repeat", type=int, default=10, help="Low-level steps per high-level command")
    parser.add_argument(
        "--vision-mode",
        type=str,
        default="trivial",
        choices=["trivial", "pixel_hash", "vlm_image_only", "vlm_image_with_state_text"],
    )
    parser.add_argument("--cam-width", type=int, default=320)
    parser.add_argument("--cam-height", type=int, default=180)
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument(
        "--save-first-frame",
        type=str,
        default="",
        help="If set, save first captured RGB as PNG (e.g. logs/isaac_poc/frame0.png)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.task_folder, args.task_class)
    set_seed(cfg)
    device = cfg["basic"]["rl_device"]
    if args.vision_mode in {"vlm_image_only", "vlm_image_with_state_text"}:
        args._vlm = build_vlm(args.vlm_model, args.vlm_base_url)

    task_class = args.task_class
    if task_class == "BoosterT12v2Env":
        from envs.boosterT12v2.BoosterT12v2Env import BoosterT12v2Env

        env = BoosterT12v2Env(cfg, None)
        run_booster_env(args, env, device)
        return

    if args.vision_mode in {"vlm_image_only", "vlm_image_with_state_text"}:
        raise SystemExit("VLM mode is currently only supported for BoosterT12v2Env in this PoC.")

    dummy_target = torch.zeros(2, device=device, dtype=torch.float32)
    if task_class == "PassBallEnv":
        from envs.passBall.PassBallEnv import PassBallEnv

        env = PassBallEnv(cfg, dummy_target)
    elif task_class == "ChaseBallEnv":
        from envs.chaseBall.ChaseBallEnv import ChaseBallEnv

        env = ChaseBallEnv(cfg, dummy_target)
    elif task_class == "TrapBallEnv":
        from envs.trapBall.TrapBallEnv import TrapBallEnv

        env = TrapBallEnv(cfg, dummy_target)
    else:
        raise SystemExit(f"Unsupported task-class for this PoC: {task_class}")

    run_pass_like_env(args, env, device)


if __name__ == "__main__":
    main()
