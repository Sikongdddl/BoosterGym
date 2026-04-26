import isaacgym  # noqa: F401
import os
import glob
import time
import json
import csv
import random
import argparse
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from envs import PassBallEnv, TrapBallEnv, ChaseBallEnv
from utils.model import ActorCritic
from core.agents.sac.agent import SACAgent


TASK_SPECS: Dict[str, Dict[str, str]] = {
    "pass": {
        "task_class": "PassBallEnv",
        "cfg_path": "envs/passBall/PassBallEnv.yaml",
        "ckpt_task_dir": "passBall",
    },
    "trap": {
        "task_class": "TrapBallEnv",
        "cfg_path": "envs/trapBall/TrapBallEnv.yaml",
        "ckpt_task_dir": "trapBall",
    },
    "chase": {
        "task_class": "ChaseBallEnv",
        "cfg_path": "envs/chaseBall/ChaseBallEnv.yaml",
        "ckpt_task_dir": "chaseBall",
    },
}


def _parse_int_list(s: str) -> List[int]:
    vals = [x.strip() for x in str(s).split(",")]
    out = [int(x) for x in vals if x != ""]
    if not out:
        raise ValueError(f"Invalid int list: {s}")
    return out


def _parse_float_list(s: str) -> List[float]:
    vals = [x.strip() for x in str(s).split(",")]
    out = [float(x) for x in vals if x != ""]
    if not out:
        raise ValueError(f"Invalid float list: {s}")
    return out


def _parse_task_list(s: str) -> List[str]:
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip() != ""]
    if not vals:
        raise ValueError(f"Invalid task list: {s}")
    unknown = [x for x in vals if x not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid: {list(TASK_SPECS.keys())}")
    return vals


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_low_ckpt(low_ckpt_arg: Optional[str], cfg: dict) -> Optional[str]:
    ckpt = low_ckpt_arg if low_ckpt_arg is not None else cfg.get("basic", {}).get("checkpoint", None)
    if ckpt in (None, "", "null"):
        # Common fallback in this repo.
        fallback = os.path.join("deploy", "models", "T1.pt")
        if os.path.isfile(fallback):
            return fallback
        return None

    if ckpt in ("-1", -1):
        candidates = sorted(
            glob.glob(os.path.join("logs", "low", "**", "*.pth"), recursive=True),
            key=os.path.getmtime,
        )
        if not candidates:
            candidates = sorted(
                glob.glob(os.path.join("logs", "**", "*.pth"), recursive=True),
                key=os.path.getmtime,
            )
        if not candidates:
            fallback = os.path.join("deploy", "models", "T1.pt")
            if os.path.isfile(fallback):
                return fallback
            return None
        return candidates[-1]
    return str(ckpt)


def _resolve_high_ckpt(task_key: str, high_ckpt_arg: str) -> str:
    if high_ckpt_arg not in ("-1", -1):
        if not os.path.isfile(high_ckpt_arg):
            raise FileNotFoundError(f"[{task_key}] high-level checkpoint not found: {high_ckpt_arg}")
        return str(high_ckpt_arg)

    task_dir = TASK_SPECS[task_key]["ckpt_task_dir"]
    candidates: List[str] = []

    # Task-specific layouts used by pass/trap.
    candidates.extend(glob.glob(os.path.join("logs", "ckpt", task_dir, "sac", "sac_agent_step_*.pt")))
    candidates.extend(
        glob.glob(os.path.join("logs", "ckpt", task_dir, "sac", "**", "sac_agent_step_*.pt"), recursive=True)
    )

    # Legacy chaseBall layout in this codebase.
    if task_key == "chase":
        candidates.extend(glob.glob(os.path.join("logs", "ckpt", "sac_agent_step_*.pt")))
        candidates.extend(glob.glob(os.path.join("logs", "ckpt", "chaseBall", "sac_agent_step_*.pt")))

    candidates = [p for p in set(candidates) if os.path.isfile(p) and os.path.getsize(p) > 0]
    candidates = sorted(candidates, key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"[{task_key}] no non-empty SAC checkpoints found. "
            f"Tried logs/ckpt/{task_dir}/sac/**/sac_agent_step_*.pt"
            + (" and logs/ckpt/sac_agent_step_*.pt" if task_key == "chase" else "")
        )
    return candidates[-1]


def _optimizer_to_device(optimizer, device: str) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _move_loaded_sac_agent_to_device(agent: SACAgent, device: str) -> SACAgent:
    agent.device = device
    for module_name in ("policy", "q1", "q2", "q1_target", "q2_target"):
        module = getattr(agent, module_name, None)
        if module is not None:
            module.to(device)
            module.eval()
    if hasattr(agent, "action_low") and torch.is_tensor(agent.action_low):
        agent.action_low = agent.action_low.to(device)
    if hasattr(agent, "action_high") and torch.is_tensor(agent.action_high):
        agent.action_high = agent.action_high.to(device)
    if hasattr(agent, "log_alpha") and torch.is_tensor(agent.log_alpha):
        agent.log_alpha = agent.log_alpha.to(device)
        agent.log_alpha.requires_grad_(True)

    _optimizer_to_device(getattr(agent, "pi_optim", None), device)
    _optimizer_to_device(getattr(agent, "q1_optim", None), device)
    _optimizer_to_device(getattr(agent, "q2_optim", None), device)
    _optimizer_to_device(getattr(agent, "alpha_optim", None), device)
    return agent


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


def _build_sac_agent_for_env(env, task_key: str, device: str) -> SACAgent:
    obs_high = env.compute_midlevel_obs().to(device)
    state_dim = int(obs_high.shape[1])
    action_dim = 3

    curr_cfg = env.controller.cfg.get("curriculum", {})
    if task_key == "trap":
        vx_min = float(curr_cfg.get("trap_vx_min", -0.25))
        vx_max = float(curr_cfg.get("trap_vx_max", 0.60))
        if vx_max < vx_min:
            vx_min, vx_max = vx_max, vx_min
        vx_range = np.array([vx_min, vx_max], dtype=np.float32)
    else:
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


def _load_high_agent(high_ckpt: str, env, task_key: str, device: str) -> SACAgent:
    payload = torch.load(high_ckpt, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "agent" in payload:
        loaded_agent = payload["agent"]
        if not isinstance(loaded_agent, SACAgent):
            raise TypeError(
                f"Unsupported checkpoint format in {high_ckpt}: payload['agent'] is {type(loaded_agent)}"
            )
        agent = _move_loaded_sac_agent_to_device(loaded_agent, device)
    elif isinstance(payload, SACAgent):
        agent = _move_loaded_sac_agent_to_device(payload, device)
    elif isinstance(payload, dict) and "policy" in payload:
        agent = _build_sac_agent_for_env(env, task_key, device)
        agent.load(high_ckpt)
        agent = _move_loaded_sac_agent_to_device(agent, device)
    else:
        raise ValueError(
            f"Unsupported high-level checkpoint format: {high_ckpt}. "
            "Expected SACAgent, {'agent': SACAgent, ...}, or SACAgent.save() weights."
        )

    exp_s = int(env.compute_midlevel_obs().shape[1])
    exp_a = 3
    got_s, got_a = _sac_io_dims(agent)
    if (got_s is not None and got_s != exp_s) or (got_a is not None and got_a != exp_a):
        raise ValueError(
            f"High checkpoint shape mismatch for task={task_key}: "
            f"expected (state_dim={exp_s}, action_dim={exp_a}), got (state_dim={got_s}, action_dim={got_a}) "
            f"from {high_ckpt}."
        )
    return agent


def _load_low_policy(low_ckpt: Optional[str], env, device: str) -> Dict:
    clip_actions = float(env.controller.cfg["normalization"]["clip_actions"])

    if low_ckpt is None:
        print("[WARN] No low-level checkpoint provided/found. Falling back to random-init ActorCritic.")
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        model.eval()
        return {"kind": "actor_critic", "module": model, "clip_actions": clip_actions, "source": "random_init"}

    low_ckpt = str(low_ckpt)
    if low_ckpt.lower().endswith(".pth"):
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        model.eval()
        low_obj = torch.load(low_ckpt, map_location=device, weights_only=True)
        if not (isinstance(low_obj, dict) and "model" in low_obj):
            raise ValueError(f"Invalid low-level .pth format (expect dict['model']): {low_ckpt}")
        model.load_state_dict(low_obj["model"], strict=False)
        return {"kind": "actor_critic", "module": model, "clip_actions": clip_actions, "source": low_ckpt}

    if low_ckpt.lower().endswith(".pt"):
        # Prefer TorchScript low-level.
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
            # Fallback: some .pt files are plain checkpoints.
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
                    f"Invalid low-level checkpoint: {low_ckpt}\n"
                    "This file is a high-level SAC checkpoint (contains key 'agent')."
                ) from jit_err
            raise ValueError(f"Unsupported low-level .pt format: {low_ckpt}") from jit_err

    raise ValueError(f"Unsupported low-level checkpoint suffix (need .pth or .pt): {low_ckpt}")


def _low_policy_act(low_policy: Dict, obs_mod: torch.Tensor) -> torch.Tensor:
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


def _task_constructor(task_key: str):
    if task_key == "pass":
        return PassBallEnv
    if task_key == "trap":
        return TrapBallEnv
    if task_key == "chase":
        return ChaseBallEnv
    raise ValueError(f"Unknown task key: {task_key}")


def _to_float_reward(rew) -> float:
    if torch.is_tensor(rew):
        return float(rew.reshape(-1)[0].detach().cpu().item())
    return float(rew)


def _should_stop_by_events(task_key: str, success: bool, final_success: bool, fail: bool, fall: bool) -> bool:
    if task_key == "pass":
        # Follow passBall training behavior: stage success already ends high-level rollout.
        return success or fall
    if task_key == "trap":
        return success or fail or fall
    if task_key == "chase":
        return success or fall
    return success or fail or fall


@torch.no_grad()
def _eval_one_episode(
    task_key: str,
    env,
    high_agent: SACAgent,
    low_policy: Dict,
    device: str,
    action_repeat: int,
    smooth: float,
    seconds: float,
    gait_freq: float,
    stop_on_event: bool,
) -> Dict:
    obs, infos = env.reset()
    obs = obs.to(device)

    low_dt = float(env.dt)
    high_dt = low_dt * int(action_repeat)
    n_high_steps = max(1, int(np.ceil(float(seconds) / max(1e-6, high_dt))))

    ep_return = 0.0
    low_steps = 0
    high_steps = 0
    success = False
    final_success = False
    fail = False
    fall = False
    hit = False

    for _ in range(n_high_steps):
        obs_high = env.compute_midlevel_obs()
        if torch.is_tensor(obs_high) and obs_high.device != torch.device(device):
            obs_high = obs_high.to(device)
        action = high_agent.select_action(obs_high.squeeze(0).detach().cpu().numpy(), eval_mode=True)
        cmd = [float(action[0]), float(action[1]), float(action[2]), float(gait_freq)]
        env.apply_high_level_command(cmd, smooth=float(smooth))

        done_any = False
        for _ in range(int(action_repeat)):
            obs_mod = obs.clone()
            # In this repo, command slots are obs[:, 6:9].
            obs_mod[:, 6], obs_mod[:, 7], obs_mod[:, 8] = cmd[0], cmd[1], cmd[2]
            act = _low_policy_act(low_policy, obs_mod)
            obs, rew, done, infos = env.step(act)
            if torch.is_tensor(obs) and obs.device != torch.device(device):
                obs = obs.to(device)

            ep_return += _to_float_reward(rew)
            low_steps += 1
            done_any = bool(torch.any(done).item())

            if isinstance(infos, dict):
                success = success or bool(infos.get("success", False))
                final_success = final_success or bool(infos.get("final_success", False))
                fail = fail or bool(infos.get("fail", False))
                fall = fall or bool(infos.get("fall", False))
                hit = hit or bool(infos.get("hit", False))

            if done_any:
                break
            if stop_on_event and _should_stop_by_events(task_key, success, final_success, fail, fall):
                break

        high_steps += 1
        if done_any:
            break
        if stop_on_event and _should_stop_by_events(task_key, success, final_success, fail, fall):
            break

    primary_success = final_success if task_key == "pass" else success
    return {
        "primary_success": float(primary_success),
        "success": float(success),
        "final_success": float(final_success),
        "fail": float(fail),
        "fall": float(fall),
        "hit": float(hit),
        "return": float(ep_return),
        "low_steps": float(low_steps),
        "high_steps": float(high_steps),
        "episode_seconds": float(low_steps * low_dt),
    }


def _aggregate_setting(episodes: List[Dict]) -> Dict:
    if not episodes:
        raise ValueError("No episodes to aggregate.")

    def c(k: str) -> int:
        return int(np.sum([int(float(ep[k]) > 0.5) for ep in episodes]))

    def m(k: str) -> float:
        return float(np.mean([float(ep[k]) for ep in episodes]))

    def s(k: str) -> float:
        return float(np.std([float(ep[k]) for ep in episodes]))

    n_eps = int(len(episodes))
    primary_success_count = c("primary_success")
    success_count = c("success")
    final_success_count = c("final_success")
    fail_count = c("fail")
    fall_count = c("fall")
    hit_count = c("hit")

    return {
        "episodes": n_eps,
        "primary_success_count": primary_success_count,
        "primary_success_rate": m("primary_success"),
        "success_count": success_count,
        "success_rate": m("success"),
        "final_success_count": final_success_count,
        "final_success_rate": m("final_success"),
        "fail_count": fail_count,
        "fail_rate": m("fail"),
        "fall_count": fall_count,
        "fall_rate": m("fall"),
        "hit_count": hit_count,
        "hit_rate": m("hit"),
        "avg_return": m("return"),
        "std_return": s("return"),
        "avg_low_steps": m("low_steps"),
        "avg_high_steps": m("high_steps"),
        "avg_episode_seconds": m("episode_seconds"),
    }


def _evaluate_task_grid(
    task_key: str,
    task_cfg_path: str,
    high_ckpt_arg: str,
    low_ckpt_arg: Optional[str],
    repeats: List[int],
    smooths: List[float],
    episodes: int,
    seconds: float,
    sim_device: Optional[str],
    rl_device: Optional[str],
    seed: int,
    stop_on_event: bool,
    gait_freq_override: Optional[float],
    verbose: bool,
) -> List[Dict]:
    with open(task_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg.setdefault("basic", {})
    cfg.setdefault("viewer", {})
    cfg["basic"]["headless"] = True
    cfg["viewer"]["record_video"] = False
    if sim_device is not None:
        cfg["basic"]["sim_device"] = sim_device
    if rl_device is not None:
        cfg["basic"]["rl_device"] = rl_device
    cfg["basic"]["seed"] = int(seed)

    low_ckpt = _resolve_low_ckpt(low_ckpt_arg, cfg)
    high_ckpt = _resolve_high_ckpt(task_key, high_ckpt_arg)
    device = str(cfg["basic"]["rl_device"])

    if verbose:
        print(
            f"[Setup][{task_key}] cfg={task_cfg_path} | device={device} | "
            f"low_ckpt={low_ckpt} | high_ckpt={high_ckpt}"
        )

    _set_global_seed(int(seed))

    env_ctor = _task_constructor(task_key)
    target_xy = torch.zeros(2, device=device, dtype=torch.float32)
    env = env_ctor(cfg, target_xy)
    low_policy = _load_low_policy(low_ckpt, env, device)
    high_agent = _load_high_agent(high_ckpt, env, task_key, device)

    # Default gait from env action space.
    default_gait = float(env.get_high_level_action_space()[3])
    gait_freq = float(default_gait if gait_freq_override is None else gait_freq_override)

    rows: List[Dict] = []
    for rep in repeats:
        for sm in smooths:
            if verbose:
                print(f"[Eval][{task_key}] repeat={rep}, smooth={sm:.3f}, episodes={episodes}")
            ep_stats = []
            for ep_idx in range(episodes):
                ep_out = _eval_one_episode(
                    task_key=task_key,
                    env=env,
                    high_agent=high_agent,
                    low_policy=low_policy,
                    device=device,
                    action_repeat=int(rep),
                    smooth=float(sm),
                    seconds=float(seconds),
                    gait_freq=float(gait_freq),
                    stop_on_event=bool(stop_on_event),
                )
                ep_out["episode_idx"] = int(ep_idx)
                ep_stats.append(ep_out)

            agg = _aggregate_setting(ep_stats)
            row = {
                "task": task_key,
                "cfg_path": task_cfg_path,
                "high_ckpt": high_ckpt,
                "low_ckpt": str(low_policy.get("source", low_ckpt)),
                "low_policy_kind": str(low_policy.get("kind", "unknown")),
                "action_repeat": int(rep),
                "smooth": float(sm),
                "seconds": float(seconds),
                "stop_on_event": bool(stop_on_event),
                "gait_freq": float(gait_freq),
            }
            row.update(agg)
            rows.append(row)

    return rows


def _write_outputs(rows: List[Dict], out_csv: str, out_json: str, meta: Dict) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    # Stable column order.
    key_order = [
        "task",
        "action_repeat",
        "smooth",
        "episodes",
        "primary_success_count",
        "primary_success_rate",
        "success_count",
        "success_rate",
        "final_success_count",
        "final_success_rate",
        "fail_count",
        "fail_rate",
        "fall_count",
        "fall_rate",
        "hit_count",
        "hit_rate",
        "avg_return",
        "std_return",
        "avg_low_steps",
        "avg_high_steps",
        "avg_episode_seconds",
        "seconds",
        "stop_on_event",
        "gait_freq",
        "low_policy_kind",
        "low_ckpt",
        "high_ckpt",
        "cfg_path",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = key_order + [k for k in sorted(all_keys) if k not in key_order]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    payload = {"meta": meta, "rows": rows}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_task_overall_summary(rows: List[Dict]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for r in rows:
        task = str(r["task"])
        if task not in summary:
            summary[task] = {
                "task": task,
                "episodes": 0,
                "primary_success_count": 0,
                "success_count": 0,
                "final_success_count": 0,
                "fail_count": 0,
                "fall_count": 0,
                "hit_count": 0,
            }
        summary[task]["episodes"] += int(r.get("episodes", 0))
        summary[task]["primary_success_count"] += int(r.get("primary_success_count", 0))
        summary[task]["success_count"] += int(r.get("success_count", 0))
        summary[task]["final_success_count"] += int(r.get("final_success_count", 0))
        summary[task]["fail_count"] += int(r.get("fail_count", 0))
        summary[task]["fall_count"] += int(r.get("fall_count", 0))
        summary[task]["hit_count"] += int(r.get("hit_count", 0))

    for task, s in summary.items():
        n = max(1, int(s["episodes"]))
        s["primary_success_rate"] = float(s["primary_success_count"] / n)
        s["success_rate"] = float(s["success_count"] / n)
        s["final_success_rate"] = float(s["final_success_count"] / n)
        s["fail_rate"] = float(s["fail_count"] / n)
        s["fall_rate"] = float(s["fall_count"] / n)
        s["hit_rate"] = float(s["hit_count"] / n)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate repeat/smooth sensitivity for pass/trap/chase tasks."
    )
    parser.add_argument("--tasks", type=str, default="pass,trap,chase", help="Comma list: pass,trap,chase")
    parser.add_argument("--repeats", type=str, default="5,10,20", help="Comma list of action_repeat values")
    parser.add_argument("--smooths", type=str, default="0.0,0.5,0.8", help="Comma list of smooth(alpha) values")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per (task,repeat,smooth) setting")
    parser.add_argument("--seconds", type=float, default=12.0, help="Max simulated seconds per episode")
    parser.add_argument("--stop_on_event", action="store_true", help="Stop an episode early on success/fail/fall")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

    parser.add_argument("--low_ckpt", type=str, default=None, help="Shared low-level ckpt (.pth/.pt). Use -1 for latest.")
    parser.add_argument("--pass_high_ckpt", type=str, default="-1", help="PassBall high-level SAC ckpt path or -1")
    parser.add_argument("--trap_high_ckpt", type=str, default="-1", help="TrapBall high-level SAC ckpt path or -1")
    parser.add_argument("--chase_high_ckpt", type=str, default="-1", help="ChaseBall high-level SAC ckpt path or -1")

    parser.add_argument("--pass_cfg", type=str, default=TASK_SPECS["pass"]["cfg_path"])
    parser.add_argument("--trap_cfg", type=str, default=TASK_SPECS["trap"]["cfg_path"])
    parser.add_argument("--chase_cfg", type=str, default=TASK_SPECS["chase"]["cfg_path"])
    parser.add_argument("--sim_device", type=str, default=None, help="Override sim device, e.g. cuda:0")
    parser.add_argument("--rl_device", type=str, default=None, help="Override rl device, e.g. cuda:0")
    parser.add_argument("--gait_freq", type=float, default=None, help="Override fixed gait frequency")
    parser.add_argument(
        "--allow_missing_tasks",
        action="store_true",
        help="Skip tasks whose ckpt auto-resolution fails instead of exiting.",
    )
    parser.add_argument("--out_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print setup/progress logs")
    parser.add_argument("--_child_mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    tasks = _parse_task_list(args.tasks)
    repeats = _parse_int_list(args.repeats)
    smooths = _parse_float_list(args.smooths)

    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_csv = args.out_csv or os.path.join("logs", "eval", f"repeat_smooth_eval_{ts}.csv")
    out_json = args.out_json or os.path.join("logs", "eval", f"repeat_smooth_eval_{ts}.json")

    # Isaac Gym/PhysX may only allow one Foundation/sim lifecycle per process.
    # For multi-task runs, fan out into child processes and merge their JSON rows.
    if len(tasks) > 1 and not args._child_mode:
        merged_rows: List[Dict] = []
        merged_skipped: List[Dict] = []

        print(f"[MultiTask] launching {len(tasks)} child processes (one task per process).")
        for task_key in tasks:
            child_csv = out_csv.replace(".csv", f".{task_key}.csv")
            child_json = out_json.replace(".json", f".{task_key}.json")

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--_child_mode",
                "--tasks",
                task_key,
                "--repeats",
                args.repeats,
                "--smooths",
                args.smooths,
                "--episodes",
                str(args.episodes),
                "--seconds",
                str(args.seconds),
                "--seed",
                str(args.seed),
                "--pass_high_ckpt",
                str(args.pass_high_ckpt),
                "--trap_high_ckpt",
                str(args.trap_high_ckpt),
                "--chase_high_ckpt",
                str(args.chase_high_ckpt),
                "--pass_cfg",
                str(args.pass_cfg),
                "--trap_cfg",
                str(args.trap_cfg),
                "--chase_cfg",
                str(args.chase_cfg),
                "--out_csv",
                child_csv,
                "--out_json",
                child_json,
            ]

            if args.low_ckpt is not None:
                cmd.extend(["--low_ckpt", str(args.low_ckpt)])
            if args.sim_device is not None:
                cmd.extend(["--sim_device", str(args.sim_device)])
            if args.rl_device is not None:
                cmd.extend(["--rl_device", str(args.rl_device)])
            if args.gait_freq is not None:
                cmd.extend(["--gait_freq", str(args.gait_freq)])
            if args.stop_on_event:
                cmd.append("--stop_on_event")
            if args.allow_missing_tasks:
                cmd.append("--allow_missing_tasks")
            if args.verbose:
                cmd.append("--verbose")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                if args.allow_missing_tasks:
                    merged_skipped.append({"task": task_key, "error": repr(e)})
                    print(f"[Skip][{task_key}] child process failed: {repr(e)}")
                    continue
                raise RuntimeError(
                    f"Child process failed for task={task_key}. Command: {' '.join(cmd)}"
                ) from e

            if not os.path.isfile(child_json):
                if args.allow_missing_tasks:
                    merged_skipped.append({"task": task_key, "error": "missing child json output"})
                    print(f"[Skip][{task_key}] child json missing: {child_json}")
                    continue
                raise FileNotFoundError(f"Expected child output not found: {child_json}")

            with open(child_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            merged_rows.extend(payload.get("rows", []))
            child_meta = payload.get("meta", {})
            child_skipped = child_meta.get("skipped", [])
            if isinstance(child_skipped, list):
                merged_skipped.extend(child_skipped)

        if not merged_rows:
            raise RuntimeError("No evaluation rows produced in multi-task mode.")

        task_summary = _build_task_overall_summary(merged_rows)
        meta = {
            "timestamp": ts,
            "tasks": tasks,
            "repeats": repeats,
            "smooths": smooths,
            "episodes": int(args.episodes),
            "seconds": float(args.seconds),
            "stop_on_event": bool(args.stop_on_event),
            "seed": int(args.seed),
            "sim_device": args.sim_device,
            "rl_device": args.rl_device,
            "gait_freq": args.gait_freq,
            "skipped": merged_skipped,
            "task_summary": task_summary,
            "mode": "multi_process_merged",
        }
        _write_outputs(merged_rows, out_csv=out_csv, out_json=out_json, meta=meta)
        print(f"[Done] rows={len(merged_rows)} | csv={os.path.abspath(out_csv)} | json={os.path.abspath(out_json)}")
        for task in tasks:
            if task in task_summary:
                s = task_summary[task]
                print(
                    f"[Summary][{task}] primary_success={s['primary_success_count']}/{s['episodes']} "
                    f"({s['primary_success_rate']:.4f})"
                )
        if merged_skipped:
            print(f"[Done] skipped_tasks={merged_skipped}")
        return

    high_ckpt_args = {
        "pass": args.pass_high_ckpt,
        "trap": args.trap_high_ckpt,
        "chase": args.chase_high_ckpt,
    }
    cfg_paths = {
        "pass": args.pass_cfg,
        "trap": args.trap_cfg,
        "chase": args.chase_cfg,
    }

    all_rows: List[Dict] = []
    skipped: List[Dict] = []

    for task_key in tasks:
        try:
            rows = _evaluate_task_grid(
                task_key=task_key,
                task_cfg_path=cfg_paths[task_key],
                high_ckpt_arg=high_ckpt_args[task_key],
                low_ckpt_arg=args.low_ckpt,
                repeats=repeats,
                smooths=smooths,
                episodes=int(args.episodes),
                seconds=float(args.seconds),
                sim_device=args.sim_device,
                rl_device=args.rl_device,
                seed=int(args.seed),
                stop_on_event=bool(args.stop_on_event),
                gait_freq_override=args.gait_freq,
                verbose=bool(args.verbose),
            )
            all_rows.extend(rows)
        except Exception as e:
            if args.allow_missing_tasks:
                msg = {"task": task_key, "error": repr(e)}
                skipped.append(msg)
                print(f"[Skip][{task_key}] {repr(e)}")
                continue
            raise

    if not all_rows:
        raise RuntimeError("No evaluation rows produced. All tasks may have failed.")

    task_summary = _build_task_overall_summary(all_rows)

    meta = {
        "timestamp": ts,
        "tasks": tasks,
        "repeats": repeats,
        "smooths": smooths,
        "episodes": int(args.episodes),
        "seconds": float(args.seconds),
        "stop_on_event": bool(args.stop_on_event),
        "seed": int(args.seed),
        "sim_device": args.sim_device,
        "rl_device": args.rl_device,
        "gait_freq": args.gait_freq,
        "skipped": skipped,
        "task_summary": task_summary,
    }
    _write_outputs(all_rows, out_csv=out_csv, out_json=out_json, meta=meta)

    print(f"[Done] rows={len(all_rows)} | csv={os.path.abspath(out_csv)} | json={os.path.abspath(out_json)}")
    for task in tasks:
        if task in task_summary:
            s = task_summary[task]
            print(
                f"[Summary][{task}] primary_success={s['primary_success_count']}/{s['episodes']} "
                f"({s['primary_success_rate']:.4f})"
            )
    if skipped:
        print(f"[Done] skipped_tasks={skipped}")


if __name__ == "__main__":
    main()
