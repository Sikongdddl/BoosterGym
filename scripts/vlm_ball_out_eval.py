from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.renderer import render_episode_mp4
from scripts.vlm_policy_poc import DEFAULT_VISION_VIEW
from scripts.vlm_policy_poc import DEFAULT_VLM_BASE_URL
from scripts.vlm_policy_poc import DEFAULT_VLM_MODEL
from scripts.vlm_policy_poc import OBSERVATION_MODES
from scripts.vlm_policy_poc import _build_vlm
from scripts.vlm_policy_poc import _get_benchmark_case
from scripts.vlm_policy_poc import _make_render_record
from scripts.vlm_policy_poc import _query_vlm_on_record
from scripts.vlm_policy_poc import _set_manual_state
from scripts.vlm_policy_poc import _state_text_summary
from scripts.vlm_policy_poc import _to_plain


def _parse_case_ids(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_ball_out_eval(
    episodes: int,
    max_steps: int,
    seed: int,
    vlm_model: str,
    vlm_base_url: str,
    artifact_dir: Optional[Path],
    num_home: int,
    num_away: int,
    start_case_id: str,
    observation_mode: str,
    vision_view: str,
    query_interval: int,
) -> Dict[str, Any]:
    start_case = _get_benchmark_case(start_case_id) if start_case_id else None
    rollout_num_home = start_case.num_home if start_case is not None else num_home
    rollout_num_away = start_case.num_away if start_case is not None else num_away
    env_max_steps = max_steps if start_case is None else int(start_case.state_spec.get("step", 0)) + max_steps
    vlm = _build_vlm(vlm_model, vlm_base_url)

    episode_summaries: List[Dict[str, Any]] = []
    out_count = 0
    home_win_count = 0
    away_win_count = 0
    timeout_count = 0

    for episode_idx in range(int(episodes)):
        controller = build_match_controller(
            num_home=rollout_num_home,
            num_away=rollout_num_away,
            seed=seed + episode_idx,
            max_steps=env_max_steps,
            wall_restitution=1.0,
            end_on_ball_out=True,
        )
        simulation = controller.simulation
        if start_case is None:
            simulation.reset()
            recent_events: List[Dict[str, Any]] = []
            reward = 0.0
            done = False
            action = None
            away_action = None
            info = {"events": [], "state": simulation.get_state(), "actions": {}}
        else:
            state = _set_manual_state(simulation, start_case.state_spec)
            recent_events = list(start_case.recent_events)
            reward = float(start_case.reward)
            done = bool(start_case.done)
            action = start_case.prior_action
            away_action = start_case.prior_away_action
            info = {"events": list(recent_events), "state": state, "actions": {}}

        used_fallback_steps = 0
        step_idx = 0
        ball_out = False
        ball_out_step = None
        ball_out_side = ""
        last_query = None

        while not done and step_idx < max_steps:
            should_query = last_query is None or step_idx % max(1, int(query_interval)) == 0
            if should_query:
                state = simulation.get_state()
                state_text = _state_text_summary(state, recent_events)
                record = _make_render_record(
                    state=state,
                    reward=reward,
                    done=done,
                    action=action,
                    away_action=away_action,
                    events=recent_events,
                )
                step_artifact_dir = None if artifact_dir is None else artifact_dir / f"episode_{episode_idx:03d}"
                query = _query_vlm_on_record(
                    vlm=vlm,
                    artifact_id=f"ep{episode_idx:03d}_step{step_idx:03d}",
                    record=record,
                    state_text=state_text,
                    artifact_dir=step_artifact_dir,
                    observation_mode=observation_mode,
                    vision_view=vision_view,
                )
                last_query = query
                team_decision = query["decision"]
                action = query["action"]
                if any(decision.source != "vlm" for decision in team_decision.players.values()):
                    used_fallback_steps += 1
            elif last_query is not None:
                action = last_query["action"]

            obs, reward, done, info = simulation.step(action)
            del obs
            away_action = info.get("actions", {}).get("away")
            recent_events = info.get("events", [])
            for event in recent_events:
                if event.get("event_type") == "ball_out_of_bounds":
                    ball_out = True
                    ball_out_step = step_idx + 1
                    ball_out_side = str(event.get("side", ""))
                    break
            step_idx += 1

        winner = info.get("state", {}).get("winner")
        if ball_out or winner == "ball_out":
            out_count += 1
        elif winner == "home":
            home_win_count += 1
        elif winner == "away":
            away_win_count += 1
        else:
            timeout_count += 1

        summary = {
            "episode": episode_idx,
            "seed": seed + episode_idx,
            "steps": step_idx,
            "winner": winner,
            "ball_out": ball_out or winner == "ball_out",
            "ball_out_step": ball_out_step,
            "ball_out_side": ball_out_side,
            "fallback_steps": used_fallback_steps,
        }
        episode_summaries.append(summary)
        print(
            f"[ball-out eval] episode={episode_idx} seed={seed + episode_idx} steps={step_idx} "
            f"winner={winner} ball_out={int(summary['ball_out'])} fallback_steps={used_fallback_steps}",
            flush=True,
        )

    result = {
        "episodes": int(episodes),
        "out_episodes": int(out_count),
        "out_rate": float(out_count) / max(1, int(episodes)),
        "home_win_episodes": int(home_win_count),
        "away_win_episodes": int(away_win_count),
        "timeout_episodes": int(timeout_count),
        "episode_summaries": episode_summaries,
        "start_case_id": start_case_id,
        "observation_mode": observation_mode,
        "vision_view": vision_view,
        "query_interval": int(query_interval),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
    }
    return result


def run_parallel_case_ball_out_eval(
    case_ids: List[str],
    max_steps: int,
    seed: int,
    vlm_model: str,
    vlm_base_url: str,
    artifact_dir: Optional[Path],
    observation_mode: str,
    vision_view: str,
    query_interval: int,
    video_dir: Optional[Path],
    video_fps: int,
) -> Dict[str, Any]:
    vlm = _build_vlm(vlm_model, vlm_base_url)
    cases = [_get_benchmark_case(case_id) for case_id in case_ids]
    workers = min(max(1, len(cases)), 3)
    runners: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases):
        controller = build_match_controller(
            num_home=case.num_home,
            num_away=case.num_away,
            seed=seed + idx,
            max_steps=max(int(case.state_spec.get("step", 0)) + max_steps, max_steps),
            wall_restitution=1.0,
            end_on_ball_out=True,
        )
        simulation = controller.simulation
        state = _set_manual_state(simulation, case.state_spec)
        recent_events = list(case.recent_events)
        reward = float(case.reward)
        done = bool(case.done)
        action = case.prior_action
        away_action = case.prior_away_action
        info = {"events": list(recent_events), "state": state, "actions": {}}
        initial_record = {
            "obs": simulation.get_obs(),
            "reward": reward,
            "done": done,
            "action": action,
            "away_action": away_action,
            "info": info,
        }
        runners.append(
            {
                "case": case,
                "case_id": case.case_id,
                "controller": controller,
                "simulation": simulation,
                "recent_events": recent_events,
                "reward": reward,
                "done": done,
                "action": action,
                "away_action": away_action,
                "info": info,
                "step_idx": 0,
                "used_fallback_steps": 0,
                "ball_out": False,
                "ball_out_step": None,
                "ball_out_side": "",
                "last_query": None,
                "episode_records": [initial_record],
            }
        )

    def _query_runner(runner: Dict[str, Any]):
        state = runner["simulation"].get_state()
        state_text = _state_text_summary(state, runner["recent_events"])
        record = _make_render_record(
            state=state,
            reward=runner["reward"],
            done=runner["done"],
            action=runner["action"],
            away_action=runner["away_action"],
            events=runner["recent_events"],
        )
        step_artifact_dir = None if artifact_dir is None else artifact_dir / runner["case_id"]
        return _query_vlm_on_record(
            vlm=vlm,
            artifact_id=f"{runner['case_id']}_step{runner['step_idx']:03d}",
            record=record,
            state_text=state_text,
            artifact_dir=step_artifact_dir,
            observation_mode=observation_mode,
            vision_view=vision_view,
        )

    while True:
        active = [runner for runner in runners if not runner["done"] and runner["step_idx"] < max_steps]
        if not active:
            break

        to_query = [
            runner
            for runner in active
            if runner["last_query"] is None or runner["step_idx"] % max(1, int(query_interval)) == 0
        ]
        if to_query:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                query_results = list(executor.map(_query_runner, to_query))
            for runner, query in zip(to_query, query_results):
                runner["last_query"] = query
                runner["action"] = query["action"]
                team_decision = query["decision"]
                if any(decision.source != "vlm" for decision in team_decision.players.values()):
                    runner["used_fallback_steps"] += 1

        for runner in active:
            if runner["last_query"] is not None:
                runner["action"] = runner["last_query"]["action"]
            obs, reward, done, info = runner["simulation"].step(runner["action"])
            runner["reward"] = reward
            runner["done"] = done
            runner["info"] = info
            runner["away_action"] = info.get("actions", {}).get("away")
            runner["recent_events"] = info.get("events", [])
            for event in runner["recent_events"]:
                if event.get("event_type") == "ball_out_of_bounds":
                    runner["ball_out"] = True
                    runner["ball_out_step"] = runner["step_idx"] + 1
                    runner["ball_out_side"] = str(event.get("side", ""))
                    break
            runner["episode_records"].append(
                {
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "action": info.get("actions", {}).get("home"),
                    "away_action": info.get("actions", {}).get("away"),
                    "info": info,
                }
            )
            runner["step_idx"] += 1

    summaries: List[Dict[str, Any]] = []
    out_count = 0
    for runner in runners:
        winner = runner["info"].get("state", {}).get("winner")
        ball_out = bool(runner["ball_out"] or winner == "ball_out")
        out_count += int(ball_out)
        video_path = ""
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            path = video_dir / f"{runner['case_id']}.mp4"
            render_episode_mp4(
                episode=runner["episode_records"],
                output_path=path,
                field_size=runner["simulation"].params.field_size,
                fps=video_fps,
            )
            video_path = str(path)
        summary = {
            "case_id": runner["case_id"],
            "seed": seed + cases.index(runner["case"]),
            "steps": int(runner["step_idx"]),
            "winner": winner,
            "ball_out": ball_out,
            "ball_out_step": runner["ball_out_step"],
            "ball_out_side": runner["ball_out_side"],
            "fallback_steps": int(runner["used_fallback_steps"]),
            "video_path": video_path,
        }
        summaries.append(summary)
        print(
            f"[parallel ball-out] case={summary['case_id']} steps={summary['steps']} "
            f"winner={summary['winner']} ball_out={int(summary['ball_out'])} "
            f"fallback_steps={summary['fallback_steps']}",
            flush=True,
        )

    return {
        "mode": "parallel_cases",
        "case_ids": list(case_ids),
        "num_cases": len(case_ids),
        "out_cases": int(out_count),
        "out_rate": float(out_count) / max(1, len(case_ids)),
        "query_interval": int(query_interval),
        "observation_mode": observation_mode,
        "vision_view": vision_view,
        "video_dir": str(video_dir) if video_dir is not None else "",
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        "case_summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate how often VLM rollouts would end by ball-out if HyperGym used out-of-bounds resets.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-home", type=int, default=2)
    parser.add_argument("--num-away", type=int, default=2)
    parser.add_argument("--start-case-id", type=str, default="")
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--save-artifacts-dir", type=str, default="")
    parser.add_argument("--save-json", type=str, default="logs/vlm_poc/vlm_ball_out_eval.json")
    parser.add_argument("--observation-mode", type=str, default="image_with_state_text", choices=sorted(OBSERVATION_MODES))
    parser.add_argument("--vision-view", type=str, default=DEFAULT_VISION_VIEW)
    parser.add_argument("--query-interval", type=int, default=1)
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--save-video-dir", type=str, default="")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    artifact_dir = Path(args.save_artifacts_dir) if args.save_artifacts_dir else None
    case_ids = _parse_case_ids(args.case_ids)
    video_dir = Path(args.save_video_dir) if args.save_video_dir else None
    if case_ids:
        result = run_parallel_case_ball_out_eval(
            case_ids=case_ids,
            max_steps=args.max_steps,
            seed=args.seed,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            artifact_dir=artifact_dir,
            observation_mode=args.observation_mode,
            vision_view=args.vision_view,
            query_interval=args.query_interval,
            video_dir=video_dir,
            video_fps=args.video_fps,
        )
    else:
        result = run_ball_out_eval(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            artifact_dir=artifact_dir,
            num_home=args.num_home,
            num_away=args.num_away,
            start_case_id=args.start_case_id,
            observation_mode=args.observation_mode,
            vision_view=args.vision_view,
            query_interval=args.query_interval,
        )
    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_plain(result), file_obj, ensure_ascii=False, indent=2)
    print("=== Ball-out Eval Result ===")
    if result.get("mode") == "parallel_cases":
        print(f"num_cases={result['num_cases']}")
        print(f"out_cases={result['out_cases']}")
        print(f"out_rate={result['out_rate']:.4f}")
    else:
        print(f"episodes={result['episodes']}")
        print(f"out_episodes={result['out_episodes']}")
        print(f"out_rate={result['out_rate']:.4f}")
        print(f"home_win_episodes={result['home_win_episodes']}")
        print(f"away_win_episodes={result['away_win_episodes']}")
        print(f"timeout_episodes={result['timeout_episodes']}")
    print(f"saved_result={output_path}", flush=True)


if __name__ == "__main__":
    main()
