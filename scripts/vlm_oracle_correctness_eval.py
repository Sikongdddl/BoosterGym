from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.policies import SimpleMatchPolicy
from scripts.vlm_policy_poc import _get_benchmark_case
from scripts.vlm_policy_poc import _make_render_record
from scripts.vlm_policy_poc import _query_vlm_on_record
from scripts.vlm_policy_poc import _set_manual_state
from scripts.vlm_policy_poc import _state_text_summary
from scripts.vlm_policy_poc import _to_plain
from scripts.vlm_policy_poc import _get_vlm_api_key
from scripts.vlm_policy_poc import OBSERVATION_MODES
from scripts.vlm_policy_poc import DEFAULT_VLM_BASE_URL
from scripts.vlm_policy_poc import DEFAULT_VLM_MODEL
from scripts.vlm_policy_poc import OpenAICompatibleVisionVLM


SKILL_TO_POLICY_ID = {
    "move": "move_to_target",
    "pass": "pass_to_target",
    "trap": "trap_ball",
    "dribble": "dribble_to_target",
}


def _build_vlm(model: str, base_url: str) -> OpenAICompatibleVisionVLM:
    return OpenAICompatibleVisionVLM(model=model, api_key=_get_vlm_api_key(), base_url=base_url)


def _euclidean(a: Any, b: Any) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(av - bv))


def _oracle_policy_id(action: Dict[str, Any]) -> str:
    skill = str(action.get("skill", "move")).lower()
    return SKILL_TO_POLICY_ID.get(skill, "move_to_target")


def _sample_episode_steps(episode: List[Dict[str, Any]], max_samples: int) -> List[int]:
    playable = list(range(1, len(episode)))
    if len(playable) <= max_samples:
        return playable
    positions = np.linspace(1, len(episode) - 1, num=max_samples, dtype=int)
    return sorted(set(int(pos) for pos in positions))


def _parse_episode_seeds(seed_text: str) -> List[int]:
    values = [item.strip() for item in seed_text.split(",") if item.strip()]
    if not values:
        raise ValueError("episode seed list is empty")
    return [int(item) for item in values]


def _parse_case_ids(case_text: str) -> List[str]:
    return [item.strip() for item in case_text.split(",") if item.strip()]


def generate_oracle_dataset(
    max_steps: int,
    num_home: int,
    num_away: int,
    samples_per_episode: int,
    episode_seeds: List[int],
    start_case_ids: List[str] | None = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    case_ids = start_case_ids or []
    episode_idx = 0
    if not case_ids:
        for env_seed in episode_seeds:
            controller = build_match_controller(
                num_home=num_home,
                num_away=num_away,
                seed=env_seed,
                max_steps=max_steps,
            )
            episode = controller.collect_match_episode(
                home_policy=SimpleMatchPolicy(team="home", seed=1000 + env_seed),
                away_policy=SimpleMatchPolicy(team="away", seed=2000 + env_seed),
                max_steps=max_steps,
            )
            sampled_steps = _sample_episode_steps(episode, max_samples=samples_per_episode)
            for step_idx in sampled_steps:
                record = episode[step_idx]
                state = record["info"]["state"]
                sample = {
                    "sample_id": f"ep{episode_idx:03d}_step{step_idx:03d}",
                    "episode": episode_idx,
                    "episode_seed": env_seed,
                    "start_case_id": "",
                    "step": step_idx,
                    "state": state,
                    "reward": record["reward"],
                    "done": record["done"],
                    "home_action": record["action"],
                    "away_action": record["away_action"],
                    "events": record["info"].get("events", []),
                }
                samples.append(sample)
            episode_idx += 1
        return samples

    for case_id in case_ids:
        case = _get_benchmark_case(case_id)
        for env_seed in episode_seeds:
            controller = build_match_controller(
                num_home=case.num_home,
                num_away=case.num_away,
                seed=env_seed,
                max_steps=max(int(case.state_spec.get("step", 0)) + max_steps, max_steps),
            )
            simulation = controller.simulation
            state = _set_manual_state(simulation, case.state_spec)
            home_policy = SimpleMatchPolicy(team="home", seed=1000 + env_seed)
            away_policy = SimpleMatchPolicy(team="away", seed=2000 + env_seed)
            episode = [{
                "obs": simulation.get_obs(),
                "reward": float(case.reward),
                "done": bool(case.done),
                "action": case.prior_action,
                "away_action": case.prior_away_action,
                "info": {
                    "events": list(case.recent_events),
                    "state": state,
                    "actions": {},
                },
            }]
            done = bool(case.done)
            for _ in range(max_steps):
                if done:
                    break
                current_state = simulation.get_state()
                home_action = home_policy(current_state)
                obs, reward, done, info = simulation.step(home_action, opponent_action=away_policy(current_state))
                episode.append({
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "action": info["actions"]["home"],
                    "away_action": info["actions"]["away"],
                    "info": info,
                })

            sampled_steps = _sample_episode_steps(episode, max_samples=samples_per_episode)
            for step_idx in sampled_steps:
                record = episode[step_idx]
                state = record["info"]["state"]
                sample = {
                    "sample_id": f"{case_id}_ep{episode_idx:03d}_step{step_idx:03d}",
                    "episode": episode_idx,
                    "episode_seed": env_seed,
                    "start_case_id": case_id,
                    "step": step_idx,
                    "state": state,
                    "reward": record["reward"],
                    "done": record["done"],
                    "home_action": record["action"],
                    "away_action": record["away_action"],
                    "events": record["info"].get("events", []),
                }
                samples.append(sample)
            episode_idx += 1
    return samples


def evaluate_samples(
    vlm: OpenAICompatibleVisionVLM,
    samples: List[Dict[str, Any]],
    artifact_dir: Path | None,
    target_threshold: float,
    observation_mode: str,
    vision_view: str,
) -> Dict[str, Any]:
    total_players = 0
    correct_policy = 0
    exact_step_matches = 0
    target_hits = 0
    target_errors: List[float] = []
    per_policy_total: Counter = Counter()
    per_policy_correct: Counter = Counter()
    per_player_total: Counter = Counter()
    per_player_correct: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)
    episode_totals: Counter = Counter()
    episode_policy_correct: Counter = Counter()
    episode_target_hits: Counter = Counter()
    episode_exact_step_matches: Counter = Counter()
    episode_target_errors: Dict[int, List[float]] = defaultdict(list)
    episode_sample_counts: Counter = Counter()
    episode_seeds: Dict[int, int] = {}
    evaluated_samples: List[Dict[str, Any]] = []

    for sample in samples:
        episode_idx = int(sample["episode"])
        episode_seeds[episode_idx] = int(sample["episode_seed"])
        record = _make_render_record(
            state=sample["state"],
            reward=sample["reward"],
            done=sample["done"],
            action=sample["home_action"],
            away_action=sample["away_action"],
            events=sample["events"],
        )
        state_text = _state_text_summary(sample["state"], sample["events"])
        query = _query_vlm_on_record(
            vlm=vlm,
            artifact_id=sample["sample_id"],
            record=record,
            state_text=state_text,
            artifact_dir=artifact_dir,
            observation_mode=observation_mode,
            vision_view=vision_view,
        )
        predicted_actions = query["decision"].players
        oracle_actions = sample["home_action"]

        sample_all_correct = True
        player_rows: List[Dict[str, Any]] = []
        episode_sample_counts[episode_idx] += 1

        for player_id, oracle_action in oracle_actions.items():
            oracle_policy = _oracle_policy_id(oracle_action)
            oracle_target = np.asarray(oracle_action["target"], dtype=np.float32)
            pred = predicted_actions[player_id]
            pred_target = np.asarray(pred.target, dtype=np.float32)
            policy_ok = pred.policy_id == oracle_policy
            target_error = _euclidean(pred_target, oracle_target)
            target_ok = target_error <= target_threshold

            total_players += 1
            correct_policy += int(policy_ok)
            target_hits += int(target_ok)
            target_errors.append(target_error)
            per_policy_total[oracle_policy] += 1
            per_policy_correct[oracle_policy] += int(policy_ok)
            per_player_total[player_id] += 1
            per_player_correct[player_id] += int(policy_ok)
            confusion[oracle_policy][pred.policy_id] += 1
            episode_totals[episode_idx] += 1
            episode_policy_correct[episode_idx] += int(policy_ok)
            episode_target_hits[episode_idx] += int(target_ok)
            episode_target_errors[episode_idx].append(target_error)
            sample_all_correct = sample_all_correct and policy_ok

            player_rows.append({
                "player_id": player_id,
                "oracle_policy_id": oracle_policy,
                "oracle_target": oracle_target.tolist(),
                "pred_policy_id": pred.policy_id,
                "pred_target": pred_target.tolist(),
                "pred_source": pred.source,
                "pred_reason": pred.reason,
                "policy_correct": policy_ok,
                "target_error": target_error,
                "target_hit": target_ok,
            })

        exact_step_matches += int(sample_all_correct)
        episode_exact_step_matches[episode_idx] += int(sample_all_correct)
        evaluated_samples.append({
            "sample_id": sample["sample_id"],
            "episode": sample["episode"],
            "episode_seed": sample["episode_seed"],
            "start_case_id": sample.get("start_case_id", ""),
            "step": sample["step"],
            "observation_mode": observation_mode,
            "vision_view": vision_view,
            "state_text": state_text,
            "all_players_policy_correct": sample_all_correct,
            "artifact_paths": query["artifact_paths"],
            "players": player_rows,
        })

    policy_accuracy = float(correct_policy) / max(1, total_players)
    target_hit_rate = float(target_hits) / max(1, total_players)
    mean_target_error = float(sum(target_errors) / max(1, len(target_errors)))
    median_target_error = float(np.median(np.asarray(target_errors, dtype=np.float32))) if target_errors else math.nan
    step_exact_match_rate = float(exact_step_matches) / max(1, len(samples))
    episode_metrics = []
    for episode_idx in sorted(episode_sample_counts):
        errors = episode_target_errors[episode_idx]
        episode_metrics.append({
            "episode": int(episode_idx),
            "episode_seed": int(episode_seeds[episode_idx]),
            "num_samples": int(episode_sample_counts[episode_idx]),
            "num_player_predictions": int(episode_totals[episode_idx]),
            "policy_accuracy": float(episode_policy_correct[episode_idx]) / max(1, int(episode_totals[episode_idx])),
            "step_exact_match_rate": float(episode_exact_step_matches[episode_idx]) / max(1, int(episode_sample_counts[episode_idx])),
            "target_hit_rate": float(episode_target_hits[episode_idx]) / max(1, int(episode_totals[episode_idx])),
            "mean_target_error": float(sum(errors) / max(1, len(errors))),
            "median_target_error": float(np.median(np.asarray(errors, dtype=np.float32))) if errors else math.nan,
        })
    macro_policy_accuracy = float(np.mean([item["policy_accuracy"] for item in episode_metrics])) if episode_metrics else math.nan
    macro_step_exact_match_rate = float(np.mean([item["step_exact_match_rate"] for item in episode_metrics])) if episode_metrics else math.nan
    macro_target_hit_rate = float(np.mean([item["target_hit_rate"] for item in episode_metrics])) if episode_metrics else math.nan
    macro_mean_target_error = float(np.mean([item["mean_target_error"] for item in episode_metrics])) if episode_metrics else math.nan
    macro_median_target_error = float(np.mean([item["median_target_error"] for item in episode_metrics])) if episode_metrics else math.nan

    return {
        "num_samples": len(samples),
        "num_player_predictions": total_players,
        "num_episodes": len(episode_metrics),
        "policy_accuracy": policy_accuracy,
        "step_exact_match_rate": step_exact_match_rate,
        "target_hit_rate": target_hit_rate,
        "mean_target_error": mean_target_error,
        "median_target_error": median_target_error,
        "episode_metrics": episode_metrics,
        "macro_policy_accuracy": macro_policy_accuracy,
        "macro_step_exact_match_rate": macro_step_exact_match_rate,
        "macro_target_hit_rate": macro_target_hit_rate,
        "macro_mean_target_error": macro_mean_target_error,
        "macro_median_target_error": macro_median_target_error,
        "target_threshold": target_threshold,
        "per_policy_accuracy": {
            policy_id: {
                "total": int(per_policy_total[policy_id]),
                "correct": int(per_policy_correct[policy_id]),
                "accuracy": float(per_policy_correct[policy_id]) / max(1, int(per_policy_total[policy_id])),
            }
            for policy_id in sorted(per_policy_total)
        },
        "per_player_accuracy": {
            player_id: {
                "total": int(per_player_total[player_id]),
                "correct": int(per_player_correct[player_id]),
                "accuracy": float(per_player_correct[player_id]) / max(1, int(per_player_total[player_id])),
            }
            for player_id in sorted(per_player_total)
        },
        "policy_confusion": {
            oracle_policy: {pred_policy: int(count) for pred_policy, count in sorted(counter.items())}
            for oracle_policy, counter in sorted(confusion.items())
        },
        "samples": evaluated_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VLM multi-player outputs against oracle scripted trajectories in HyperGym."
    )
    parser.add_argument("--episodes", type=int, default=3, help="number of oracle episodes to generate")
    parser.add_argument("--max-steps", type=int, default=30, help="max steps per oracle episode")
    parser.add_argument("--samples-per-episode", type=int, default=8, help="sampled states per oracle episode")
    parser.add_argument("--seed", type=int, default=7, help="base seed for oracle episode generation")
    parser.add_argument(
        "--episode-seeds",
        type=str,
        default="",
        help="optional explicit env seeds, e.g. 7,8,9; overrides --seed and fixes the trajectory set for later ablations",
    )
    parser.add_argument("--num-home", type=int, default=2, help="home team player count")
    parser.add_argument("--num-away", type=int, default=2, help="away team player count")
    parser.add_argument("--target-threshold", type=float, default=0.75, help="meters for target-hit metric")
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL, help="vision model name")
    parser.add_argument(
        "--vlm-base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL),
        help="OpenAI-compatible API base url",
    )
    parser.add_argument(
        "--save-artifacts-dir",
        type=str,
        default="logs/vlm_poc/oracle_correctness",
        help="directory for saved frame/json artifacts",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="logs/vlm_poc/oracle_correctness_summary.json",
        help="path for final json summary",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="image_with_state_text",
        choices=sorted(OBSERVATION_MODES),
        help="image_with_state_text: 图像+状态文本；image_only: 仅图像，不把状态文本发给 VLM",
    )
    parser.add_argument(
        "--vision-view",
        type=str,
        default="global",
        help="视觉输入视角。global 为原始全局视角；ego_home_0 这类值表示以对应球员为中心的局部视角",
    )
    parser.add_argument(
        "--start-case-ids",
        type=str,
        default="",
        help="optional comma-separated benchmark case ids; if provided, generate oracle rollouts from those manual start states",
    )
    args = parser.parse_args()

    if args.episode_seeds:
        episode_seeds = _parse_episode_seeds(args.episode_seeds)
    else:
        episode_seeds = [args.seed + idx for idx in range(args.episodes)]
    start_case_ids = _parse_case_ids(args.start_case_ids)

    artifact_dir = Path(args.save_artifacts_dir) if args.save_artifacts_dir else None
    samples = generate_oracle_dataset(
        max_steps=args.max_steps,
        num_home=args.num_home,
        num_away=args.num_away,
        samples_per_episode=args.samples_per_episode,
        episode_seeds=episode_seeds,
        start_case_ids=start_case_ids,
    )
    vlm = _build_vlm(model=args.vlm_model, base_url=args.vlm_base_url)
    result = evaluate_samples(
        vlm=vlm,
        samples=samples,
        artifact_dir=artifact_dir,
        target_threshold=args.target_threshold,
        observation_mode=args.observation_mode,
        vision_view=args.vision_view,
    )
    result["config"] = {
        "episodes": args.episodes,
        "episode_seeds": episode_seeds,
        "max_steps": args.max_steps,
        "samples_per_episode": args.samples_per_episode,
        "seed": args.seed,
        "num_home": args.num_home,
        "num_away": args.num_away,
        "vlm_model": args.vlm_model,
        "vlm_base_url": args.vlm_base_url,
        "observation_mode": args.observation_mode,
        "vision_view": args.vision_view,
        "start_case_ids": start_case_ids,
    }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_plain(result), file_obj, ensure_ascii=False, indent=2)

    print("=== Oracle Correctness Result ===")
    print(f"num_episodes={result['num_episodes']}")
    print(f"num_samples={result['num_samples']}")
    print(f"num_player_predictions={result['num_player_predictions']}")
    print(f"policy_accuracy={result['policy_accuracy']:.4f}")
    print(f"step_exact_match_rate={result['step_exact_match_rate']:.4f}")
    print(f"target_hit_rate={result['target_hit_rate']:.4f}")
    print(f"mean_target_error={result['mean_target_error']:.4f}")
    print(f"median_target_error={result['median_target_error']:.4f}")
    print(f"macro_policy_accuracy={result['macro_policy_accuracy']:.4f}")
    print(f"macro_step_exact_match_rate={result['macro_step_exact_match_rate']:.4f}")
    print(f"saved_result={output_path}")


if __name__ == "__main__":
    main()
