from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import isaacgym

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vlm_oracle_correctness_eval import evaluate_samples
from scripts.vlm_oracle_correctness_eval import generate_oracle_dataset
from scripts.vlm_oracle_correctness_eval import _parse_case_ids
from scripts.vlm_oracle_correctness_eval import _parse_episode_seeds
from scripts.vlm_policy_poc import DEFAULT_VLM_BASE_URL
from scripts.vlm_policy_poc import DEFAULT_VLM_MODEL
from scripts.vlm_policy_poc import OBSERVATION_MODES
from scripts.vlm_policy_poc import OpenAICompatibleVisionVLM
from scripts.vlm_policy_poc import _get_vlm_api_key
from scripts.vlm_policy_poc import _make_render_record
from scripts.vlm_policy_poc import _parse_team_decision
from scripts.vlm_policy_poc import _save_artifact
from scripts.vlm_policy_poc import _state_text_summary
from scripts.vlm_policy_poc import _team_decision_to_action
from scripts.vlm_policy_poc import _to_plain
from scripts.vlm_policy_poc import render_record_with_camera


IMAGE_SOURCES = {
    "hyper_bev",
    "isaac_topdown",
}


def _build_vlm(model: str, base_url: str) -> OpenAICompatibleVisionVLM:
    return OpenAICompatibleVisionVLM(model=model, api_key=_get_vlm_api_key(), base_url=base_url)


def _capture_isaac_image(booster_env, state: Dict[str, Any], width: int, height: int):
    booster_env.set_from_hyper_state(state)
    rgb = booster_env.controller.capture_policy_frame(
        booster_env.root_states,
        env_idx=0,
        follow_actor_index=0,
        width=width,
        height=height,
    )
    from PIL import Image

    return Image.fromarray(rgb)


def _query_vlm_on_sample(
    vlm: OpenAICompatibleVisionVLM,
    sample: Dict[str, Any],
    artifact_dir: Path | None,
    observation_mode: str,
    image_source: str,
    vision_view: str,
    booster_env,
    frame_size: tuple[int, int],
):
    state = sample["state"]
    record = _make_render_record(
        state=state,
        reward=sample["reward"],
        done=sample["done"],
        action=sample["home_action"],
        away_action=sample["away_action"],
        events=sample["events"],
    )
    state_text = _state_text_summary(state, sample["events"])
    if image_source == "hyper_bev":
        image = render_record_with_camera(
            record=record,
            field_size=tuple(float(v) for v in state.get("field_size", [10.0, 6.0])),
            frame_size=frame_size,
            camera_mode=vision_view,
        )
    elif image_source == "isaac_topdown":
        image = _capture_isaac_image(booster_env, state, width=frame_size[0], height=frame_size[1])
    else:
        raise ValueError(f"Unknown image_source={image_source}")

    if artifact_dir is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file_obj:
            frame_path = Path(file_obj.name)
        image.save(frame_path)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        frame_path = artifact_dir / f"{sample['sample_id']}.png"
        image.save(frame_path)

    raw_decision: Dict[str, Any]
    fallback_reason = "schema_invalid"
    try:
        prompt_state_text = state_text if observation_mode == "image_with_state_text" else ""
        raw_decision = vlm.decide(
            state=state,
            state_text=prompt_state_text,
            image_path=frame_path,
            vision_view=vision_view,
        )
    except Exception as exc:
        raw_decision = {"error": str(exc)}
        fallback_reason = f"vlm_error:{exc}"

    team_decision = _parse_team_decision(
        raw_decision if isinstance(raw_decision, dict) else {},
        state=state,
        fallback_reason=fallback_reason,
    )
    action = _team_decision_to_action(team_decision)

    artifact_payload = {
        "artifact_id": sample["sample_id"],
        "observation_mode": observation_mode,
        "vision_view": vision_view,
        "image_source": image_source,
        "state_text": state_text,
        "state": state,
        "raw_decision": raw_decision,
        "parsed_decision": {
            player_id: {
                "policy_id": decision.policy_id,
                "target": decision.target,
                "source": decision.source,
                "reason": decision.reason,
            }
            for player_id, decision in team_decision.players.items()
        },
        "env_action": action,
        "record": record,
    }

    if artifact_dir is not None:
        artifact_paths = _save_artifact(artifact_dir, sample["sample_id"], image, artifact_payload)
    else:
        artifact_paths = {"image_path": str(frame_path)}
        frame_path.unlink(missing_ok=True)
    return {
        "raw_decision": raw_decision,
        "decision": team_decision,
        "action": action,
        "artifact_paths": artifact_paths,
    }


def evaluate_samples_with_image_source(
    vlm: OpenAICompatibleVisionVLM,
    samples,
    artifact_dir: Path | None,
    target_threshold: float,
    observation_mode: str,
    vision_view: str,
    image_source: str,
    booster_env,
    frame_size: tuple[int, int],
):
    from collections import Counter, defaultdict
    import math

    total_players = 0
    correct_policy = 0
    exact_step_matches = 0
    target_hits = 0
    target_errors = []
    per_policy_total = Counter()
    per_policy_correct = Counter()
    per_player_total = Counter()
    per_player_correct = Counter()
    confusion = defaultdict(Counter)
    episode_totals = Counter()
    episode_policy_correct = Counter()
    episode_target_hits = Counter()
    episode_exact_step_matches = Counter()
    episode_target_errors = defaultdict(list)
    episode_sample_counts = Counter()
    episode_seeds = {}
    evaluated_samples = []

    from scripts.vlm_oracle_correctness_eval import _euclidean, _oracle_policy_id

    for sample in samples:
        episode_idx = int(sample["episode"])
        episode_seeds[episode_idx] = int(sample["episode_seed"])
        query = _query_vlm_on_sample(
            vlm=vlm,
            sample=sample,
            artifact_dir=artifact_dir,
            observation_mode=observation_mode,
            image_source=image_source,
            vision_view=vision_view,
            booster_env=booster_env,
            frame_size=frame_size,
        )
        predicted_actions = query["decision"].players
        oracle_actions = sample["home_action"]
        sample_all_correct = True
        player_rows = []
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
            player_rows.append(
                {
                    "player_id": player_id,
                    "oracle_policy_id": oracle_policy,
                    "oracle_target": oracle_target.tolist(),
                    "pred_policy_id": pred.policy_id,
                    "pred_target": pred_target.tolist(),
                    "policy_ok": bool(policy_ok),
                    "target_ok": bool(target_ok),
                    "target_error": float(target_error),
                    "reason": pred.reason,
                    "source": pred.source,
                }
            )
        exact_step_matches += int(sample_all_correct)
        episode_exact_step_matches[episode_idx] += int(sample_all_correct)
        evaluated_samples.append(
            {
                "sample_id": sample["sample_id"],
                "episode": episode_idx,
                "episode_seed": int(sample["episode_seed"]),
                "step": int(sample["step"]),
                "start_case_id": sample.get("start_case_id", ""),
                "artifact_paths": query["artifact_paths"],
                "players": player_rows,
                "all_players_policy_correct": bool(sample_all_correct),
            }
        )

    episode_metrics = []
    for episode_idx in sorted(episode_sample_counts):
        errors = episode_target_errors[episode_idx]
        episode_metrics.append(
            {
                "episode": int(episode_idx),
                "episode_seed": int(episode_seeds.get(episode_idx, -1)),
                "num_samples": int(episode_sample_counts[episode_idx]),
                "num_player_predictions": int(episode_totals[episode_idx]),
                "policy_accuracy": float(episode_policy_correct[episode_idx]) / max(1, int(episode_totals[episode_idx])),
                "step_exact_match_rate": float(episode_exact_step_matches[episode_idx]) / max(1, int(episode_sample_counts[episode_idx])),
                "target_hit_rate": float(episode_target_hits[episode_idx]) / max(1, int(episode_totals[episode_idx])),
                "mean_target_error": float(np.mean(errors)) if errors else math.nan,
                "median_target_error": float(np.median(errors)) if errors else math.nan,
            }
        )

    return {
        "num_episodes": int(len(episode_metrics)),
        "num_samples": int(len(samples)),
        "num_player_predictions": int(total_players),
        "policy_accuracy": float(correct_policy) / max(1, total_players),
        "step_exact_match_rate": float(exact_step_matches) / max(1, len(samples)),
        "target_hit_rate": float(target_hits) / max(1, total_players),
        "mean_target_error": float(np.mean(target_errors)) if target_errors else math.nan,
        "median_target_error": float(np.median(target_errors)) if target_errors else math.nan,
        "macro_policy_accuracy": float(np.mean([item["policy_accuracy"] for item in episode_metrics])) if episode_metrics else math.nan,
        "macro_step_exact_match_rate": float(np.mean([item["step_exact_match_rate"] for item in episode_metrics])) if episode_metrics else math.nan,
        "per_policy_accuracy": {
            policy_id: float(per_policy_correct[policy_id]) / max(1, int(per_policy_total[policy_id]))
            for policy_id in sorted(per_policy_total)
        },
        "per_player_accuracy": {
            player_id: float(per_player_correct[player_id]) / max(1, int(per_player_total[player_id]))
            for player_id in sorted(per_player_total)
        },
        "confusion": {
            oracle_policy: {pred_policy: int(count) for pred_policy, count in sorted(counter.items())}
            for oracle_policy, counter in sorted(confusion.items())
        },
        "episode_metrics": episode_metrics,
        "samples": evaluated_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare HyperGym BEV images vs IsaacGym top-down images on the same oracle states.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--samples-per-episode", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-seeds", type=str, default="")
    parser.add_argument("--num-home", type=int, default=2)
    parser.add_argument("--num-away", type=int, default=2)
    parser.add_argument("--target-threshold", type=float, default=0.75)
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--save-artifacts-dir", type=str, default="logs/vlm_poc/oracle_image_source")
    parser.add_argument("--save-json", type=str, default="logs/vlm_poc/oracle_image_source_summary.json")
    parser.add_argument("--observation-mode", type=str, default="image_only", choices=sorted(OBSERVATION_MODES))
    parser.add_argument("--vision-view", type=str, default="global")
    parser.add_argument("--image-source", type=str, default="hyper_bev", choices=sorted(IMAGE_SOURCES))
    parser.add_argument("--start-case-ids", type=str, default="")
    parser.add_argument("--frame-width", type=int, default=960)
    parser.add_argument("--frame-height", type=int, default=640)
    args = parser.parse_args()

    if args.episode_seeds:
        episode_seeds = _parse_episode_seeds(args.episode_seeds)
    else:
        episode_seeds = [args.seed + idx for idx in range(args.episodes)]
    start_case_ids = _parse_case_ids(args.start_case_ids)

    samples = generate_oracle_dataset(
        max_steps=args.max_steps,
        num_home=args.num_home,
        num_away=args.num_away,
        samples_per_episode=args.samples_per_episode,
        episode_seeds=episode_seeds,
        start_case_ids=start_case_ids,
    )
    artifact_dir = Path(args.save_artifacts_dir) if args.save_artifacts_dir else None
    vlm = _build_vlm(args.vlm_model, args.vlm_base_url)

    booster_env = None
    if args.image_source == "isaac_topdown":
        import yaml
        import isaacgym

        cfg_path = ROOT / "envs" / "boosterT12v2" / "BoosterT12v2Env.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.load(fh.read(), Loader=yaml.FullLoader)
        cfg.setdefault("basic", {})
        cfg["basic"]["task"] = "BoosterT12v2Env"
        cfg["basic"]["headless"] = True
        cfg.setdefault("viewer", {})
        cfg["viewer"]["capture_for_policy"] = True
        from envs.boosterT12v2.BoosterT12v2Env import BoosterT12v2Env

        booster_env = BoosterT12v2Env(cfg, None)

    result = evaluate_samples_with_image_source(
        vlm=vlm,
        samples=samples,
        artifact_dir=artifact_dir,
        target_threshold=args.target_threshold,
        observation_mode=args.observation_mode,
        vision_view=args.vision_view,
        image_source=args.image_source,
        booster_env=booster_env,
        frame_size=(args.frame_width, args.frame_height),
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
        "image_source": args.image_source,
        "start_case_ids": start_case_ids,
        "frame_size": [args.frame_width, args.frame_height],
    }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_plain(result), file_obj, ensure_ascii=False, indent=2)

    print("=== Oracle Image Source Result ===")
    print(f"image_source={args.image_source}")
    print(f"num_episodes={result['num_episodes']}")
    print(f"num_samples={result['num_samples']}")
    print(f"num_player_predictions={result['num_player_predictions']}")
    print(f"policy_accuracy={result['policy_accuracy']:.4f}")
    print(f"step_exact_match_rate={result['step_exact_match_rate']:.4f}")
    print(f"target_hit_rate={result['target_hit_rate']:.4f}")
    print(f"mean_target_error={result['mean_target_error']:.4f}")
    print(f"saved_result={output_path}")


if __name__ == "__main__":
    main()
