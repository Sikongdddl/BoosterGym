from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.renderer import render_episode_mp4, render_record
from scripts.tmp.vlm_policy_poc import _get_vlm_api_key, _make_render_record, _to_plain
from scripts.tmp.vlm_vs_vlm import (
    DEFAULT_VLM_BASE_URL,
    DEFAULT_VLM_MODEL,
    OpenAICompatibleVisionVLM,
    TeamVLMDecision,
    _parse_team_decision,
    _should_query_team,
    _state_text_summary,
    _team_decision_to_action,
)


def _decision_to_plain(team_decision: Optional[TeamVLMDecision]) -> Optional[Dict[str, Any]]:
    if team_decision is None:
        return None
    return {
        player_id: {
            "policy_id": decision.policy_id,
            "target": decision.target,
            "source": decision.source,
            "reason": decision.reason,
        }
        for player_id, decision in team_decision.players.items()
    }


def _save_step_frame(image: Image.Image, output_path: Path, max_width: int, jpeg_quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if max_width > 0 and rgb.width > max_width:
        new_height = max(1, int(round(rgb.height * (max_width / rgb.width))))
        rgb = rgb.resize((max_width, new_height), Image.Resampling.BILINEAR)
    with io.BytesIO() as buffer:
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        output_path.write_bytes(buffer.getvalue())


def _query_team_on_frame(
    vlm: OpenAICompatibleVisionVLM,
    *,
    team: str,
    state: Dict[str, Any],
    state_text: str,
    frame_path: Path,
) -> Dict[str, Any]:
    raw_decision: Dict[str, Any]
    fallback_reason = "schema_invalid"
    try:
        raw_decision = vlm.decide(team=team, state=state, state_text=state_text, image_path=frame_path)
    except Exception as exc:
        raw_decision = {"error": str(exc)}
        fallback_reason = f"vlm_error:{exc}"
    parsed = _parse_team_decision(
        raw_decision if isinstance(raw_decision, dict) else {},
        team=team,
        state=state,
        fallback_reason=fallback_reason,
    )
    return {
        "raw_decision": raw_decision,
        "decision": parsed,
        "action": _team_decision_to_action(parsed),
    }


def generate_dataset(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    num_home: int,
    num_away: int,
    query_interval: int,
    vlm_model: str,
    vlm_base_url: str,
    save_dir: Path,
    frame_max_width: int,
    frame_jpeg_quality: int,
    save_video: bool,
    fps: int,
) -> Dict[str, Any]:
    save_dir.mkdir(parents=True, exist_ok=True)
    vlm = OpenAICompatibleVisionVLM(
        model=vlm_model,
        api_key=_get_vlm_api_key(),
        base_url=vlm_base_url,
    )
    manifest_entries: List[Dict[str, Any]] = []
    manifest_path = save_dir / "episodes.jsonl"

    for episode_idx in range(int(episodes)):
        episode_seed = int(seed + episode_idx)
        episode_dir = save_dir / f"episode_{episode_idx:05d}_seed{episode_seed}"
        frames_dir = episode_dir / "frames"
        episode_dir.mkdir(parents=True, exist_ok=True)
        controller = build_match_controller(
            num_home=num_home,
            num_away=num_away,
            seed=episode_seed,
            max_steps=max_steps,
            end_on_ball_out=True,
        )
        simulation = controller.simulation
        simulation.reset()

        records: List[Dict[str, Any]] = []
        recent_events: List[Dict[str, Any]] = []
        reward = 0.0
        done = False
        winner: Optional[str] = None
        step_idx = 0
        previous_owner_id: Optional[str] = None

        cached_home_action: Optional[Dict[str, Any]] = None
        cached_away_action: Optional[Dict[str, Any]] = None
        cached_home_decision: Optional[TeamVLMDecision] = None
        cached_away_decision: Optional[TeamVLMDecision] = None
        home_last_query_step: Optional[int] = None
        away_last_query_step: Optional[int] = None
        home_fallback_steps = 0
        away_fallback_steps = 0

        steps_path = episode_dir / "steps.jsonl"
        with open(steps_path, "w", encoding="utf-8") as steps_file:
            while not done and step_idx < max_steps:
                state = simulation.get_state()
                record = _make_render_record(
                    state=state,
                    reward=reward,
                    done=done,
                    action=cached_home_action,
                    away_action=cached_away_action,
                    events=recent_events,
                )
                records.append(record)
                frame_image = render_record(
                    record=record,
                    field_size=tuple(float(v) for v in state.get("field_size", [10.0, 6.0])),
                    frame_size=(960, 640),
                )
                frame_path = frames_dir / f"step_{step_idx:04d}.jpg"
                _save_step_frame(
                    image=frame_image,
                    output_path=frame_path,
                    max_width=frame_max_width,
                    jpeg_quality=frame_jpeg_quality,
                )

                home_state_text = _state_text_summary(state, recent_events, team="home")
                away_state_text = _state_text_summary(state, recent_events, team="away")
                should_query_home = _should_query_team(
                    team="home",
                    state=state,
                    recent_events=recent_events,
                    cached_decision=cached_home_decision,
                    cached_action=cached_home_action,
                    last_query_step=home_last_query_step,
                    step_idx=step_idx,
                    max_query_interval=query_interval,
                    previous_owner_id=previous_owner_id,
                )
                should_query_away = _should_query_team(
                    team="away",
                    state=state,
                    recent_events=recent_events,
                    cached_decision=cached_away_decision,
                    cached_action=cached_away_action,
                    last_query_step=away_last_query_step,
                    step_idx=step_idx,
                    max_query_interval=query_interval,
                    previous_owner_id=previous_owner_id,
                )

                home_query_payload = None
                away_query_payload = None
                if should_query_home:
                    home_query = _query_team_on_frame(
                        vlm=vlm,
                        team="home",
                        state=state,
                        state_text=home_state_text,
                        frame_path=frame_path,
                    )
                    cached_home_decision = home_query["decision"]
                    cached_home_action = home_query["action"]
                    home_last_query_step = step_idx
                    if any(decision.source != "vlm" for decision in cached_home_decision.players.values()):
                        home_fallback_steps += 1
                    home_query_payload = {
                        "raw_decision": home_query["raw_decision"],
                        "parsed_decision": _decision_to_plain(cached_home_decision),
                        "env_action": cached_home_action,
                    }

                if should_query_away:
                    away_query = _query_team_on_frame(
                        vlm=vlm,
                        team="away",
                        state=state,
                        state_text=away_state_text,
                        frame_path=frame_path,
                    )
                    cached_away_decision = away_query["decision"]
                    cached_away_action = away_query["action"]
                    away_last_query_step = step_idx
                    if any(decision.source != "vlm" for decision in cached_away_decision.players.values()):
                        away_fallback_steps += 1
                    away_query_payload = {
                        "raw_decision": away_query["raw_decision"],
                        "parsed_decision": _decision_to_plain(cached_away_decision),
                        "env_action": cached_away_action,
                    }

                if cached_home_action is None or cached_away_action is None:
                    raise RuntimeError("Both teams must have an action before stepping the simulator.")

                next_obs, reward, done, info = simulation.step(cached_home_action, opponent_action=cached_away_action)
                del next_obs
                recent_events = list(info.get("events", []))
                winner = info.get("winner")

                step_payload = {
                    "episode": episode_idx,
                    "seed": episode_seed,
                    "step": step_idx,
                    "frame_path": str(frame_path.relative_to(episode_dir)),
                    "state": state,
                    "home_state_text": home_state_text,
                    "away_state_text": away_state_text,
                    "recent_events": recent_events,
                    "home_should_query": should_query_home,
                    "away_should_query": should_query_away,
                    "home_query": home_query_payload,
                    "away_query": away_query_payload,
                    "active_home_decision": _decision_to_plain(cached_home_decision),
                    "active_away_decision": _decision_to_plain(cached_away_decision),
                    "home_action": cached_home_action,
                    "away_action": cached_away_action,
                    "reward": reward,
                    "done": done,
                    "winner": winner,
                }
                steps_file.write(json.dumps(_to_plain(step_payload), ensure_ascii=False) + "\n")
                previous_owner_id = state.get("ball_owner_id")
                step_idx += 1

        video_path = None
        if save_video:
            video_path = episode_dir / "episode.mp4"
            render_episode_mp4(
                episode=records,
                output_path=video_path,
                field_size=tuple(float(v) for v in simulation.params.field_size),
                fps=fps,
                frame_size=(960, 640),
            )

        episode_summary = {
            "episode": episode_idx,
            "seed": episode_seed,
            "steps": step_idx,
            "winner": winner,
            "home_fallback_steps": home_fallback_steps,
            "away_fallback_steps": away_fallback_steps,
            "frames_dir": str(frames_dir.relative_to(save_dir)),
            "steps_path": str(steps_path.relative_to(save_dir)),
            "video_path": str(video_path.relative_to(save_dir)) if video_path is not None else None,
        }
        (episode_dir / "summary.json").write_text(
            json.dumps(_to_plain(episode_summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_entries.append(episode_summary)
        with open(manifest_path, "a", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(_to_plain(episode_summary), ensure_ascii=False) + "\n")
        print(
            f"[dataset] ep={episode_idx} seed={episode_seed} steps={step_idx} "
            f"winner={winner} home_fb={home_fallback_steps} away_fb={away_fallback_steps}"
        )

    result = {
        "episodes": int(episodes),
        "seed": int(seed),
        "num_home": int(num_home),
        "num_away": int(num_away),
        "max_steps": int(max_steps),
        "query_interval": int(query_interval),
        "frame_max_width": int(frame_max_width),
        "frame_jpeg_quality": int(frame_jpeg_quality),
        "save_video": bool(save_video),
        "fps": int(fps),
        "vlm_model": vlm_model,
        "vlm_base_url": vlm_base_url,
        "save_dir": str(save_dir),
        "episode_summaries": manifest_entries,
    }
    (save_dir / "dataset_summary.json").write_text(
        json.dumps(_to_plain(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact VLM-vs-VLM HyperGym dataset for BC/RL.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-home", type=int, default=2)
    parser.add_argument("--num-away", type=int, default=2)
    parser.add_argument("--query-interval", type=int, default=5)
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--save-dir", type=str, default="datasets/vlm_vs_vlm_dataset")
    parser.add_argument("--frame-max-width", type=int, default=320)
    parser.add_argument("--frame-jpeg-quality", type=int, default=80)
    parser.add_argument("--save-video", action="store_true", help="Also export an MP4 per episode for visual inspection.")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    generate_dataset(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        num_home=args.num_home,
        num_away=args.num_away,
        query_interval=args.query_interval,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        save_dir=Path(args.save_dir),
        frame_max_width=args.frame_max_width,
        frame_jpeg_quality=args.frame_jpeg_quality,
        save_video=args.save_video,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
