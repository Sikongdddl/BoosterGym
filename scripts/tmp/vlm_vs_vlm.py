from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as url_error
from urllib import request as url_request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.renderer import render_episode_mp4, render_record
from scripts.tmp.vlm_policy_poc import _clip_target, _get_vlm_api_key, _make_render_record, _prepare_vlm_image_data_url, _save_artifact, _to_plain
from scripts.tmp.vlm_policy_poc import DEFAULT_VLM_BASE_URL, DEFAULT_VLM_MODEL


ALLOWED_POLICY_IDS = {
    "move_to_target",
    "pass_to_target",
    "trap_ball",
}


@dataclass
class VLMDecision:
    policy_id: str
    target: np.ndarray
    source: str
    reason: str


@dataclass
class TeamVLMDecision:
    players: Dict[str, VLMDecision]


class OpenAICompatibleVisionVLM:
    def __init__(self, model: str, api_key: str, base_url: str, image_max_width: int = 320, image_jpeg_quality: int = 80):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.image_max_width = int(image_max_width)
        self.image_jpeg_quality = int(image_jpeg_quality)

    def decide(self, team: str, state: Dict[str, Any], state_text: str, image_path: Path) -> Dict[str, Any]:
        field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
        team_players = [player for player in state.get("players", []) if player.get("team") == team]
        player_schema = ", ".join(
            f'"{player["player_id"]}": {{"policy_id": "<move_to_target|pass_to_target|trap_ball>", "target": [x, y], "reason": "<short_reason>"}}'
            for player in team_players
        )
        team_player_ids = ", ".join(player["player_id"] for player in team_players)
        image_data_url = _prepare_vlm_image_data_url(
            image_path=image_path,
            max_width=self.image_max_width,
            jpeg_quality=self.image_jpeg_quality,
        )

        attack_direction = "right" if team == "home" else "left"
        own_prefix = f"{team}_"
        schema_text = 'Return ONLY valid JSON with this shape: {"players": {' + player_schema + "}}"
        policy_semantics = (
            "Policy semantics:\n"
            "- move_to_target: use when this team should run to space, close down a loose ball, or reposition.\n"
            "- trap_ball: use when the ball is free or moving and this team should first secure control near the ball.\n"
            "- pass_to_target: this means kicking the ball to a target, not only passing to a teammate. Use it for passes, clearances, and direct shots on goal when a player can plausibly strike the ball now.\n"
            "Hard constraints:\n"
            f"- If owner starts with '{own_prefix}', do not output trap_ball because this team already controls the ball.\n"
            "- If owner is free, prefer move_to_target or trap_ball over pass_to_target.\n"
            "- If this team already controls the ball, prefer move_to_target for ball progression because dribble is not available in this environment.\n"
            "- If this team has the ball near goal and the shooting lane is open, prefer pass_to_target aimed inside the goal mouth instead of a harmless extra pass.\n"
            "- Treat pass_to_target as the only available kick action: if a direct shot is best, encode that shot with pass_to_target.\n"
            "- Treat the touchlines as dangerous: when the ball is near the top or bottom boundary, avoid targets that keep pushing play along or into the sideline.\n"
            "- Near a sideline, prefer recycling the ball back inward or switching to safer interior space over continuing a risky edge run.\n"
            f"- Favor targets that progress the attack toward the {attack_direction} side.\n"
            "- Avoid meaningless pass_to_target to the current ball location or to a point behind the attack."
        )
        prompt = (
            f"You are a high-level soccer tactics model for the {team} team.\n"
            "The image is the primary input. The text is auxiliary context.\n"
            f"You must output one action for every {team} player: {team_player_ids}.\n"
            f"Field size: width={field[0]:.2f}, height={field[1]:.2f}\n"
            f"State summary: {state_text}\n"
            f"{policy_semantics}\n"
            f"{schema_text}\n"
            "The answer must be a single JSON object and every target must stay inside field bounds."
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 260,
        }
        request_obj = url_request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with url_request.urlopen(request_obj, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except url_error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            raise RuntimeError(f"OpenAI-compatible API HTTPError: {exc.code} {exc.reason} {detail}") from exc
        except url_error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible API URLError: {exc.reason}") from exc

        try:
            text_output = body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"API response has no assistant content: {body}") from exc

        if isinstance(text_output, list):
            text_output = "".join(part.get("text", "") for part in text_output if isinstance(part, dict))

        try:
            return json.loads(text_output)
        except json.JSONDecodeError:
            cleaned = str(text_output).strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.replace("json\n", "", 1).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end + 1])
            raise RuntimeError(f"VLM output is not valid JSON: {text_output}")


def _state_text_summary(state: Dict[str, Any], recent_events: List[Dict[str, Any]], team: str) -> str:
    ball = np.asarray(state["ball_position"], dtype=np.float32)
    ball_vel = np.asarray(state.get("ball_velocity", [0.0, 0.0]), dtype=np.float32)
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    owner = state.get("ball_owner_id")
    team_players = [p for p in state.get("players", []) if p.get("team") == team]
    opp_team = "away" if team == "home" else "home"
    opp_players = [p for p in state.get("players", []) if p.get("team") == opp_team]

    def nearest(players: List[Dict[str, Any]]) -> tuple[str, float]:
        nearest_id = "none"
        nearest_dist = 999.0
        for player in players:
            player_pos = np.asarray(player["position"], dtype=np.float32)
            dist = float(np.linalg.norm(player_pos - ball))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = player["player_id"]
        return nearest_id, nearest_dist

    nearest_team_id, nearest_team_dist = nearest(team_players)
    nearest_opp_id, nearest_opp_dist = nearest(opp_players)
    opponent_goal_center = np.asarray([field[0], 0.5 * field[1]], dtype=np.float32) if team == "home" else np.asarray([0.0, 0.5 * field[1]], dtype=np.float32)
    ball_to_opponent_goal = float(np.linalg.norm(opponent_goal_center - ball))
    ball_speed = float(np.linalg.norm(ball_vel))

    event_tokens: List[str] = []
    for event in recent_events[-3:]:
        token = str(event.get("event_type", "unknown"))
        if "team" in event:
            token += f":{event['team']}"
        if "by" in event:
            token += f":{event['by']}"
        event_tokens.append(token)
    event_text = ", ".join(event_tokens) if event_tokens else "none"
    team_layout = ", ".join(
        f"{player['player_id']}=({float(player['position'][0]):.2f},{float(player['position'][1]):.2f})"
        for player in team_players
    )
    opp_layout = ", ".join(
        f"{player['player_id']}=({float(player['position'][0]):.2f},{float(player['position'][1]):.2f})"
        for player in opp_players
    )

    return (
        f"step={int(state.get('step', 0))}; "
        f"ball=({ball[0]:.2f},{ball[1]:.2f}); "
        f"ball_v=({ball_vel[0]:.2f},{ball_vel[1]:.2f}); speed={ball_speed:.2f}; "
        f"owner={owner or 'free'}; "
        f"nearest_team={nearest_team_id}@{nearest_team_dist:.2f}; "
        f"nearest_opponent={nearest_opp_id}@{nearest_opp_dist:.2f}; "
        f"ball_to_opponent_goal={ball_to_opponent_goal:.2f}; "
        f"{team}_players=[{team_layout}]; "
        f"{opp_team}_players=[{opp_layout}]; "
        f"recent_events=[{event_text}]"
    )


def _fallback_decision_for_player(player_id: str, team: str, state: Dict[str, Any], fallback_reason: str) -> VLMDecision:
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    ball = np.asarray(state["ball_position"], dtype=np.float32)
    goal = np.asarray([field[0], 0.5 * field[1]], dtype=np.float32) if team == "home" else np.asarray([0.0, 0.5 * field[1]], dtype=np.float32)
    team_players = [player for player in state.get("players", []) if player.get("team") == team]
    player = next((item for item in team_players if item["player_id"] == player_id), None)
    owner_id = state.get("ball_owner_id")
    step_sign = 1.0 if team == "home" else -1.0
    if owner_id == player_id and player is not None:
        pos = np.asarray(player["position"], dtype=np.float32)
        forward = pos + np.asarray([0.8 * step_sign, 0.0], dtype=np.float32)
        return VLMDecision(
            policy_id="move_to_target",
            target=_clip_target(0.7 * forward + 0.3 * goal, field),
            source="fallback",
            reason=fallback_reason,
        )
    if owner_id is None:
        nearest = None
        nearest_dist = 999.0
        for team_player in team_players:
            ppos = np.asarray(team_player["position"], dtype=np.float32)
            dist = float(np.linalg.norm(ppos - ball))
            if dist < nearest_dist:
                nearest = team_player
                nearest_dist = dist
        if nearest is not None and nearest["player_id"] == player_id:
            return VLMDecision(
                policy_id="trap_ball",
                target=_clip_target(ball, field),
                source="fallback",
                reason=fallback_reason,
            )
    if player is not None:
        return VLMDecision(
            policy_id="move_to_target",
            target=_clip_target(np.asarray(player["position"], dtype=np.float32), field),
            source="fallback",
            reason=fallback_reason,
        )
    return VLMDecision(
        policy_id="trap_ball",
        target=_clip_target(ball, field),
        source="fallback",
        reason=fallback_reason,
    )


def _parse_single_decision(raw: Dict[str, Any], team: str, state: Dict[str, Any], player_id: str, fallback_reason: str) -> VLMDecision:
    default = _fallback_decision_for_player(player_id, team=team, state=state, fallback_reason=fallback_reason)
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    ball = np.asarray(state["ball_position"], dtype=np.float32)

    if not isinstance(raw, dict):
        return default
    policy_id = str(raw.get("policy_id", "")).strip()
    target = raw.get("target")
    reason = str(raw.get("reason", ""))

    if policy_id not in ALLOWED_POLICY_IDS:
        return default
    if not isinstance(target, (list, tuple)) or len(target) < 2:
        return default
    try:
        tx = float(target[0])
        ty = float(target[1])
    except Exception:
        return default
    if not np.isfinite(tx) or not np.isfinite(ty):
        return default

    parsed_target = _clip_target(np.asarray([tx, ty], dtype=np.float32), field)
    owner_id = state.get("ball_owner_id")

    if policy_id == "trap_ball" and owner_id is not None and owner_id.startswith(f"{team}_"):
        return default
    if policy_id == "pass_to_target":
        ball_delta = float(np.linalg.norm(parsed_target - ball))
        if owner_id is None and ball_delta < 0.6:
            return default
        if ball_delta < 0.12:
            return default

    return VLMDecision(policy_id=policy_id, target=parsed_target, source="vlm", reason=reason)


def _parse_team_decision(raw: Dict[str, Any], team: str, state: Dict[str, Any], fallback_reason: str) -> TeamVLMDecision:
    team_players = [player for player in state.get("players", []) if player.get("team") == team]
    raw_players = raw.get("players", raw if isinstance(raw, dict) else {})
    parsed: Dict[str, VLMDecision] = {}
    for player in team_players:
        player_id = player["player_id"]
        player_raw = raw_players.get(player_id, {}) if isinstance(raw_players, dict) else {}
        parsed[player_id] = _parse_single_decision(player_raw, team=team, state=state, player_id=player_id, fallback_reason=fallback_reason)
    return TeamVLMDecision(players=parsed)


def _decision_to_action(decision: VLMDecision) -> Dict[str, Any]:
    mapping = {
        "move_to_target": "move",
        "pass_to_target": "pass",
        "trap_ball": "trap",
    }
    return {
        "skill": mapping.get(decision.policy_id, "move"),
        "target": decision.target.copy(),
    }


def _team_decision_to_action(team_decision: TeamVLMDecision) -> Dict[str, Any]:
    return {player_id: _decision_to_action(decision) for player_id, decision in team_decision.players.items()}


def _lookup_player(state: Dict[str, Any], player_id: str) -> Optional[Dict[str, Any]]:
    for player in state.get("players", []):
        if player.get("player_id") == player_id:
            return player
    return None


def _player_decision_complete(state: Dict[str, Any], player_id: str, decision: VLMDecision) -> bool:
    player = _lookup_player(state, player_id)
    if player is None:
        return True
    player_pos = np.asarray(player["position"], dtype=np.float32)
    target = np.asarray(decision.target, dtype=np.float32)
    ball = np.asarray(state["ball_position"], dtype=np.float32)
    ball_speed = float(np.linalg.norm(np.asarray(state.get("ball_velocity", [0.0, 0.0]), dtype=np.float32)))
    owner_id = state.get("ball_owner_id")

    if decision.policy_id == "move_to_target":
        return float(np.linalg.norm(player_pos - target)) <= 0.32
    if decision.policy_id == "trap_ball":
        near_ball = float(np.linalg.norm(player_pos - ball)) <= 0.38
        return near_ball and (ball_speed <= 0.05 or owner_id == player_id)
    if decision.policy_id == "pass_to_target":
        return owner_id != player_id
    return False


def _should_query_team(
    *,
    team: str,
    state: Dict[str, Any],
    recent_events: List[Dict[str, Any]],
    cached_decision: Optional[TeamVLMDecision],
    cached_action: Optional[Dict[str, Any]],
    last_query_step: Optional[int],
    step_idx: int,
    max_query_interval: int,
    previous_owner_id: Optional[str],
) -> bool:
    if cached_decision is None or cached_action is None or last_query_step is None:
        return True
    if step_idx - last_query_step >= max(1, int(max_query_interval)):
        return True

    owner_id = state.get("ball_owner_id")
    if owner_id != previous_owner_id:
        return True

    important_events = {
        "goal_scored",
        "ball_out_of_bounds",
        "loose_ball",
        "dead_ball",
        "loose_ball_scramble",
        "turnover",
        "ball_control_gained",
        "ball_control_lost",
        "trap_completed",
        "pass_started",
        "move_touch",
        "steal_attempt_won",
    }
    for event in recent_events:
        event_type = event.get("event_type")
        event_team = event.get("team")
        if event_type in important_events and (event_team is None or event_team == team):
            return True

    for player_id, decision in cached_decision.players.items():
        if _player_decision_complete(state, player_id, decision):
            return True
    return False


def _build_vlm(model: str, base_url: str) -> OpenAICompatibleVisionVLM:
    return OpenAICompatibleVisionVLM(model=model, api_key=_get_vlm_api_key(), base_url=base_url)


def _query_team_vlm(
    vlm: OpenAICompatibleVisionVLM,
    team: str,
    artifact_id: str,
    record: Dict[str, Any],
    state_text: str,
    artifact_dir: Optional[Path],
) -> Dict[str, Any]:
    state = record["info"]["state"]
    image = render_record(
        record=record,
        field_size=tuple(float(v) for v in state.get("field_size", [10.0, 6.0])),
        frame_size=(960, 640),
    )
    if artifact_dir is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file_obj:
            frame_path = Path(file_obj.name)
        image.save(frame_path)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        frame_path = artifact_dir / f"{artifact_id}_{team}.png"
        image.save(frame_path)

    raw_decision: Dict[str, Any]
    fallback_reason = "schema_invalid"
    try:
        raw_decision = vlm.decide(team=team, state=state, state_text=state_text, image_path=frame_path)
    except Exception as exc:
        raw_decision = {"error": str(exc)}
        fallback_reason = f"vlm_error:{exc}"

    team_decision = _parse_team_decision(raw_decision if isinstance(raw_decision, dict) else {}, team=team, state=state, fallback_reason=fallback_reason)
    action = _team_decision_to_action(team_decision)
    artifact_payload = {
        "artifact_id": artifact_id,
        "team": team,
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
    artifact_paths = {"image_path": str(frame_path)}
    if artifact_dir is not None:
        artifact_paths = _save_artifact(artifact_dir, f"{artifact_id}_{team}", image, artifact_payload)
    elif frame_path.exists():
        frame_path.unlink(missing_ok=True)

    return {
        "raw_decision": raw_decision,
        "decision": team_decision,
        "action": action,
        "artifact_paths": artifact_paths,
    }


def run_matches(
    episodes: int,
    max_steps: int,
    seed: int,
    num_home: int,
    num_away: int,
    vlm_model: str,
    vlm_base_url: str,
    save_dir: Path,
    fps: int,
    query_interval: int,
) -> Dict[str, Any]:
    vlm = _build_vlm(vlm_model, vlm_base_url)
    save_dir.mkdir(parents=True, exist_ok=True)

    episode_summaries: List[Dict[str, Any]] = []
    for episode_idx in range(int(episodes)):
        controller = build_match_controller(
            num_home=num_home,
            num_away=num_away,
            seed=seed + episode_idx,
            max_steps=max_steps,
            end_on_ball_out=True,
        )
        simulation = controller.simulation
        state = simulation.reset()
        recent_events: List[Dict[str, Any]] = []
        reward = 0.0
        done = False
        home_action = None
        away_action = None
        records: List[Dict[str, Any]] = [{
            "obs": state,
            "reward": 0.0,
            "done": False,
            "action": None,
            "away_action": None,
            "info": {
                "events": [],
                "state": simulation.get_state(),
                "actions": {},
            },
        }]
        home_fallback_steps = 0
        away_fallback_steps = 0
        step_idx = 0
        cached_home_action: Optional[Dict[str, Any]] = None
        cached_away_action: Optional[Dict[str, Any]] = None
        cached_home_decision: Optional[TeamVLMDecision] = None
        cached_away_decision: Optional[TeamVLMDecision] = None
        home_last_query_step: Optional[int] = None
        away_last_query_step: Optional[int] = None
        previous_owner_id: Optional[str] = None

        while not done and step_idx < max_steps:
            state = simulation.get_state()
            record = _make_render_record(
                state=state,
                reward=reward,
                done=done,
                action=home_action,
                away_action=away_action,
                events=recent_events,
            )
            step_artifact_dir = save_dir / f"artifacts_ep_{episode_idx:03d}"
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
            if should_query_home:
                home_query = _query_team_vlm(
                    vlm=vlm,
                    team="home",
                    artifact_id=f"ep{episode_idx:03d}_step{step_idx:03d}",
                    record=record,
                    state_text=_state_text_summary(state, recent_events, team="home"),
                    artifact_dir=step_artifact_dir,
                )
                cached_home_action = home_query["action"]
                cached_home_decision = home_query["decision"]
                home_last_query_step = step_idx
                if any(decision.source != "vlm" for decision in home_query["decision"].players.values()):
                    home_fallback_steps += 1
            if should_query_away:
                away_query = _query_team_vlm(
                    vlm=vlm,
                    team="away",
                    artifact_id=f"ep{episode_idx:03d}_step{step_idx:03d}",
                    record=record,
                    state_text=_state_text_summary(state, recent_events, team="away"),
                    artifact_dir=step_artifact_dir,
                )
                cached_away_action = away_query["action"]
                cached_away_decision = away_query["decision"]
                away_last_query_step = step_idx
                if any(decision.source != "vlm" for decision in away_query["decision"].players.values()):
                    away_fallback_steps += 1

            home_action = cached_home_action
            away_action = cached_away_action

            obs, reward, done, info = simulation.step(home_action, opponent_action=away_action)
            records.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "action": info["actions"]["home"],
                "away_action": info["actions"]["away"],
                "info": info,
            })
            recent_events = info.get("events", [])
            previous_owner_id = state.get("ball_owner_id")
            step_idx += 1

        winner = records[-1]["info"]["state"].get("winner")
        summary = {
            "episode": episode_idx,
            "seed": seed + episode_idx,
            "steps": step_idx,
            "winner": winner,
            "home_fallback_steps": home_fallback_steps,
            "away_fallback_steps": away_fallback_steps,
            "ended_by_goal": winner in {"home", "away"},
        }
        video_path = save_dir / f"vlm_vs_vlm_{num_home}v{num_away}_ep{episode_idx:03d}.mp4"
        render_episode_mp4(
            episode=records,
            output_path=video_path,
            field_size=simulation.params.field_size,
            fps=fps,
        )
        summary["video_path"] = str(video_path)
        episode_summaries.append(summary)
        print(
            f"[vlm-vs-vlm] {num_home}v{num_away} ep={episode_idx} seed={seed + episode_idx} "
            f"steps={step_idx} winner={winner} home_fb={home_fallback_steps} away_fb={away_fallback_steps}"
        )

    return {
        "episodes": int(episodes),
        "num_home": int(num_home),
        "num_away": int(num_away),
        "vlm_model": vlm_model,
        "vlm_base_url": vlm_base_url,
        "query_interval": int(query_interval),
        "episode_summaries": episode_summaries,
        "save_dir": str(save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperGym VLM-vs-VLM matches and record videos.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--num-home", type=int, default=1)
    parser.add_argument("--num-away", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--query-interval", type=int, default=5, help="最大查询间隔；动作完成、球权变化或关键事件会提前触发重问。")
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    result = run_matches(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        num_home=args.num_home,
        num_away=args.num_away,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        save_dir=Path(args.save_dir),
        fps=args.fps,
        query_interval=args.query_interval,
    )
    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as file_obj:
            json.dump(_to_plain(result), file_obj, ensure_ascii=False, indent=2)
        print(f"saved_result={output}")


if __name__ == "__main__":
    main()
