from __future__ import annotations

import argparse
import base64
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.renderer import render_record


ALLOWED_POLICY_IDS = {
    "move_to_target",
    "pass_to_target",
    "trap_ball",
    "dribble_to_target",
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


@dataclass
class BenchmarkCase:
    case_id: str
    title: str
    description: str
    num_home: int
    num_away: int
    expected_policy_ids: List[str]
    expected_target_note: str
    state_spec: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    expected_player_id: str = "home_0"
    prior_action: Optional[Dict[str, Any]] = None
    prior_away_action: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    done: bool = False


def _clip_target(target: np.ndarray, field_size: np.ndarray) -> np.ndarray:
    clipped = np.asarray(target, dtype=np.float32).copy()
    clipped[0] = float(np.clip(clipped[0], 0.0, field_size[0]))
    clipped[1] = float(np.clip(clipped[1], 0.0, field_size[1]))
    return clipped


def _to_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _state_text_summary(state: Dict[str, Any], recent_events: List[Dict[str, Any]]) -> str:
    ball = np.asarray(state["ball_position"], dtype=np.float32)
    ball_vel = np.asarray(state.get("ball_velocity", [0.0, 0.0]), dtype=np.float32)
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    owner = state.get("ball_owner_id")
    home_players = [p for p in state.get("players", []) if p.get("team") == "home"]
    away_players = [p for p in state.get("players", []) if p.get("team") == "away"]

    nearest_home_dist = 999.0
    nearest_home_id = "none"
    for player in home_players:
        player_pos = np.asarray(player["position"], dtype=np.float32)
        dist = float(np.linalg.norm(player_pos - ball))
        if dist < nearest_home_dist:
            nearest_home_dist = dist
            nearest_home_id = player["player_id"]

    nearest_away_dist = 999.0
    nearest_away_id = "none"
    for player in away_players:
        player_pos = np.asarray(player["position"], dtype=np.float32)
        dist = float(np.linalg.norm(player_pos - ball))
        if dist < nearest_away_dist:
            nearest_away_dist = dist
            nearest_away_id = player["player_id"]

    opponent_goal_center = np.asarray([field[0], 0.5 * field[1]], dtype=np.float32)
    ball_to_opponent_goal = float(np.linalg.norm(opponent_goal_center - ball))
    ball_speed = float(np.linalg.norm(ball_vel))

    event_tokens: List[str] = []
    for event in recent_events[-3:]:
        name = str(event.get("event_type", "unknown"))
        if "team" in event:
            name += f":{event['team']}"
        event_tokens.append(name)
    event_text = ", ".join(event_tokens) if event_tokens else "none"
    home_layout = ", ".join(
        f"{player['player_id']}=({float(player['position'][0]):.2f},{float(player['position'][1]):.2f})"
        for player in home_players
    )
    away_layout = ", ".join(
        f"{player['player_id']}=({float(player['position'][0]):.2f},{float(player['position'][1]):.2f})"
        for player in away_players
    )

    return (
        f"step={int(state.get('step', 0))}; "
        f"ball=({ball[0]:.2f},{ball[1]:.2f}); "
        f"ball_v=({ball_vel[0]:.2f},{ball_vel[1]:.2f}); speed={ball_speed:.2f}; "
        f"owner={owner or 'free'}; "
        f"nearest_home={nearest_home_id}@{nearest_home_dist:.2f}; "
        f"nearest_away={nearest_away_id}@{nearest_away_dist:.2f}; "
        f"ball_to_opponent_goal={ball_to_opponent_goal:.2f}; "
        f"home_players=[{home_layout}]; "
        f"away_players=[{away_layout}]; "
        f"recent_events=[{event_text}]"
    )


def _make_render_record(
    state: Dict[str, Any],
    reward: float,
    done: bool,
    action: Optional[Dict[str, Any]],
    away_action: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "obs": None,
        "reward": float(reward),
        "done": bool(done),
        "action": action,
        "away_action": away_action,
        "info": {
            "events": list(events),
            "state": state,
        },
    }


class OpenAICompatibleVisionVLM:
    """OpenAI-compatible vision client with a constrained tactic prompt."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def decide(self, state: Dict[str, Any], state_text: str, image_path: Path) -> Dict[str, Any]:
        field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
        home_players = [player for player in state.get("players", []) if player.get("team") == "home"]
        player_schema = ", ".join(
            f'"{player["player_id"]}": {{"policy_id": "<move_to_target|pass_to_target|trap_ball|dribble_to_target>", "target": [x, y], "reason": "<short_reason>"}}'
            for player in home_players
        )
        home_player_ids = ", ".join(player["player_id"] for player in home_players)
        with open(image_path, "rb") as file_obj:
            image_b64 = base64.b64encode(file_obj.read()).decode("utf-8")

        schema_text = (
            "Return ONLY valid JSON with this shape: "
            '{"players": {' + player_schema + "}}"
        )
        policy_semantics = (
            "Policy semantics:\n"
            "- move_to_target: use when home should run to space, close down a loose ball, or reposition.\n"
            "- trap_ball: use when the ball is free or moving and home should first secure control near the ball.\n"
            "- pass_to_target: use only when a home player can plausibly play the ball now and the target is a useful forward or lateral destination, not the current ball position.\n"
            "- dribble_to_target: use only when home already controls the ball and should carry it into better space.\n"
            "Hard constraints:\n"
            "- If owner starts with 'home_', do not output trap_ball because home already controls the ball.\n"
            "- If owner is free, prefer move_to_target or trap_ball over pass_to_target.\n"
            "- Avoid meaningless pass_to_target to the current ball location or to a point behind the attack."
        )
        prompt = (
            "You are a high-level soccer tactics model for the home team.\n"
            "The image is the primary input. The text is auxiliary context.\n"
            f"You must output one action for every home player: {home_player_ids}.\n"
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
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 220,
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


def _fallback_decision_for_player(player_id: str, state: Dict[str, Any], fallback_reason: str) -> VLMDecision:
    field = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
    ball = np.asarray(state["ball_position"], dtype=np.float32)
    goal = np.asarray([field[0], 0.5 * field[1]], dtype=np.float32)
    home_players = [player for player in state.get("players", []) if player.get("team") == "home"]
    player = next((item for item in home_players if item["player_id"] == player_id), None)
    owner_id = state.get("ball_owner_id")
    if owner_id == player_id and player is not None:
        pos = np.asarray(player["position"], dtype=np.float32)
        forward = pos + np.asarray([0.8, 0.0], dtype=np.float32)
        return VLMDecision(
            policy_id="dribble_to_target",
            target=_clip_target(0.7 * forward + 0.3 * goal, field),
            source="fallback",
            reason=fallback_reason,
        )
    if owner_id is None:
        nearest = None
        nearest_dist = 999.0
        for home_player in home_players:
            ppos = np.asarray(home_player["position"], dtype=np.float32)
            dist = float(np.linalg.norm(ppos - ball))
            if dist < nearest_dist:
                nearest = home_player
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


def _parse_single_decision(raw: Dict[str, Any], state: Dict[str, Any], player_id: str, fallback_reason: str) -> VLMDecision:
    default = _fallback_decision_for_player(player_id, state, fallback_reason)
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

    if policy_id == "pass_to_target":
        ball_delta = float(np.linalg.norm(parsed_target - ball))
        if state.get("ball_owner_id") is None and ball_delta < 0.6:
            return default
        if ball_delta < 0.12:
            return default

    return VLMDecision(policy_id=policy_id, target=parsed_target, source="vlm", reason=reason)


def _parse_team_decision(raw: Dict[str, Any], state: Dict[str, Any], fallback_reason: str) -> TeamVLMDecision:
    home_players = [player for player in state.get("players", []) if player.get("team") == "home"]
    raw_players = raw.get("players", raw if isinstance(raw, dict) else {})
    parsed: Dict[str, VLMDecision] = {}
    for player in home_players:
        player_id = player["player_id"]
        player_raw = raw_players.get(player_id, {}) if isinstance(raw_players, dict) else {}
        parsed[player_id] = _parse_single_decision(player_raw, state=state, player_id=player_id, fallback_reason=fallback_reason)
    return TeamVLMDecision(players=parsed)


def _decision_to_action(decision: VLMDecision) -> Dict[str, Any]:
    mapping = {
        "move_to_target": "move",
        "pass_to_target": "pass",
        "trap_ball": "trap",
        "dribble_to_target": "dribble",
    }
    skill = mapping.get(decision.policy_id, "move")
    return {
        "skill": skill,
        "target": decision.target.copy(),
    }


def _team_decision_to_action(team_decision: TeamVLMDecision) -> Dict[str, Any]:
    return {
        player_id: _decision_to_action(decision)
        for player_id, decision in team_decision.players.items()
    }


def _build_vlm(model: str, base_url: str) -> Any:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY")
    return OpenAICompatibleVisionVLM(model=model, api_key=api_key, base_url=base_url)


def _save_artifact(
    artifact_dir: Path,
    artifact_id: str,
    image,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    image_path = artifact_dir / f"{artifact_id}.png"
    json_path = artifact_dir / f"{artifact_id}.json"
    image.save(image_path)
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_plain(payload), file_obj, ensure_ascii=False, indent=2)
    return {
        "image_path": str(image_path),
        "json_path": str(json_path),
    }


def _query_vlm_on_record(
    vlm: OpenAICompatibleVisionVLM,
    artifact_id: str,
    record: Dict[str, Any],
    state_text: str,
    artifact_dir: Optional[Path],
    keep_temp_frame: bool = False,
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
        frame_path = artifact_dir / f"{artifact_id}.png"
        image.save(frame_path)

    raw_decision: Dict[str, Any]
    fallback_reason = "schema_invalid"
    try:
        raw_decision = vlm.decide(state=state, state_text=state_text, image_path=frame_path)
    except Exception as exc:
        raw_decision = {"error": str(exc)}
        fallback_reason = f"vlm_error:{exc}"

    team_decision = _parse_team_decision(raw_decision if isinstance(raw_decision, dict) else {}, state=state, fallback_reason=fallback_reason)
    action = _team_decision_to_action(team_decision)
    artifact_paths = {"image_path": str(frame_path)}

    artifact_payload = {
        "artifact_id": artifact_id,
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
        artifact_paths = _save_artifact(artifact_dir, artifact_id, image, artifact_payload)
    elif not keep_temp_frame and frame_path.exists():
        frame_path.unlink(missing_ok=True)

    return {
        "raw_decision": raw_decision,
        "decision": team_decision,
        "action": action,
        "artifact_paths": artifact_paths,
    }


def _set_manual_state(simulation, state_spec: Dict[str, Any]) -> Dict[str, Any]:
    simulation.reset()
    simulation.step_count = int(state_spec.get("step", 0))
    simulation.winner = state_spec.get("winner")
    simulation.prev_actions = {}
    simulation.last_actions = {}

    player_specs = state_spec.get("players", [])
    if len(player_specs) != len(simulation.players):
        raise ValueError(f"state_spec player count {len(player_specs)} does not match simulation player count {len(simulation.players)}")

    for player, player_spec in zip(simulation.players, player_specs):
        player.position = np.asarray(player_spec["position"], dtype=np.float32).copy()
        player.velocity = np.asarray(player_spec.get("velocity", [0.0, 0.0]), dtype=np.float32).copy()
        player.has_ball = bool(player_spec.get("has_ball", False))

    simulation.ball.position = np.asarray(state_spec["ball_position"], dtype=np.float32).copy()
    simulation.ball.velocity = np.asarray(state_spec.get("ball_velocity", [0.0, 0.0]), dtype=np.float32).copy()
    simulation.ball.owner_id = state_spec.get("ball_owner_id")

    owner_id = simulation.ball.owner_id
    if owner_id is not None:
        for player in simulation.players:
            player.has_ball = player.player_id == owner_id
            if player.player_id == owner_id:
                simulation.ball.position = player.position.copy()
                simulation.ball.velocity[:] = 0.0
    return simulation.get_state()


def _default_benchmark_cases() -> List[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="opening_loose_ball",
            title="Opening Loose Ball",
            description="开局自由球在 home_0 脚前，合理动作应先拿球或停球。",
            num_home=1,
            num_away=1,
            expected_policy_ids=["move_to_target", "trap_ball"],
            expected_target_note="Target should stay near the loose ball, not behind home_0.",
            state_spec={
                "step": 0,
                "ball_position": [1.35, 2.10],
                "ball_velocity": [0.0, 0.0],
                "ball_owner_id": None,
                "players": [
                    {"position": [1.0, 2.10], "velocity": [0.0, 0.0], "has_ball": False},
                    {"position": [6.0, 1.8], "velocity": [0.0, 0.0], "has_ball": False},
                ],
            },
            recent_events=[],
        ),
        BenchmarkCase(
            case_id="rolling_ball_trap",
            title="Rolling Ball Trap",
            description="自由球缓慢滚向 home_0，合理动作应优先停球而不是传球。",
            num_home=1,
            num_away=1,
            expected_policy_ids=["trap_ball", "move_to_target"],
            expected_target_note="Target should be near the current ball path to secure it.",
            state_spec={
                "step": 6,
                "ball_position": [2.20, 2.20],
                "ball_velocity": [0.18, 0.02],
                "ball_owner_id": None,
                "players": [
                    {"position": [1.60, 2.00], "velocity": [0.05, 0.02], "has_ball": False},
                    {"position": [4.80, 2.40], "velocity": [-0.04, 0.00], "has_ball": False},
                ],
            },
            recent_events=[
                {"event_type": "move", "team": "home"},
                {"event_type": "move", "team": "away"},
            ],
        ),
        BenchmarkCase(
            case_id="away_pressing_loose_ball",
            title="Away Pressing Loose Ball",
            description="客队更接近自由球，合理动作应向球或拦截点移动，而不是仓促传球。",
            num_home=1,
            num_away=1,
            expected_policy_ids=["move_to_target", "trap_ball"],
            expected_target_note="Target should close the loose ball or intercept lane before away_0 reaches it.",
            state_spec={
                "step": 12,
                "ball_position": [4.50, 3.10],
                "ball_velocity": [0.0, 0.0],
                "ball_owner_id": None,
                "players": [
                    {"position": [3.60, 2.50], "velocity": [0.02, 0.03], "has_ball": False},
                    {"position": [4.85, 3.05], "velocity": [-0.03, 0.01], "has_ball": False},
                ],
            },
            recent_events=[
                {"event_type": "ball_player_collision", "player": "away_0"},
            ],
        ),
        BenchmarkCase(
            case_id="home_midfield_possession",
            title="Home Midfield Possession",
            description="home_0 在中场稳控球，合理动作是向前推进或带球进入空间。",
            num_home=1,
            num_away=1,
            expected_policy_ids=["move_to_target", "dribble_to_target", "pass_to_target"],
            expected_target_note="Target should advance toward the opponent goal or open space, not back to the current ball position.",
            state_spec={
                "step": 20,
                "ball_position": [5.10, 3.00],
                "ball_velocity": [0.0, 0.0],
                "ball_owner_id": "home_0",
                "players": [
                    {"position": [5.10, 3.00], "velocity": [0.0, 0.0], "has_ball": True},
                    {"position": [6.30, 2.70], "velocity": [-0.05, 0.02], "has_ball": False},
                ],
            },
            recent_events=[
                {"event_type": "trap_completed", "team": "home", "by": "home_0"},
            ],
        ),
        BenchmarkCase(
            case_id="home_near_goal",
            title="Home Near Goal",
            description="home_0 已经带球逼近右侧球门，合理动作应继续向危险区域推进或直接把球送向门前。",
            num_home=1,
            num_away=1,
            expected_policy_ids=["pass_to_target", "move_to_target", "dribble_to_target"],
            expected_target_note="Target should stay near the opponent goal mouth or a nearby attacking lane, not retreat.",
            state_spec={
                "step": 26,
                "ball_position": [8.60, 3.00],
                "ball_velocity": [0.0, 0.0],
                "ball_owner_id": "home_0",
                "players": [
                    {"position": [8.60, 3.00], "velocity": [0.0, 0.0], "has_ball": True},
                    {"position": [8.90, 2.20], "velocity": [0.0, 0.0], "has_ball": False},
                ],
            },
            recent_events=[
                {"event_type": "move", "team": "home", "by": "home_0"},
            ],
        ),
        BenchmarkCase(
            case_id="two_v_two_switch",
            title="2v2 Possession Switch",
            description="2v2 中 home_0 控球，home_1 在右前方空位。合理动作通常是推进或把球送向空位。",
            num_home=2,
            num_away=2,
            expected_policy_ids=["pass_to_target", "move_to_target", "dribble_to_target"],
            expected_target_note="Target should bias toward home_1 or the right attacking half-space.",
            state_spec={
                "step": 32,
                "ball_position": [5.80, 2.70],
                "ball_velocity": [0.0, 0.0],
                "ball_owner_id": "home_0",
                "players": [
                    {"position": [5.80, 2.70], "velocity": [0.0, 0.0], "has_ball": True},
                    {"position": [7.40, 4.10], "velocity": [0.02, -0.01], "has_ball": False},
                    {"position": [6.60, 2.80], "velocity": [-0.03, 0.00], "has_ball": False},
                    {"position": [7.00, 1.90], "velocity": [-0.02, 0.01], "has_ball": False},
                ],
            },
            recent_events=[
                {"event_type": "turnover", "from_player": "away_0", "to_player": "home_0"},
            ],
        ),
    ]


def _get_benchmark_case(case_id: str) -> BenchmarkCase:
    for case in _default_benchmark_cases():
        if case.case_id == case_id:
            return case
    available = ", ".join(case.case_id for case in _default_benchmark_cases())
    raise ValueError(f"Unknown start_case_id={case_id}. Available cases: {available}")


def run_rollout(
    episodes: int,
    max_steps: int,
    seed: int,
    vlm_model: str,
    vlm_base_url: str,
    artifact_dir: Optional[Path],
    num_home: int,
    num_away: int,
    start_case_id: str,
) -> Dict[str, Any]:
    start_case = _get_benchmark_case(start_case_id) if start_case_id else None
    rollout_num_home = start_case.num_home if start_case is not None else num_home
    rollout_num_away = start_case.num_away if start_case is not None else num_away
    env_max_steps = max_steps if start_case is None else int(start_case.state_spec.get("step", 0)) + max_steps
    controller = build_match_controller(
        num_home=rollout_num_home,
        num_away=rollout_num_away,
        seed=seed,
        max_steps=env_max_steps,
    )
    simulation = controller.simulation
    vlm = _build_vlm(vlm_model, vlm_base_url)

    success_count = 0
    episode_summaries: List[Dict[str, Any]] = []

    for episode_idx in range(int(episodes)):
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
            info = {
                "events": list(recent_events),
                "state": state,
                "actions": {},
            }
        used_fallback_steps = 0
        step_idx = 0

        while not done and step_idx < max_steps:
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
            )

            team_decision = query["decision"]
            action = query["action"]
            if any(decision.source != "vlm" for decision in team_decision.players.values()):
                used_fallback_steps += 1

            obs, reward, done, info = simulation.step(action)
            del obs
            away_action = info.get("actions", {}).get("away")
            recent_events = info.get("events", [])

            if step_artifact_dir is not None:
                step_json_path = step_artifact_dir / f"ep{episode_idx:03d}_step{step_idx:03d}.json"
                if step_json_path.exists():
                    with open(step_json_path, "r", encoding="utf-8") as file_obj:
                        artifact_payload = json.load(file_obj)
                    artifact_payload["post_step"] = _to_plain({
                        "reward": reward,
                        "done": done,
                        "events": recent_events,
                        "away_action": away_action,
                        "state": info.get("state", {}),
                    })
                    with open(step_json_path, "w", encoding="utf-8") as file_obj:
                        json.dump(artifact_payload, file_obj, ensure_ascii=False, indent=2)

            step_idx += 1

        winner = info.get("state", {}).get("winner")
        success = winner == "home"
        if success:
            success_count += 1
        summary = {
            "episode": episode_idx,
            "steps": step_idx,
            "winner": winner,
            "success": success,
            "fallback_steps": used_fallback_steps,
        }
        episode_summaries.append(summary)
        print(
            f"[PoC rollout] episode={episode_idx} steps={step_idx} winner={winner} "
            f"success={int(success)} fallback_steps={used_fallback_steps}"
        )

    success_rate = float(success_count) / max(1, int(episodes))
    result = {
        "mode": "rollout",
        "episodes": int(episodes),
        "successes": int(success_count),
        "success_rate": success_rate,
        "episode_summaries": episode_summaries,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        "start_case_id": start_case_id,
    }
    print("\n=== VLM PoC Rollout Result ===")
    print(f"episodes={result['episodes']}")
    print(f"successes={result['successes']}")
    print(f"success_rate={result['success_rate']:.4f}")
    return result


def run_benchmark(
    vlm_model: str,
    vlm_base_url: str,
    artifact_dir: Optional[Path],
) -> Dict[str, Any]:
    vlm = _build_vlm(vlm_model, vlm_base_url)
    case_results: List[Dict[str, Any]] = []
    matched_count = 0

    for case in _default_benchmark_cases():
        controller = build_match_controller(
            num_home=case.num_home,
            num_away=case.num_away,
            seed=0,
            max_steps=max(100, int(case.state_spec.get("step", 0)) + 1),
        )
        state = _set_manual_state(controller.simulation, case.state_spec)
        state_text = _state_text_summary(state, case.recent_events)
        record = _make_render_record(
            state=state,
            reward=case.reward,
            done=case.done,
            action=case.prior_action,
            away_action=case.prior_away_action,
            events=case.recent_events,
        )
        query = _query_vlm_on_record(
            vlm=vlm,
            artifact_id=case.case_id,
            record=record,
            state_text=state_text,
            artifact_dir=artifact_dir,
        )
        team_decision = query["decision"]
        decision = team_decision.players[case.expected_player_id]
        target = np.asarray(decision.target, dtype=np.float32)
        policy_match = decision.policy_id in case.expected_policy_ids
        if policy_match:
            matched_count += 1
        all_players_present = len(team_decision.players) == case.num_home and all(
            player_id.startswith("home_") for player_id in team_decision.players
        )

        summary = {
            "case_id": case.case_id,
            "title": case.title,
            "description": case.description,
            "expected_player_id": case.expected_player_id,
            "expected_policy_ids": case.expected_policy_ids,
            "expected_target_note": case.expected_target_note,
            "state_text": state_text,
            "raw_decision": query["raw_decision"],
            "parsed_decision": {
                player_id: {
                    "policy_id": player_decision.policy_id,
                    "target": [float(player_decision.target[0]), float(player_decision.target[1])],
                    "source": player_decision.source,
                    "reason": player_decision.reason,
                }
                for player_id, player_decision in team_decision.players.items()
            },
            "primary_player_decision": {
                "policy_id": decision.policy_id,
                "target": [float(target[0]), float(target[1])],
                "source": decision.source,
                "reason": decision.reason,
            },
            "policy_match": policy_match,
            "all_players_present": all_players_present,
            "artifact_paths": query["artifact_paths"],
        }
        case_results.append(summary)
        print(
            f"[PoC benchmark] case={case.case_id} player={case.expected_player_id} policy={decision.policy_id} "
            f"source={decision.source} match={int(policy_match)}"
        )

    match_rate = float(matched_count) / max(1, len(case_results))
    result = {
        "mode": "benchmark",
        "cases": len(case_results),
        "policy_matches": matched_count,
        "policy_match_rate": match_rate,
        "case_results": case_results,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
    }
    print("\n=== VLM PoC Benchmark Result ===")
    print(f"cases={result['cases']}")
    print(f"policy_matches={result['policy_matches']}")
    print(f"policy_match_rate={result['policy_match_rate']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="隔离式 VLM 高层策略 PoC（图像输入 -> policy_id+target 输出）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rollout",
        choices=["rollout", "benchmark"],
        help="rollout: 多步对局；benchmark: 固定状态单步战术评测",
    )
    parser.add_argument("--episodes", type=int, default=20, help="rollout 模式的评估回合数")
    parser.add_argument("--max-steps", type=int, default=300, help="rollout 模式的每回合最大步数")
    parser.add_argument("--seed", type=int, default=7, help="rollout 模式的随机种子")
    parser.add_argument("--num-home", type=int, default=1, help="rollout 模式主队球员数")
    parser.add_argument("--num-away", type=int, default=1, help="rollout 模式客队球员数")
    parser.add_argument(
        "--start-case-id",
        type=str,
        default="",
        help="可选：rollout 直接从某个 benchmark case 状态开始，例如 home_midfield_possession",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen3vl",
        help="OpenAI 兼容视觉模型名",
    )
    parser.add_argument(
        "--vlm-base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI 兼容 API 基地址，例如 https://xxx/v1",
    )
    parser.add_argument(
        "--save-artifacts-dir",
        type=str,
        default="",
        help="可选：保存每步或每个 benchmark case 的图像和 JSON artifact",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="可选：将最终统计结果保存到 JSON 文件",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.save_artifacts_dir) if args.save_artifacts_dir else None

    if args.mode == "rollout":
        result = run_rollout(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            artifact_dir=artifact_dir,
            num_home=args.num_home,
            num_away=args.num_away,
            start_case_id=args.start_case_id,
        )
    else:
        result = run_benchmark(
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            artifact_dir=artifact_dir,
        )

    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as file_obj:
            json.dump(_to_plain(result), file_obj, ensure_ascii=False, indent=2)
        print(f"saved_result={output}")


if __name__ == "__main__":
    main()
