from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import build_match_controller
from envs.hyperGym.policies import SimpleMatchPolicy
from scripts.tmp.vlm_policy_poc import _get_vlm_api_key, _make_render_record, _set_manual_state, _to_plain
from scripts.tmp.vlm_vs_vlm import (
    DEFAULT_VLM_BASE_URL,
    DEFAULT_VLM_MODEL,
    OpenAICompatibleVisionVLM,
    _query_team_vlm,
    _state_text_summary,
)


FIELD_W = 10.0
FIELD_H = 6.0
FIELD_DIAG = float(math.sqrt(FIELD_W ** 2 + FIELD_H ** 2))
TEAM_IDS = ("home", "away")
PLAYER_IDS = {
    "home": ("home_0", "home_1"),
    "away": ("away_0", "away_1"),
}


@dataclass
class ScenarioSpec:
    family_id: str
    episode_seed: int
    state_spec: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    description: str
    tags: List[str]


def _clip_xy(x: float, y: float) -> np.ndarray:
    return np.asarray([
        float(np.clip(x, 0.35, FIELD_W - 0.35)),
        float(np.clip(y, 0.35, FIELD_H - 0.35)),
    ], dtype=np.float32)


def _team_sign(team: str) -> float:
    return 1.0 if team == "home" else -1.0


def _goal_center(team: str) -> np.ndarray:
    return np.asarray([FIELD_W, FIELD_H * 0.5], dtype=np.float32) if team == "home" else np.asarray([0.0, FIELD_H * 0.5], dtype=np.float32)


def _attack_point(team: str, dist_to_goal: float, lateral: float = 0.0) -> np.ndarray:
    goal = _goal_center(team)
    x = goal[0] - _team_sign(team) * dist_to_goal
    y = FIELD_H * 0.5 + lateral
    return _clip_xy(float(x), float(y))


def _support_point(anchor: np.ndarray, team: str, forward: float, lateral: float) -> np.ndarray:
    return _clip_xy(float(anchor[0] + _team_sign(team) * forward), float(anchor[1] + lateral))


def _player_spec(position: np.ndarray, *, velocity: Tuple[float, float] = (0.0, 0.0), has_ball: bool = False) -> Dict[str, Any]:
    return {
        "position": [float(position[0]), float(position[1])],
        "velocity": [float(velocity[0]), float(velocity[1])],
        "has_ball": bool(has_ball),
    }


def _arrange_players(home_0: np.ndarray, home_1: np.ndarray, away_0: np.ndarray, away_1: np.ndarray, *, owner_id: str | None = None) -> List[Dict[str, Any]]:
    return [
        _player_spec(home_0, has_ball=(owner_id == "home_0")),
        _player_spec(home_1, has_ball=(owner_id == "home_1")),
        _player_spec(away_0, has_ball=(owner_id == "away_0")),
        _player_spec(away_1, has_ball=(owner_id == "away_1")),
    ]


def _random_team(rng: np.random.Generator) -> str:
    return "home" if float(rng.random()) < 0.5 else "away"


def _scenario_controlled_buildup(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    defend_team = "away" if owner_team == "home" else "home"
    owner_id = f"{owner_team}_0"
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(4.2, 5.4)), lateral=float(rng.uniform(-0.5, 0.5)))
    support = _support_point(anchor, owner_team, forward=float(rng.uniform(1.0, 1.6)), lateral=float(rng.uniform(0.8, 1.3)) * (1.0 if rng.random() < 0.5 else -1.0))
    press = _support_point(anchor, defend_team, forward=float(rng.uniform(0.7, 1.2)), lateral=float(rng.uniform(-0.4, 0.4)))
    cover = _support_point(anchor, defend_team, forward=float(rng.uniform(1.3, 1.9)), lateral=float(rng.uniform(0.9, 1.4)) * (1.0 if rng.random() < 0.5 else -1.0))
    players = _arrange_players(anchor, support, press, cover, owner_id=owner_id) if owner_team == "home" else _arrange_players(press, cover, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="controlled_buildup",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(10, 35)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "trap_completed", "team": owner_team, "by": owner_id}],
        description="One team already controls the ball in midfield with a visible support option.",
        tags=["possession", owner_team, "midfield"],
    )


def _scenario_near_goal_finishing(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    defend_team = "away" if owner_team == "home" else "home"
    owner_id = f"{owner_team}_0"
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(1.0, 1.8)), lateral=float(rng.uniform(-0.55, 0.55)))
    support = _support_point(anchor, owner_team, forward=float(rng.uniform(0.15, 0.45)), lateral=float(rng.uniform(0.7, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0))
    press = _support_point(anchor, defend_team, forward=float(rng.uniform(0.2, 0.6)), lateral=float(rng.uniform(0.15, 0.45)) * (1.0 if rng.random() < 0.5 else -1.0))
    cover = _support_point(anchor, defend_team, forward=float(rng.uniform(0.8, 1.3)), lateral=float(rng.uniform(0.8, 1.2)) * (1.0 if rng.random() < 0.5 else -1.0))
    players = _arrange_players(anchor, support, press, cover, owner_id=owner_id) if owner_team == "home" else _arrange_players(press, cover, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="near_goal_finishing",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(20, 45)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "ball_control_gained", "team": owner_team, "by": owner_id}],
        description="The owner is already near goal and should treat pass as the kick/shoot action.",
        tags=["finishing", owner_team, "near_goal"],
    )


def _scenario_sideline_danger(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    owner_id = f"{owner_team}_0"
    lateral_sign = 1.0 if rng.random() < 0.5 else -1.0
    sideline_y = 0.55 if lateral_sign < 0 else FIELD_H - 0.55
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(2.8, 4.8)), lateral=sideline_y - FIELD_H * 0.5)
    support = _clip_xy(float(anchor[0] - _team_sign(owner_team) * rng.uniform(0.4, 1.0)), float(anchor[1] - lateral_sign * rng.uniform(0.8, 1.3)))
    press = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(0.1, 0.6)), float(anchor[1] - lateral_sign * rng.uniform(0.15, 0.4)))
    cover = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(0.5, 1.2)), float(anchor[1] - lateral_sign * rng.uniform(0.9, 1.5)))
    players = _arrange_players(anchor, support, press, cover, owner_id=owner_id) if owner_team == "home" else _arrange_players(press, cover, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="sideline_danger",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(15, 40)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "move_touch", "team": owner_team, "by": owner_id}],
        description="Ball is controlled near the touchline and should be recycled inward instead of forced down the line.",
        tags=["sideline", owner_team],
    )


def _scenario_pressured_possession(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    defend_team = "away" if owner_team == "home" else "home"
    owner_id = f"{owner_team}_0"
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(3.0, 4.8)), lateral=float(rng.uniform(-0.4, 0.4)))
    support = _support_point(anchor, owner_team, forward=float(rng.uniform(0.8, 1.3)), lateral=float(rng.uniform(0.6, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0))
    press = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(0.25, 0.55)), float(anchor[1] + rng.uniform(-0.2, 0.2)))
    cover = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(1.0, 1.5)), float(anchor[1] + rng.uniform(-0.8, 0.8)))
    players = _arrange_players(anchor, support, press, cover, owner_id=owner_id) if owner_team == "home" else _arrange_players(press, cover, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="pressured_possession",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(18, 42)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "trap_completed", "team": owner_team, "by": owner_id}],
        description="Ball carrier is under immediate pressure and should release or protect the ball intelligently.",
        tags=["pressure", owner_team],
    )


def _scenario_loose_ball_scramble(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    center = _clip_xy(float(rng.uniform(3.0, 7.0)), float(rng.uniform(1.4, 4.6)))
    home_0 = _clip_xy(float(center[0] - rng.uniform(0.4, 0.8)), float(center[1] + rng.uniform(-0.3, 0.3)))
    home_1 = _clip_xy(float(center[0] - rng.uniform(1.0, 1.8)), float(center[1] + rng.uniform(0.8, 1.4) * (1.0 if rng.random() < 0.5 else -1.0)))
    away_0 = _clip_xy(float(center[0] + rng.uniform(0.3, 0.7)), float(center[1] + rng.uniform(-0.35, 0.35)))
    away_1 = _clip_xy(float(center[0] + rng.uniform(1.0, 1.8)), float(center[1] + rng.uniform(0.8, 1.4) * (1.0 if rng.random() < 0.5 else -1.0)))
    ball_vel = [float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.05, 0.05))]
    return ScenarioSpec(
        family_id="loose_ball_scramble",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(10, 35)), "ball_position": [float(center[0]), float(center[1])], "ball_velocity": ball_vel, "ball_owner_id": None, "players": _arrange_players(home_0, home_1, away_0, away_1)},
        recent_events=[{"event_type": "loose_ball_scramble"}],
        description="A contested free ball should trigger one secure-trap role and one support role.",
        tags=["free_ball", "scramble"],
    )


def _scenario_second_ball_continuation(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    passer_team = _random_team(rng)
    center = _attack_point(passer_team, dist_to_goal=float(rng.uniform(2.2, 3.8)), lateral=float(rng.uniform(-0.8, 0.8)))
    receiver = _clip_xy(float(center[0] - _team_sign(passer_team) * rng.uniform(0.2, 0.5)), float(center[1] + rng.uniform(-0.25, 0.25)))
    support = _clip_xy(float(receiver[0] - _team_sign(passer_team) * rng.uniform(1.0, 1.5)), float(receiver[1] + rng.uniform(0.9, 1.4) * (1.0 if rng.random() < 0.5 else -1.0)))
    defender_0 = _clip_xy(float(center[0] + _team_sign(passer_team) * rng.uniform(0.25, 0.65)), float(center[1] + rng.uniform(-0.25, 0.25)))
    defender_1 = _clip_xy(float(center[0] + _team_sign(passer_team) * rng.uniform(0.9, 1.4)), float(center[1] + rng.uniform(0.7, 1.2) * (1.0 if rng.random() < 0.5 else -1.0)))
    ball_vel = [float(_team_sign(passer_team) * rng.uniform(0.18, 0.28)), float(rng.uniform(-0.04, 0.04))]
    players = _arrange_players(receiver, support, defender_0, defender_1) if passer_team == "home" else _arrange_players(defender_0, defender_1, receiver, support)
    return ScenarioSpec(
        family_id="second_ball_continuation",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(16, 44)), "ball_position": [float(center[0]), float(center[1])], "ball_velocity": ball_vel, "ball_owner_id": None, "players": players},
        recent_events=[{"event_type": "pass_started", "team": passer_team, "from": f"{passer_team}_0", "target": [float(center[0]), float(center[1])]}],
        description="The ball has just been kicked into a dangerous zone and the next touch matters.",
        tags=["second_ball", passer_team],
    )


def _scenario_transition_attack(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    defend_team = "away" if owner_team == "home" else "home"
    owner_id = f"{owner_team}_0"
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(3.0, 4.2)), lateral=float(rng.uniform(-0.4, 0.4)))
    support = _support_point(anchor, owner_team, forward=float(rng.uniform(1.3, 2.0)), lateral=float(rng.uniform(0.7, 1.2)) * (1.0 if rng.random() < 0.5 else -1.0))
    retreat_0 = _support_point(anchor, defend_team, forward=float(rng.uniform(1.0, 1.8)), lateral=float(rng.uniform(-0.5, 0.5)))
    retreat_1 = _support_point(anchor, defend_team, forward=float(rng.uniform(1.8, 2.5)), lateral=float(rng.uniform(0.8, 1.3)) * (1.0 if rng.random() < 0.5 else -1.0))
    players = _arrange_players(anchor, support, retreat_0, retreat_1, owner_id=owner_id) if owner_team == "home" else _arrange_players(retreat_0, retreat_1, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="transition_attack",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(12, 36)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "turnover", "to_player": owner_id}],
        description="A turnover has created space for a quick transition attack.",
        tags=["transition", owner_team],
    )


def _scenario_emergency_defending(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    defend_team = _random_team(rng)
    attack_team = "away" if defend_team == "home" else "home"
    owner_id = f"{attack_team}_0"
    anchor = _attack_point(attack_team, dist_to_goal=float(rng.uniform(1.6, 2.4)), lateral=float(rng.uniform(-0.6, 0.6)))
    attack_support = _support_point(anchor, attack_team, forward=float(rng.uniform(0.4, 0.8)), lateral=float(rng.uniform(0.7, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0))
    defend_0 = _clip_xy(float(anchor[0] - _team_sign(attack_team) * rng.uniform(0.7, 1.1)), float(FIELD_H * 0.5 + rng.uniform(-0.5, 0.5)))
    defend_1 = _clip_xy(float(anchor[0] - _team_sign(attack_team) * rng.uniform(0.2, 0.7)), float(anchor[1] + rng.uniform(0.7, 1.1) * (1.0 if rng.random() < 0.5 else -1.0)))
    players = _arrange_players(anchor, attack_support, defend_0, defend_1, owner_id=owner_id) if attack_team == "home" else _arrange_players(defend_0, defend_1, anchor, attack_support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="emergency_defending",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(24, 50)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "turnover", "to_player": owner_id}],
        description="The defending team starts in emergency mode near its own goal.",
        tags=["defending", defend_team, "danger_zone"],
    )


def _scenario_box_crowding_rebound(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    attack_team = _random_team(rng)
    center = _attack_point(attack_team, dist_to_goal=float(rng.uniform(0.9, 1.6)), lateral=float(rng.uniform(-0.45, 0.45)))
    a0 = _clip_xy(float(center[0] - _team_sign(attack_team) * rng.uniform(0.15, 0.45)), float(center[1] + rng.uniform(-0.25, 0.25)))
    a1 = _clip_xy(float(center[0] - _team_sign(attack_team) * rng.uniform(0.55, 0.95)), float(center[1] + rng.uniform(0.55, 0.9) * (1.0 if rng.random() < 0.5 else -1.0)))
    d0 = _clip_xy(float(center[0] + _team_sign(attack_team) * rng.uniform(0.15, 0.45)), float(center[1] + rng.uniform(-0.25, 0.25)))
    d1 = _clip_xy(float(center[0] + _team_sign(attack_team) * rng.uniform(0.55, 0.95)), float(center[1] + rng.uniform(0.55, 0.9) * (1.0 if rng.random() < 0.5 else -1.0)))
    ball_vel = [float(_team_sign(attack_team) * rng.uniform(-0.05, 0.12)), float(rng.uniform(-0.05, 0.05))]
    players = _arrange_players(a0, a1, d0, d1) if attack_team == "home" else _arrange_players(d0, d1, a0, a1)
    return ScenarioSpec(
        family_id="box_crowding_rebound",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(30, 55)), "ball_position": [float(center[0]), float(center[1])], "ball_velocity": ball_vel, "ball_owner_id": None, "players": players},
        recent_events=[{"event_type": "ball_player_collision"}, {"event_type": "loose_ball_scramble"}],
        description="The ball is bouncing loose in the box and the next touch should be decisive.",
        tags=["box", "rebound", attack_team],
    )


def _scenario_dead_ball_restart_analog(rng: np.random.Generator, episode_seed: int) -> ScenarioSpec:
    owner_team = _random_team(rng)
    owner_id = f"{owner_team}_0"
    lateral_sign = 1.0 if rng.random() < 0.5 else -1.0
    anchor = _attack_point(owner_team, dist_to_goal=float(rng.uniform(3.2, 5.0)), lateral=lateral_sign * rng.uniform(1.4, 2.2))
    support = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(0.9, 1.4)), float(anchor[1] - lateral_sign * rng.uniform(0.8, 1.2)))
    block_0 = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(0.4, 0.9)), float(anchor[1] - lateral_sign * rng.uniform(0.1, 0.4)))
    block_1 = _clip_xy(float(anchor[0] + _team_sign(owner_team) * rng.uniform(1.0, 1.6)), float(anchor[1] - lateral_sign * rng.uniform(0.8, 1.3)))
    players = _arrange_players(anchor, support, block_0, block_1, owner_id=owner_id) if owner_team == "home" else _arrange_players(block_0, block_1, anchor, support, owner_id=owner_id)
    return ScenarioSpec(
        family_id="dead_ball_restart_analog",
        episode_seed=episode_seed,
        state_spec={"step": int(rng.integers(8, 25)), "ball_position": [float(anchor[0]), float(anchor[1])], "ball_velocity": [0.0, 0.0], "ball_owner_id": owner_id, "players": players},
        recent_events=[{"event_type": "dead_ball"}, {"event_type": "ball_out_of_bounds", "side": "top" if lateral_sign > 0 else "bottom"}],
        description="A restart-like setup near the sideline should produce an intentional re-entry action.",
        tags=["restart", owner_team, "sideline"],
    )


SCENARIO_BUILDERS = {
    "controlled_buildup": _scenario_controlled_buildup,
    "near_goal_finishing": _scenario_near_goal_finishing,
    "sideline_danger": _scenario_sideline_danger,
    "pressured_possession": _scenario_pressured_possession,
    "loose_ball_scramble": _scenario_loose_ball_scramble,
    "second_ball_continuation": _scenario_second_ball_continuation,
    "transition_attack": _scenario_transition_attack,
    "emergency_defending": _scenario_emergency_defending,
    "box_crowding_rebound": _scenario_box_crowding_rebound,
    "dead_ball_restart_analog": _scenario_dead_ball_restart_analog,
}


def _build_scenario(family_id: str, episode_seed: int) -> ScenarioSpec:
    return SCENARIO_BUILDERS[family_id](np.random.default_rng(episode_seed), episode_seed)


def _resolve_vlm_api_key() -> str:
    try:
        return _get_vlm_api_key()
    except RuntimeError:
        pass
    notes_path = ROOT / "docs" / "hyperGym.md"
    if notes_path.exists():
        text = notes_path.read_text(encoding="utf-8")
        match = re.search(r"API key: `([^`]+)`", text)
        if match:
            return match.group(1).strip()
    raise RuntimeError("未设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY，且 docs/hyperGym.md 中也未找到 API key")


def _rollout_scripted_scenario(scenario: ScenarioSpec, *, max_steps: int) -> List[Dict[str, Any]]:
    controller = build_match_controller(num_home=2, num_away=2, seed=scenario.episode_seed, max_steps=max(100, int(scenario.state_spec.get("step", 0)) + max_steps), end_on_ball_out=True)
    simulation = controller.simulation
    _set_manual_state(simulation, scenario.state_spec)
    home_policy = SimpleMatchPolicy(team="home", seed=1000 + scenario.episode_seed)
    away_policy = SimpleMatchPolicy(team="away", seed=2000 + scenario.episode_seed)
    recent_events = list(scenario.recent_events)
    rollout: List[Dict[str, Any]] = []
    for step_idx in range(max_steps):
        state = simulation.get_state()
        home_action = home_policy(state)
        away_action = away_policy(state)
        step_row = {
            "step": step_idx,
            "state": state,
            "events_before_step": list(recent_events),
            "actions": {
                "home": {player_id: {"skill": action["skill"], "target": np.asarray(action["target"], dtype=np.float32).copy()} for player_id, action in home_action.items()},
                "away": {player_id: {"skill": action["skill"], "target": np.asarray(action["target"], dtype=np.float32).copy()} for player_id, action in away_action.items()},
            },
        }
        _, reward, done, info = simulation.step(home_action, opponent_action=away_action)
        step_row["reward"] = float(reward)
        step_row["events_after_step"] = list(info.get("events", []))
        step_row["done"] = bool(done)
        rollout.append(step_row)
        recent_events = list(info.get("events", []))
        if done:
            break
    return rollout


def _sample_rollout_indices(rollout: List[Dict[str, Any]], sample_steps: int) -> List[int]:
    if not rollout:
        return []
    if len(rollout) <= sample_steps:
        return list(range(len(rollout)))
    positions = np.linspace(0, len(rollout) - 1, num=sample_steps, dtype=int)
    return sorted(set(int(pos) for pos in positions))


def _compare_action_dict(vlm_action: Dict[str, Any], scripted_action: Dict[str, Any]) -> Dict[str, float]:
    skill_match = 1.0 if str(vlm_action["skill"]) == str(scripted_action["skill"]) else 0.0
    vlm_target = np.asarray(vlm_action["target"], dtype=np.float32)
    scripted_target = np.asarray(scripted_action["target"], dtype=np.float32)
    target_error = float(np.linalg.norm(vlm_target - scripted_target))
    similarity = 0.5 + 0.5 * max(0.0, 1.0 - target_error / 2.0) if skill_match else 0.0
    return {"skill_match": skill_match, "target_error": target_error, "similarity": similarity}


def _summarize_samples(sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = {
        "home": {"player_steps": 0, "skill_matches": 0.0, "similarity_sum": 0.0, "target_errors": []},
        "away": {"player_steps": 0, "skill_matches": 0.0, "similarity_sum": 0.0, "target_errors": []},
        "team_step_exact_match": {"home": 0, "away": 0},
    }
    for sample in sample_rows:
        for team in TEAM_IDS:
            team_all_match = True
            for player_id in PLAYER_IDS[team]:
                compared = sample["teams"][team]["players"][player_id]
                aggregate[team]["player_steps"] += 1
                aggregate[team]["skill_matches"] += compared["skill_match"]
                aggregate[team]["similarity_sum"] += compared["similarity"]
                aggregate[team]["target_errors"].append(compared["target_error"])
                if compared["skill_match"] < 1.0:
                    team_all_match = False
            aggregate["team_step_exact_match"][team] += int(team_all_match)

    result = {"sample_count": len(sample_rows), "per_team": {}, "sample_rows": sample_rows}
    for team in TEAM_IDS:
        player_steps = max(1, int(aggregate[team]["player_steps"]))
        errors = aggregate[team]["target_errors"] or [FIELD_DIAG]
        result["per_team"][team] = {
            "player_steps": int(aggregate[team]["player_steps"]),
            "skill_match_rate": float(aggregate[team]["skill_matches"]) / player_steps,
            "mean_similarity": float(aggregate[team]["similarity_sum"]) / player_steps,
            "mean_target_error": float(np.mean(errors)),
            "median_target_error": float(np.median(errors)),
            "team_step_exact_match_rate": float(aggregate["team_step_exact_match"][team]) / max(1, len(sample_rows)),
        }
    result["overall_mean_similarity"] = float(np.mean([result["per_team"]["home"]["mean_similarity"], result["per_team"]["away"]["mean_similarity"]]))
    return result


def _query_vlm_on_scripted_samples(scenario: ScenarioSpec, scripted_rollout: List[Dict[str, Any]], *, sample_steps: int, vlm_model: str, vlm_base_url: str) -> Dict[str, Any]:
    vlm = OpenAICompatibleVisionVLM(model=vlm_model, api_key=_resolve_vlm_api_key(), base_url=vlm_base_url)
    sample_indices = _sample_rollout_indices(scripted_rollout, sample_steps)
    sample_rows: List[Dict[str, Any]] = []
    for sample_order, rollout_idx in enumerate(sample_indices):
        scripted_row = scripted_rollout[rollout_idx]
        record = _make_render_record(
            state=scripted_row["state"],
            reward=float(scripted_row["reward"]),
            done=bool(scripted_row["done"]),
            action=scripted_row["actions"]["home"],
            away_action=scripted_row["actions"]["away"],
            events=scripted_row["events_before_step"],
        )
        sample_row = {"sample_order": sample_order, "rollout_step": int(rollout_idx), "teams": {}}
        for team in TEAM_IDS:
            team_query = _query_team_vlm(
                vlm=vlm,
                team=team,
                artifact_id=f"{scenario.family_id}_seed{scenario.episode_seed}_sample{sample_order:02d}_step{rollout_idx:03d}",
                record=record,
                state_text=_state_text_summary(scripted_row["state"], scripted_row["events_before_step"], team=team),
                artifact_dir=None,
            )
            player_rows: Dict[str, Any] = {}
            for player_id in PLAYER_IDS[team]:
                vlm_action = team_query["action"][player_id]
                scripted_action = scripted_row["actions"][team][player_id]
                player_rows[player_id] = {
                    "vlm": {"skill": vlm_action["skill"], "target": np.asarray(vlm_action["target"], dtype=np.float32).copy()},
                    "scripted": {"skill": scripted_action["skill"], "target": np.asarray(scripted_action["target"], dtype=np.float32).copy()},
                    **_compare_action_dict(vlm_action, scripted_action),
                }
            sample_row["teams"][team] = {"raw_decision": team_query["raw_decision"], "players": player_rows}
        sample_rows.append(sample_row)
    return _summarize_samples(sample_rows)


def _evaluate_family(family_id: str, *, episodes: int, base_seed: int, rollout_steps: int, sample_steps: int, vlm_model: str, vlm_base_url: str, save_dir: str) -> Dict[str, Any]:
    family_dir = Path(save_dir) / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries: List[Dict[str, Any]] = []
    for episode_idx in range(episodes):
        episode_seed = int(base_seed + episode_idx)
        scenario = _build_scenario(family_id, episode_seed)
        scripted_rollout = _rollout_scripted_scenario(scenario, max_steps=rollout_steps)
        comparison = _query_vlm_on_scripted_samples(scenario, scripted_rollout, sample_steps=sample_steps, vlm_model=vlm_model, vlm_base_url=vlm_base_url)
        episode_summary = {
            "family_id": family_id,
            "episode_idx": episode_idx,
            "episode_seed": episode_seed,
            "scenario": {"description": scenario.description, "tags": scenario.tags, "state_spec": scenario.state_spec, "recent_events": scenario.recent_events},
            "comparison": comparison,
            "scripted_rollout": scripted_rollout,
        }
        episode_summaries.append(episode_summary)
        (family_dir / f"episode_{episode_idx:02d}_seed{episode_seed}.json").write_text(json.dumps(_to_plain(episode_summary), ensure_ascii=False, indent=2), encoding="utf-8")

    home_skill = [item["comparison"]["per_team"]["home"]["skill_match_rate"] for item in episode_summaries]
    away_skill = [item["comparison"]["per_team"]["away"]["skill_match_rate"] for item in episode_summaries]
    home_sim = [item["comparison"]["per_team"]["home"]["mean_similarity"] for item in episode_summaries]
    away_sim = [item["comparison"]["per_team"]["away"]["mean_similarity"] for item in episode_summaries]
    home_exact = [item["comparison"]["per_team"]["home"]["team_step_exact_match_rate"] for item in episode_summaries]
    away_exact = [item["comparison"]["per_team"]["away"]["team_step_exact_match_rate"] for item in episode_summaries]

    result = {
        "family_id": family_id,
        "episodes": episodes,
        "rollout_steps": rollout_steps,
        "sample_steps": sample_steps,
        "aggregate": {
            "home_skill_match_rate_mean": float(np.mean(home_skill)),
            "away_skill_match_rate_mean": float(np.mean(away_skill)),
            "home_similarity_mean": float(np.mean(home_sim)),
            "away_similarity_mean": float(np.mean(away_sim)),
            "home_step_exact_match_mean": float(np.mean(home_exact)),
            "away_step_exact_match_mean": float(np.mean(away_exact)),
        },
        "episode_summaries": [
            {
                "episode_idx": item["episode_idx"],
                "episode_seed": item["episode_seed"],
                "home_skill_match_rate": item["comparison"]["per_team"]["home"]["skill_match_rate"],
                "away_skill_match_rate": item["comparison"]["per_team"]["away"]["skill_match_rate"],
                "home_similarity": item["comparison"]["per_team"]["home"]["mean_similarity"],
                "away_similarity": item["comparison"]["per_team"]["away"]["mean_similarity"],
                "scenario_tags": item["scenario"]["tags"],
            }
            for item in episode_summaries
        ],
    }
    (family_dir / "summary.json").write_text(json.dumps(_to_plain(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _family_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _evaluate_family(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM behavior on manually generated HyperGym scenarios.")
    parser.add_argument("--families", type=str, default=",".join(SCENARIO_BUILDERS.keys()))
    parser.add_argument("--episodes-per-family", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vlm-model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_VLM_BASE_URL))
    parser.add_argument("--save-dir", type=str, default="logs/vlm_scenario_eval")
    args = parser.parse_args()

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    invalid = [item for item in families if item not in SCENARIO_BUILDERS]
    if invalid:
        raise ValueError(f"Unknown families: {invalid}. Available: {sorted(SCENARIO_BUILDERS)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for family_idx, family_id in enumerate(families):
        worker_args.append(
            {
                "family_id": family_id,
                "episodes": int(args.episodes_per_family),
                "base_seed": int(args.base_seed + family_idx * 1000),
                "rollout_steps": int(args.rollout_steps),
                "sample_steps": int(args.sample_steps),
                "vlm_model": args.vlm_model,
                "vlm_base_url": args.vlm_base_url,
                "save_dir": str(save_dir),
            }
        )

    family_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {executor.submit(_family_worker, item): item["family_id"] for item in worker_args}
        for future in as_completed(future_map):
            family_id = future_map[future]
            result = future.result()
            family_results.append(result)
            aggregate = result["aggregate"]
            print(
                f"[scenario] {family_id} "
                f"home_skill={aggregate['home_skill_match_rate_mean']:.3f} "
                f"away_skill={aggregate['away_skill_match_rate_mean']:.3f} "
                f"home_sim={aggregate['home_similarity_mean']:.3f} "
                f"away_sim={aggregate['away_similarity_mean']:.3f}"
            )

    family_results.sort(key=lambda item: families.index(item["family_id"]))
    overall = {
        "families": families,
        "episodes_per_family": int(args.episodes_per_family),
        "rollout_steps": int(args.rollout_steps),
        "sample_steps": int(args.sample_steps),
        "vlm_model": args.vlm_model,
        "vlm_base_url": args.vlm_base_url,
        "family_results": family_results,
    }
    (save_dir / "summary.json").write_text(json.dumps(_to_plain(overall), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
