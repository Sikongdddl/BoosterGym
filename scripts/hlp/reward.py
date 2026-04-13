from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def team_from_player_id(player_id: str | None) -> str | None:
    if not player_id:
        return None
    if player_id.startswith("home_"):
        return "home"
    if player_id.startswith("away_"):
        return "away"
    return None


@dataclass
class TeamRewardConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "step_cost": -0.01,
        "ball_progress": 0.35,
        "final_third_possession": 0.08,
        "trap_completed": 0.12,
        "pass_started": 0.30,
        "move_touch": 0.04,
        "ball_control_gained": 0.20,
        "ball_control_lost": -0.22,
        "turnover_won": 0.28,
        "turnover_lost": -0.30,
        "steal_won": 0.18,
        "dead_ball": -0.05,
        "loose_ball": -0.03,
        "ball_out_of_bounds": -0.40,
        "goal_scored": 5.0,
        "goal_conceded": -5.0,
    })


class TeamRewardShaper:
    def __init__(self, config: TeamRewardConfig | None = None):
        self.config = config or TeamRewardConfig()
        self._last_ball_x = 0.0
        self._last_owner_team: str | None = None

    def reset(self, state: Dict, team: str) -> None:
        self._last_ball_x = float(state["ball_position"][0])
        self._last_owner_team = team_from_player_id(state.get("ball_owner_id"))

    def compute_reward(self, team: str, prev_state: Dict, next_state: Dict, events: List[Dict]) -> float:
        weights = self.config.weights
        reward = float(weights["step_cost"])
        field_w = max(float(next_state["field_size"][0]), 1e-6)
        attack_sign = 1.0 if team == "home" else -1.0

        prev_ball_x = float(prev_state["ball_position"][0])
        next_ball_x = float(next_state["ball_position"][0])
        reward += attack_sign * (next_ball_x - prev_ball_x) / field_w * float(weights["ball_progress"])

        owner_team = team_from_player_id(next_state.get("ball_owner_id"))
        if owner_team == team:
            final_third_threshold = 0.66 * field_w if team == "home" else 0.34 * field_w
            in_final_third = next_ball_x >= final_third_threshold if team == "home" else next_ball_x <= final_third_threshold
            if in_final_third:
                reward += float(weights["final_third_possession"])

        for event in events:
            event_type = str(event.get("event_type", ""))
            event_team = str(event.get("team")) if event.get("team") is not None else None

            if event_type == "goal_scored":
                reward += float(weights["goal_scored"] if event_team == team else weights["goal_conceded"])
            elif event_type == "trap_completed" and event_team == team:
                reward += float(weights["trap_completed"])
            elif event_type == "pass_started" and event_team == team:
                reward += float(weights["pass_started"])
            elif event_type == "move_touch" and event_team == team:
                reward += float(weights["move_touch"])
            elif event_type == "ball_control_gained":
                gained_team = team_from_player_id(event.get("to_player"))
                if gained_team == team:
                    reward += float(weights["ball_control_gained"])
            elif event_type == "ball_control_lost":
                lost_team = team_from_player_id(event.get("from_player"))
                if lost_team == team:
                    reward += float(weights["ball_control_lost"])
            elif event_type == "turnover":
                from_team = team_from_player_id(event.get("from_player"))
                to_team = team_from_player_id(event.get("to_player"))
                if to_team == team:
                    reward += float(weights["turnover_won"])
                elif from_team == team:
                    reward += float(weights["turnover_lost"])
            elif event_type == "steal_attempt_won":
                to_team = team_from_player_id(event.get("to_player"))
                if to_team == team:
                    reward += float(weights["steal_won"])
            elif event_type == "dead_ball" and event_team == team:
                reward += float(weights["dead_ball"])
            elif event_type == "loose_ball" and event_team == team:
                reward += float(weights["loose_ball"])
            elif event_type == "ball_out_of_bounds":
                if self._last_owner_team == team or owner_team == team:
                    reward += float(weights["ball_out_of_bounds"])

        self._last_ball_x = next_ball_x
        self._last_owner_team = owner_team
        return reward
