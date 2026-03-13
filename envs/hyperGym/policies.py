from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ScriptedSoccerPolicy:
    team: str
    rng: np.random.Generator

    def act(self, state: Dict) -> Dict:
        field_size = np.asarray(state.get("field_size", [10.0, 6.0]), dtype=np.float32)
        field_w = float(field_size[0])
        field_h = float(field_size[1])
        players = state["players"]
        teammates = [player for player in players if player["team"] == self.team]
        opponents = [player for player in players if player["team"] != self.team]
        ball = np.asarray(state["ball_position"], dtype=np.float32)
        ball_vel = np.asarray(state.get("ball_velocity", [0.0, 0.0]), dtype=np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))
        owner_id = state["ball_owner_id"]

        attack_sign = 1.0 if self.team == "home" else -1.0
        goal_x = field_w if self.team == "home" else 0.0
        own_goal_x = 0.0 if self.team == "home" else field_w
        goal_center = np.asarray([goal_x, field_h * 0.5], dtype=np.float32)
        own_goal = np.asarray([own_goal_x, field_h * 0.5], dtype=np.float32)
        owner = next((player for player in teammates if player["player_id"] == owner_id), None)
        active = owner
        if active is None:
            active = min(teammates, key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - ball)))
        primary_opponent = min(opponents, key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - ball)))
        my_pos = np.asarray(active["position"], dtype=np.float32)
        opp_pos = np.asarray(primary_opponent["position"], dtype=np.float32)
        support_teammate = self._select_support_teammate(teammates, active["player_id"], goal_center)

        if owner_id == active["player_id"]:
            dist_to_goal = float(np.linalg.norm(goal_center - my_pos))
            lane_open = abs(float(my_pos[1] - goal_center[1])) < 1.25
            if dist_to_goal < 2.4 and lane_open:
                shot_target = goal_center + np.asarray([0.45 * attack_sign, 0.0], dtype=np.float32)
                return {"skill": "pass", "target": shot_target}

            if support_teammate is not None and self._is_pass_lane_open(my_pos, np.asarray(support_teammate["position"], dtype=np.float32), opponents):
                mate_pos = np.asarray(support_teammate["position"], dtype=np.float32)
                lead_target = mate_pos + np.asarray([0.35 * attack_sign, 0.0], dtype=np.float32)
                if float(np.linalg.norm(mate_pos - my_pos)) > 1.4 and self.rng.random() < 0.35:
                    return {"skill": "pass", "target": lead_target}

            if float(np.linalg.norm(opp_pos - my_pos)) < 0.85:
                lane_offset = self.rng.uniform(-1.1, 1.1)
                through_target = np.asarray([
                    my_pos[0] + attack_sign * self.rng.uniform(1.6, 2.8),
                    np.clip(goal_center[1] + lane_offset, 0.35, field_h - 0.35),
                ], dtype=np.float32)
                return {"skill": "pass", "target": through_target}

            if self.rng.random() < 0.22:
                touch_target = np.asarray([
                    my_pos[0] + attack_sign * self.rng.uniform(1.2, 2.1),
                    np.clip(0.6 * my_pos[1] + 0.4 * goal_center[1] + self.rng.uniform(-0.5, 0.5), 0.35, field_h - 0.35),
                ], dtype=np.float32)
                return {"skill": "pass", "target": touch_target}

            forward_target = np.asarray([
                my_pos[0] + attack_sign * self.rng.uniform(0.7, 1.15),
                np.clip(0.6 * my_pos[1] + 0.4 * goal_center[1] + self.rng.uniform(-0.35, 0.35), 0.3, field_h - 0.3),
            ], dtype=np.float32)
            return {"skill": "move", "target": forward_target}

        if owner_id is None:
            if active["player_id"] == min(teammates, key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - ball)))["player_id"]:
                if float(np.linalg.norm(my_pos - ball)) < 0.4 and ball_speed < 0.08:
                    direct_target = goal_center + np.asarray([0.35 * attack_sign, self.rng.uniform(-0.4, 0.4)], dtype=np.float32)
                    return {"skill": "pass", "target": direct_target}
                return {"skill": "trap", "target": ball.copy()}
            return {"skill": "move", "target": ball.copy()}

        if owner_id in {player["player_id"] for player in opponents}:
            ball_owner = next(player for player in opponents if player["player_id"] == owner_id)
            owner_pos = np.asarray(ball_owner["position"], dtype=np.float32)
            intercept = 0.62 * owner_pos + 0.38 * own_goal
            intercept[1] += self.rng.uniform(-0.3, 0.3)
            return {"skill": "move", "target": intercept}

        if support_teammate is not None:
            support_pos = np.asarray(support_teammate["position"], dtype=np.float32)
            support_target = np.asarray([
                0.55 * ball[0] + 0.45 * support_pos[0],
                np.clip(0.6 * ball[1] + 0.4 * support_pos[1], 0.25, field_h - 0.25),
            ], dtype=np.float32)
            return {"skill": "move", "target": support_target}

        return {"skill": "move", "target": ball.copy()}

    def _select_support_teammate(self, teammates: list[Dict], active_id: str, goal_center: np.ndarray) -> Dict | None:
        candidates = [player for player in teammates if player["player_id"] != active_id]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - goal_center)),
        )

    def _is_pass_lane_open(self, start: np.ndarray, target: np.ndarray, opponents: list[Dict]) -> bool:
        segment = target - start
        seg_norm_sq = float(np.dot(segment, segment))
        if seg_norm_sq < 1e-8:
            return False
        for opponent in opponents:
            opp_pos = np.asarray(opponent["position"], dtype=np.float32)
            t = float(np.clip(np.dot(opp_pos - start, segment) / seg_norm_sq, 0.0, 1.0))
            closest = start + t * segment
            if float(np.linalg.norm(opp_pos - closest)) < 0.55:
                return False
        return True


class SimpleMatchPolicy:
    def __init__(self, team: str, seed: int = 0):
        self.team = team
        self.rng = np.random.default_rng(seed)
        self._policy = ScriptedSoccerPolicy(team=team, rng=self.rng)

    def __call__(self, state: Dict) -> Dict:
        return self._policy.act(state)
