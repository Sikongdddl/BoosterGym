from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ScriptedSoccerPolicy:
    team: str
    rng: np.random.Generator

    def act(self, state: Dict) -> Dict[str, Dict]:
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
        primary = owner
        if primary is None:
            primary = min(teammates, key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - ball)))

        primary_opponent = min(opponents, key=lambda player: float(np.linalg.norm(np.asarray(player["position"], dtype=np.float32) - ball)))
        primary_pos = np.asarray(primary["position"], dtype=np.float32)
        opponent_pos = np.asarray(primary_opponent["position"], dtype=np.float32)
        support_targets = self._build_support_targets(
            teammates=teammates,
            primary_id=primary["player_id"],
            goal_center=goal_center,
            own_goal=own_goal,
            ball=ball,
            attack_sign=attack_sign,
            field_h=field_h,
        )

        actions: Dict[str, Dict] = {}
        for teammate in teammates:
            if teammate["player_id"] == primary["player_id"]:
                actions[teammate["player_id"]] = self._primary_action(
                    primary=primary,
                    teammates=teammates,
                    opponents=opponents,
                    goal_center=goal_center,
                    attack_sign=attack_sign,
                    field_h=field_h,
                    ball=ball,
                    ball_speed=ball_speed,
                    owner_id=owner_id,
                    opponent_pos=opponent_pos,
                )
            else:
                target = support_targets.get(teammate["player_id"], np.asarray(teammate["position"], dtype=np.float32))
                actions[teammate["player_id"]] = {"skill": "move", "target": target}
        return actions

    def _primary_action(
        self,
        primary: Dict,
        teammates: list[Dict],
        opponents: list[Dict],
        goal_center: np.ndarray,
        attack_sign: float,
        field_h: float,
        ball: np.ndarray,
        ball_speed: float,
        owner_id: str | None,
        opponent_pos: np.ndarray,
    ) -> Dict:
        primary_pos = np.asarray(primary["position"], dtype=np.float32)
        support_teammate = self._select_support_teammate(teammates, primary["player_id"], goal_center)

        if owner_id == primary["player_id"]:
            dist_to_goal = float(np.linalg.norm(goal_center - primary_pos))
            lane_open = abs(float(primary_pos[1] - goal_center[1])) < 1.25
            if dist_to_goal < 2.4 and lane_open:
                shot_target = goal_center + np.asarray([0.45 * attack_sign, 0.0], dtype=np.float32)
                return {"skill": "pass", "target": shot_target}

            if support_teammate is not None and self._is_pass_lane_open(primary_pos, np.asarray(support_teammate["position"], dtype=np.float32), opponents):
                mate_pos = np.asarray(support_teammate["position"], dtype=np.float32)
                lead_target = mate_pos + np.asarray([0.35 * attack_sign, 0.0], dtype=np.float32)
                if float(np.linalg.norm(mate_pos - primary_pos)) > 1.4 and self.rng.random() < 0.35:
                    return {"skill": "pass", "target": lead_target}

            if float(np.linalg.norm(opponent_pos - primary_pos)) < 0.85:
                lane_offset = self.rng.uniform(-1.1, 1.1)
                through_target = np.asarray([
                    primary_pos[0] + attack_sign * self.rng.uniform(1.6, 2.8),
                    np.clip(goal_center[1] + lane_offset, 0.35, field_h - 0.35),
                ], dtype=np.float32)
                return {"skill": "pass", "target": through_target}

            if self.rng.random() < 0.22:
                touch_target = np.asarray([
                    primary_pos[0] + attack_sign * self.rng.uniform(1.2, 2.1),
                    np.clip(0.6 * primary_pos[1] + 0.4 * goal_center[1] + self.rng.uniform(-0.5, 0.5), 0.35, field_h - 0.35),
                ], dtype=np.float32)
                return {"skill": "pass", "target": touch_target}

            forward_target = np.asarray([
                primary_pos[0] + attack_sign * self.rng.uniform(0.7, 1.15),
                np.clip(0.6 * primary_pos[1] + 0.4 * goal_center[1] + self.rng.uniform(-0.35, 0.35), 0.3, field_h - 0.3),
            ], dtype=np.float32)
            return {"skill": "move", "target": forward_target}

        if owner_id is None:
            if float(np.linalg.norm(primary_pos - ball)) < 0.4 and ball_speed < 0.08:
                direct_target = goal_center + np.asarray([0.35 * attack_sign, self.rng.uniform(-0.4, 0.4)], dtype=np.float32)
                return {"skill": "pass", "target": direct_target}
            return {"skill": "trap", "target": ball.copy()}

        if owner_id in {player["player_id"] for player in opponents}:
            ball_owner = next(player for player in opponents if player["player_id"] == owner_id)
            owner_pos = np.asarray(ball_owner["position"], dtype=np.float32)
            intercept = 0.62 * owner_pos + 0.38 * np.asarray([0.0 if self.team == "home" else float(goal_center[0]), goal_center[1]], dtype=np.float32)
            intercept[1] += self.rng.uniform(-0.3, 0.3)
            return {"skill": "move", "target": intercept}

        return {"skill": "move", "target": ball.copy()}

    def _build_support_targets(
        self,
        teammates: list[Dict],
        primary_id: str,
        goal_center: np.ndarray,
        own_goal: np.ndarray,
        ball: np.ndarray,
        attack_sign: float,
        field_h: float,
    ) -> Dict[str, np.ndarray]:
        targets: Dict[str, np.ndarray] = {}
        support_idx = 0
        for teammate in teammates:
            if teammate["player_id"] == primary_id:
                continue
            pos = np.asarray(teammate["position"], dtype=np.float32)
            lane_y = np.clip(goal_center[1] + (-0.9 + 1.8 * support_idx), 0.35, field_h - 0.35)
            if attack_sign > 0:
                lane_x = max(float(ball[0]) + 1.2 + 0.3 * support_idx, pos[0])
            else:
                lane_x = min(float(ball[0]) - 1.2 - 0.3 * support_idx, pos[0])
            attack_target = np.asarray([lane_x, lane_y], dtype=np.float32)
            cover_target = 0.55 * ball + 0.45 * own_goal
            targets[teammate["player_id"]] = 0.7 * attack_target + 0.3 * cover_target
            support_idx += 1
        return targets

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

    def __call__(self, state: Dict) -> Dict[str, Dict]:
        return self._policy.act(state)
