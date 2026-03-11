from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from envs.hyperGym.ball import Ball
from envs.hyperGym.data_interface import DataInterface
from envs.hyperGym.player import Player
from envs.hyperGym.training_interface import TrainingInterface


@dataclass
class HyperParams:
    field_size: Tuple[float, float] = (10.0, 6.0)
    max_steps: int = 200
    ball_speed: float = 0.28
    steal_radius: float = 0.2
    shoot_radius: float = 1.2
    base_intercept_prob: float = 0.15
    defender_speed_scale: float = 1.0
    defender_press_bias: float = 1.0


class HyperGymSimulation:
    def __init__(
        self,
        num_home: int = 2,
        num_away: int = 2,
        params: HyperParams | None = None,
        seed: int | None = None,
    ):
        self.params = params or HyperParams()
        self.rng = np.random.default_rng(seed)
        self.data = DataInterface()
        self.training = TrainingInterface()
        self.players: List[Player] = []
        self.ball = Ball(position=np.zeros(2, dtype=np.float32))
        self.step_count = 0
        self.num_home = num_home
        self.num_away = num_away
        self._build_players()
        self.reset()

    def _build_players(self) -> None:
        self.players = []
        for idx in range(self.num_home):
            self.players.append(Player(player_id=f"home_{idx}", team="home", position=np.zeros(2, dtype=np.float32)))
        for idx in range(self.num_away):
            self.players.append(Player(player_id=f"away_{idx}", team="away", position=np.zeros(2, dtype=np.float32), max_speed=0.11))

    def _spawn_layout(self) -> None:
        width, height = self.params.field_size
        for idx, player in enumerate(self.players):
            if player.team == "home":
                x = 1.0 + idx * 0.8
                y = height * (0.35 + 0.3 * idx / max(1, self.num_home))
            else:
                away_idx = idx - self.num_home
                x = width * 0.6 + away_idx * 0.7
                y = height * (0.3 + 0.4 * away_idx / max(1, self.num_away))
            player.reset(np.asarray([x, y], dtype=np.float32))

        owner = self.players[0]
        owner.has_ball = True
        self.ball.reset(owner.position, owner.player_id)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.data.reset()
        self._spawn_layout()
        self.training.reset(self.ball.position)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return self.data.build_observation(self.players, self.ball, self.params.field_size, controlled_team="home")

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        step_events: List[Dict] = []

        self._apply_home_action(action, step_events)
        self._apply_away_policy(step_events)
        self._resolve_ball_motion(step_events)
        self._resolve_possession(step_events)

        state = self._build_state()
        done = self.step_count >= self.params.max_steps or state["goal"]
        state["done"] = done
        reward = self.training.compute_reward(state, step_events)

        info = {
            "events": step_events,
            "state": state,
        }
        return self.get_obs(), reward, done, info

    def _apply_home_action(self, action: Dict, step_events: List[Dict]) -> None:
        action_type = action.get("type", "hold")
        owner = self._ball_owner()
        if owner is None or owner.team != "home":
            return

        if action_type == "move":
            target = np.asarray(action.get("target", owner.position), dtype=np.float32)
            owner.move_towards(target)
            self.ball.attach_to(owner.player_id, owner.position)
        elif action_type == "pass":
            receiver_idx = int(action.get("receiver", 0))
            teammates = [player for player in self.players if player.team == "home"]
            receiver_idx = int(np.clip(receiver_idx, 0, len(teammates) - 1))
            receiver = teammates[receiver_idx]
            if receiver.player_id != owner.player_id:
                owner.has_ball = False
                self.ball.release_towards(receiver.position, self.params.ball_speed)
                step_events.append({"event_type": "pass_started", "from": owner.player_id, "to": receiver.player_id})
        elif action_type == "shoot":
            goal = np.asarray([self.params.field_size[0], self.params.field_size[1] * 0.5], dtype=np.float32)
            owner.has_ball = False
            self.ball.release_towards(goal, self.params.ball_speed * 1.15)
            step_events.append({"event_type": "shot_taken", "by": owner.player_id})
        else:
            self.ball.attach_to(owner.player_id, owner.position)

    def _apply_away_policy(self, step_events: List[Dict]) -> None:
        ball_target = self.ball.position.copy()
        for defender in self.players:
            if defender.team != "away":
                continue
            defender.move_towards(
                ball_target,
                speed_scale=self.params.defender_speed_scale * self.params.defender_press_bias,
            )
            defender.clamp(self.params.field_size)

    def _resolve_ball_motion(self, step_events: List[Dict]) -> None:
        owner = self._ball_owner()
        if owner is not None:
            owner.clamp(self.params.field_size)
            self.ball.attach_to(owner.player_id, owner.position)
            return

        current = self.ball.position.copy()
        self.ball.step_free()
        self.ball.position[0] = float(np.clip(self.ball.position[0], 0.0, self.params.field_size[0]))
        self.ball.position[1] = float(np.clip(self.ball.position[1], 0.0, self.params.field_size[1]))

        for defender in [player for player in self.players if player.team == "away"]:
            intercept_prob = self._estimate_intercept_prob(defender, current, self.ball.position)
            if self.rng.random() < intercept_prob:
                defender.has_ball = True
                self.ball.attach_to(defender.player_id, defender.position)
                step_events.append({"event_type": "intercepted", "by": defender.player_id})
                return

        for teammate in [player for player in self.players if player.team == "home"]:
            if teammate.distance_to(self.ball.position) <= teammate.control_radius:
                teammate.has_ball = True
                self.ball.attach_to(teammate.player_id, teammate.position)
                step_events.append({"event_type": "pass_completed", "to": teammate.player_id})
                return

    def _resolve_possession(self, step_events: List[Dict]) -> None:
        owner = self._ball_owner()
        if owner is None:
            return

        nearby = [player for player in self.players if player.team != owner.team and player.distance_to(owner.position) <= self.params.steal_radius]
        if nearby and self.rng.random() < 0.08 * self.params.defender_press_bias:
            thief = nearby[0]
            owner.has_ball = False
            thief.has_ball = True
            self.ball.attach_to(thief.player_id, thief.position)
            step_events.append({"event_type": "turnover", "from_player": owner.player_id, "to_player": thief.player_id})

    def _estimate_intercept_prob(self, defender: Player, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        segment = seg_end - seg_start
        seg_norm = float(np.linalg.norm(segment))
        if seg_norm < 1e-8:
            return 0.0
        rel = defender.position - seg_start
        t = float(np.clip(np.dot(rel, segment) / max(seg_norm ** 2, 1e-8), 0.0, 1.0))
        closest = seg_start + t * segment
        dist = float(np.linalg.norm(defender.position - closest))
        raw = self.params.base_intercept_prob * max(0.0, 1.0 - dist / 1.2) * self.params.defender_press_bias
        return float(np.clip(raw, 0.0, 0.9))

    def _ball_owner(self) -> Player | None:
        if self.ball.owner_id is None:
            return None
        for player in self.players:
            if player.player_id == self.ball.owner_id:
                return player
        return None

    def _build_state(self) -> Dict:
        goal = self.ball.position[0] >= self.params.field_size[0] and abs(self.ball.position[1] - self.params.field_size[1] * 0.5) <= 1.0
        return {
            "step": self.step_count,
            "ball_position": self.ball.position.copy(),
            "ball_owner_id": self.ball.owner_id,
            "goal": goal,
            "players": [player.copy_public_state() for player in self.players],
        }

