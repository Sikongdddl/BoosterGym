from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

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
    dribble_speed: float = 0.1
    enable_dribble: bool = False
    sim_dt: float = 1.0
    ball_radius: float = 0.11
    ball_density: float = 80.0
    ball_linear_damping: float = 0.015
    ball_angular_damping: float = 0.01
    ball_default_z: float = 0.12
    ball_restitution: float = 0.0
    wall_restitution: float = 0.0
    goal_half_width: float = 1.0
    player_collision_radius: float = 0.22
    pass_kick_radius: float = 0.28
    steal_radius: float = 0.2
    shoot_radius: float = 1.2
    trap_radius: float = 0.24
    trap_velocity_scale: float = 0.18
    trap_stop_speed: float = 0.03
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
        self.opponent_policy: Callable[[Dict], Dict] | None = None
        self.players: List[Player] = []
        self.ball = Ball(
            position=np.zeros(2, dtype=np.float32),
            radius=self.params.ball_radius,
            density=self.params.ball_density,
            linear_damping=self.params.ball_linear_damping,
            angular_damping=self.params.ball_angular_damping,
            default_z=self.params.ball_default_z,
            restitution=self.params.wall_restitution,
        )
        self.step_count = 0
        self.num_home = num_home
        self.num_away = num_away
        self.winner: str | None = None
        self.last_actions: Dict[str, Dict] = {}
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
        owner.has_ball = False
        self.ball.reset(owner.position, None)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.winner = None
        self.last_actions = {}
        self.data.reset()
        self._spawn_layout()
        self.training.reset(self.ball.position)
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return self.data.build_observation(self.players, self.ball, self.params.field_size, controlled_team="home")

    def get_state(self) -> Dict:
        state = self._build_state()
        state["done"] = self.step_count >= self.params.max_steps or self.winner is not None
        return state

    def step(self, action: Dict, opponent_action: Dict | None = None) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        step_events: List[Dict] = []

        home_action = self._normalize_action(action, team="home")
        if opponent_action is None and self.opponent_policy is not None:
            opponent_action = self.opponent_policy(self.get_state())
        away_action = self._normalize_action(opponent_action, team="away")

        self.last_actions = {"home": home_action, "away": away_action}
        self._apply_team_action(home_action, team="home", step_events=step_events)
        self._apply_team_action(away_action, team="away", step_events=step_events)
        self._resolve_player_collisions(step_events)
        self._resolve_ball_motion(step_events)
        self._resolve_possession(step_events)
        self._resolve_player_collisions(step_events)

        state = self._build_state()
        done = self.step_count >= self.params.max_steps or self.winner is not None
        state["done"] = done
        reward = self.training.compute_reward(state, step_events)

        info = {
            "events": step_events,
            "state": state,
            "actions": {
                "home": home_action,
                "away": away_action,
            },
        }
        return self.get_obs(), reward, done, info

    def _normalize_action(self, action: Dict | None, team: str) -> Dict:
        if action is None:
            return {"skill": "move", "target": self.ball.position.copy()}

        skill = str(action.get("skill", action.get("type", "move"))).lower()
        if skill == "shoot":
            skill = "pass"

        normalized = {"skill": skill}
        if "target" in action and action["target"] is not None:
            normalized["target"] = np.asarray(action["target"], dtype=np.float32)
        elif "receiver" in action:
            teammates = [player for player in self.players if player.team == team]
            receiver_idx = int(np.clip(int(action["receiver"]), 0, len(teammates) - 1))
            normalized["target"] = teammates[receiver_idx].position.copy()
        else:
            normalized["target"] = self.ball.position.copy()
        return normalized

    def _apply_team_action(self, action: Dict, team: str, step_events: List[Dict]) -> None:
        skill = action.get("skill", "move")
        target = self._clip_target(np.asarray(action.get("target", self.ball.position), dtype=np.float32))
        owner = self._ball_owner()
        team_owner = owner if owner is not None and owner.team == team else None

        if skill == "dribble" and not self.params.enable_dribble:
            skill = "move"

        if skill == "pass":
            kicker = team_owner or self._nearest_player(self.ball.position, team=team)
            if kicker is None or kicker.distance_to(self.ball.position) > self.params.pass_kick_radius:
                return
            kicker.has_ball = False
            self.ball.release_towards(target, self.params.ball_speed)
            step_events.append({
                "event_type": "pass_started",
                "team": team,
                "from": kicker.player_id,
                "target": target.copy(),
            })
            return

        if skill == "dribble":
            actor = team_owner or self._nearest_player(self.ball.position, team=team)
            if actor is None:
                return
            if team_owner is None:
                actor.move_towards(self.ball.position)
                actor.clamp(self.params.field_size)
                return
            actor.move_towards(target, speed_scale=self.params.dribble_speed / max(actor.max_speed, 1e-6))
            actor.clamp(self.params.field_size)
            self.ball.attach_to(actor.player_id, actor.position)
            step_events.append({"event_type": "dribble", "team": team, "by": actor.player_id, "target": target.copy()})
            return

        if skill == "trap":
            trapper = self._nearest_player(target, team=team)
            if trapper is None:
                return
            trapper.move_towards(target)
            trapper.clamp(self.params.field_size)
            if self.ball.owner_id is None and trapper.distance_to(self.ball.position) <= self.params.trap_radius:
                self.ball.velocity = self.ball.velocity * self.params.trap_velocity_scale
                speed = float(np.linalg.norm(self.ball.velocity))
                if speed < self.params.trap_stop_speed:
                    self.ball.velocity[:] = 0.0
                step_events.append({
                    "event_type": "trap_completed",
                    "team": team,
                    "by": trapper.player_id,
                    "ball_speed": float(np.linalg.norm(self.ball.velocity)),
                })
            else:
                step_events.append({"event_type": "trap_attempt", "team": team, "by": trapper.player_id, "target": target.copy()})
            return

        mover = team_owner or self._nearest_player(target, team=team)
        if mover is None:
            return
        mover.move_towards(target)
        mover.clamp(self.params.field_size)
        step_events.append({"event_type": "move", "team": team, "by": mover.player_id, "target": target.copy()})

    def _resolve_ball_motion(self, step_events: List[Dict]) -> None:
        owner = self._ball_owner()
        if owner is not None:
            owner.clamp(self.params.field_size)
            self.ball.attach_to(owner.player_id, owner.position)
            return

        current = self.ball.position.copy()
        self.ball.step_free(dt=self.params.sim_dt)

        scorer = self._check_goal(current, self.ball.position)
        if scorer is not None:
            self.winner = scorer
            step_events.append({"event_type": "goal_scored", "team": scorer})
            return

        max_x, max_y = self.params.field_size
        min_x = self.ball.radius
        min_y = self.ball.radius
        clip_x = float(np.clip(self.ball.position[0], min_x, max(max_x - self.ball.radius, min_x)))
        clip_y = float(np.clip(self.ball.position[1], min_y, max(max_y - self.ball.radius, min_y)))
        if not np.isclose(clip_x, float(self.ball.position[0])):
            self.ball.position[0] = clip_x
            self.ball.stop_axis(0)
        if not np.isclose(clip_y, float(self.ball.position[1])):
            self.ball.position[1] = clip_y
            self.ball.stop_axis(1)

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

    def _ball_owner(self) -> Player | None:
        if self.ball.owner_id is None:
            return None
        for player in self.players:
            if player.player_id == self.ball.owner_id:
                return player
        return None

    def _nearest_player(self, target: np.ndarray, team: str) -> Player | None:
        teammates = [player for player in self.players if player.team == team]
        if not teammates:
            return None
        return min(teammates, key=lambda player: player.distance_to(target))

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        width, height = self.params.field_size
        clipped = np.asarray(target, dtype=np.float32).copy()
        clipped[0] = float(np.clip(clipped[0], 0.0, width))
        clipped[1] = float(np.clip(clipped[1], 0.0, height))
        return clipped

    def _check_goal(self, start: np.ndarray, end: np.ndarray) -> str | None:
        width, height = self.params.field_size
        goal_min_y = height * 0.5 - self.params.goal_half_width
        goal_max_y = height * 0.5 + self.params.goal_half_width

        if end[0] + self.ball.radius >= width and goal_min_y <= end[1] <= goal_max_y:
            return "home"
        if end[0] - self.ball.radius <= 0.0 and goal_min_y <= end[1] <= goal_max_y:
            return "away"
        return None

    def _resolve_player_collisions(self, step_events: List[Dict]) -> None:
        min_dist = max(1e-6, 2.0 * self.params.player_collision_radius)
        width, height = self.params.field_size

        for idx in range(len(self.players)):
            for jdx in range(idx + 1, len(self.players)):
                a = self.players[idx]
                b = self.players[jdx]
                delta = b.position - a.position
                dist = float(np.linalg.norm(delta))
                if dist >= min_dist:
                    continue

                if dist < 1e-8:
                    angle = self.rng.uniform(0.0, 2.0 * np.pi)
                    direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                else:
                    direction = delta / dist

                overlap = min_dist - max(dist, 1e-8)
                push = direction * (0.5 * overlap)
                a.position = a.position - push
                b.position = b.position + push
                a.position[0] = float(np.clip(a.position[0], 0.0, width))
                a.position[1] = float(np.clip(a.position[1], 0.0, height))
                b.position[0] = float(np.clip(b.position[0], 0.0, width))
                b.position[1] = float(np.clip(b.position[1], 0.0, height))

                if self.ball.owner_id == a.player_id:
                    self.ball.attach_to(a.player_id, a.position)
                elif self.ball.owner_id == b.player_id:
                    self.ball.attach_to(b.player_id, b.position)

                step_events.append({
                    "event_type": "player_collision_resolved",
                    "players": (a.player_id, b.player_id),
                })

    def _build_state(self) -> Dict:
        return {
            "step": self.step_count,
            "field_size": np.asarray(self.params.field_size, dtype=np.float32),
            "goal_half_width": float(self.params.goal_half_width),
            "ball_position": self.ball.position.copy(),
            "ball_velocity": self.ball.velocity.copy(),
            "ball_owner_id": self.ball.owner_id,
            "ball_dynamics": self.ball.get_dynamics(),
            "goal": self.winner is not None,
            "winner": self.winner,
            "players": [player.copy_public_state() for player in self.players],
        }
