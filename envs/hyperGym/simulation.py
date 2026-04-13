from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

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
    ball_player_restitution: float = 0.35
    ball_player_damping: float = 0.88
    ball_motion_substeps: int = 8
    pass_kick_radius: float = 0.38
    steal_radius: float = 0.2
    shoot_radius: float = 1.2
    trap_radius: float = 0.38
    trap_velocity_scale: float = 0.18
    trap_stop_speed: float = 0.03
    contest_radius: float = 0.55
    trap_safe_speed: float = 0.22
    pass_safe_speed: float = 0.12
    unstable_action_switch_penalty: float = 0.12
    unstable_collision_penalty: float = 0.2
    unstable_contest_penalty: float = 0.35
    unstable_ball_speed_penalty: float = 0.25
    loose_ball_bias: float = 0.7
    loose_ball_speed_range: Tuple[float, float] = (0.12, 0.35)
    dead_ball_speed_scale: float = 0.08
    base_intercept_prob: float = 0.15
    defender_speed_scale: float = 1.0
    defender_press_bias: float = 1.0
    end_on_ball_out: bool = False
    control_capture_radius: float = 0.22
    control_release_radius: float = 0.28
    move_touch_radius: float = 0.24
    move_touch_speed: float = 0.12


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
        self.prev_actions: Dict[str, Dict] = {}
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
        min_gap = self.params.player_collision_radius * 2.4
        placed_positions: List[np.ndarray] = []

        for idx, player in enumerate(self.players):
            if player.team == "home":
                lane_idx = idx
                base_x = 1.0 + 0.7 * lane_idx
                base_y = height * (0.28 + 0.44 * (lane_idx + 1) / (self.num_home + 1))
                x_low, x_high = 0.7, min(width * 0.42, base_x + 0.45)
            else:
                away_idx = idx - self.num_home
                base_x = width * 0.64 + 0.6 * away_idx
                base_y = height * (0.24 + 0.52 * (away_idx + 1) / (self.num_away + 1))
                x_low, x_high = max(width * 0.58, base_x - 0.45), width - 0.7

            y_low = 0.5
            y_high = height - 0.5
            candidate = np.asarray([base_x, base_y], dtype=np.float32)
            for _ in range(24):
                sampled = np.asarray([
                    self.rng.uniform(x_low, x_high),
                    self.rng.uniform(y_low, y_high),
                ], dtype=np.float32)
                if all(float(np.linalg.norm(sampled - existing)) >= min_gap for existing in placed_positions):
                    candidate = sampled
                    break
            player.reset(candidate)
            placed_positions.append(candidate.copy())

        start_ball = np.asarray([
            self.rng.uniform(width * 0.28, width * 0.72),
            self.rng.uniform(height * 0.2, height * 0.8),
        ], dtype=np.float32)
        for _ in range(24):
            if all(float(np.linalg.norm(start_ball - pos)) >= min_gap * 0.75 for pos in placed_positions):
                break
            start_ball = np.asarray([
                self.rng.uniform(width * 0.28, width * 0.72),
                self.rng.uniform(height * 0.2, height * 0.8),
            ], dtype=np.float32)
        start_ball = self._clip_ball_position(start_ball)
        self.ball.reset(start_ball, None)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.winner = None
        self.prev_actions = {}
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

    def step(self, action: Dict[str, Any], opponent_action: Dict[str, Any] | None = None) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        step_events: List[Dict] = []
        # IMPORTANT: keep ball control as a weak label only.
        # Downstream VLM experiments assume the ball is never hard-attached to a player during normal updates.
        self._refresh_ball_control(step_events, emit_events=False)

        home_action = self._normalize_actions(action, team="home")
        if opponent_action is None and self.opponent_policy is not None:
            opponent_action = self.opponent_policy(self.get_state())
        away_action = self._normalize_actions(opponent_action, team="away")

        self.prev_actions = dict(self.last_actions)
        self.last_actions = {**home_action, **away_action}
        # IMPORTANT: this update order is deliberate.
        # Actions may add impulse-like ball interactions first; physics, contest resolution, and control refresh happen afterwards.
        # Reordering these stages easily reintroduces sticky control or ball teleport artifacts.
        self._apply_team_actions(home_action, team="home", step_events=step_events)
        self._apply_team_actions(away_action, team="away", step_events=step_events)
        self._resolve_player_collisions(step_events)
        self._resolve_ball_motion(step_events)
        self._resolve_possession(step_events)
        self._resolve_scramble(step_events)
        self._resolve_player_collisions(step_events)
        self._refresh_ball_control(step_events, emit_events=True)

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

    def _normalize_single_action(self, action: Dict | None, team: str) -> Dict:
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

    def _normalize_actions(self, action: Dict[str, Any] | None, team: str) -> Dict[str, Dict]:
        teammates = [player for player in self.players if player.team == team]
        if action is None:
            return {
                player.player_id: self._normalize_single_action(None, team=team)
                for player in teammates
            }

        if "players" in action and isinstance(action["players"], dict):
            raw_actions = action["players"]
            normalized: Dict[str, Dict] = {}
            for player in teammates:
                if player.player_id in raw_actions:
                    normalized[player.player_id] = self._normalize_single_action(raw_actions.get(player.player_id), team=team)
                else:
                    normalized[player.player_id] = self._idle_action(player)
            return normalized

        if any(player.player_id in action for player in teammates):
            normalized = {}
            for player in teammates:
                if player.player_id in action:
                    normalized[player.player_id] = self._normalize_single_action(action.get(player.player_id), team=team)
                else:
                    normalized[player.player_id] = self._idle_action(player)
            return normalized

        legacy_action = self._normalize_single_action(action, team=team)
        active_player = self._select_legacy_executor(team=team, action=legacy_action)
        normalized = {}
        for player in teammates:
            if active_player is not None and player.player_id == active_player.player_id:
                normalized[player.player_id] = legacy_action
            else:
                normalized[player.player_id] = {
                    "skill": "move",
                    "target": player.position.copy(),
                }
        return normalized

    def _idle_action(self, player: Player) -> Dict:
        return {
            "skill": "move",
            "target": player.position.copy(),
        }

    def _select_legacy_executor(self, team: str, action: Dict) -> Player | None:
        skill = action.get("skill", "move")
        target = self._clip_target(np.asarray(action.get("target", self.ball.position), dtype=np.float32))
        owner = self._ball_owner()
        team_owner = owner if owner is not None and owner.team == team else None

        if skill == "pass":
            return team_owner or self._nearest_player(self.ball.position, team=team)
        if skill == "dribble":
            return team_owner or self._nearest_player(self.ball.position, team=team)
        if skill == "trap":
            return self._nearest_player(target, team=team)
        return team_owner or self._nearest_player(target, team=team)

    def _apply_team_actions(self, actions: Dict[str, Dict], team: str, step_events: List[Dict]) -> None:
        teammates = [player for player in self.players if player.team == team]
        for player in teammates:
            action = actions.get(player.player_id)
            if action is None:
                continue
            self._apply_player_action(player, action, step_events)

    def _apply_player_action(self, actor: Player, action: Dict, step_events: List[Dict]) -> None:
        skill = action.get("skill", "move")
        target = self._clip_target(np.asarray(action.get("target", self.ball.position), dtype=np.float32))
        team = actor.team
        owner = self._ball_owner()

        if skill == "dribble" and not self.params.enable_dribble:
            skill = "move"

        if skill == "pass":
            if actor.distance_to(self.ball.position) > self._ball_interaction_radius(self.params.pass_kick_radius):
                return
            actor.has_ball = False
            if owner is not None and owner.player_id == actor.player_id:
                self.ball.owner_id = None
            if self._should_lose_ball(actor, action_type="pass", step_events=step_events):
                self._apply_loose_ball(actor, target, team, step_events)
            else:
                self.ball.release_towards(target, self.params.ball_speed)
                step_events.append({
                    "event_type": "pass_started",
                    "team": team,
                    "from": actor.player_id,
                    "target": target.copy(),
                })
            return

        if skill == "dribble":
            if owner is None or owner.player_id != actor.player_id:
                actor.move_towards(self.ball.position)
                actor.clamp(self.params.field_size)
                step_events.append({"event_type": "move", "team": team, "by": actor.player_id, "target": self.ball.position.copy()})
                return
            actor.move_towards(target, speed_scale=self.params.dribble_speed / max(actor.max_speed, 1e-6))
            actor.clamp(self.params.field_size)
            self.ball.attach_to(actor.player_id, actor.position)
            step_events.append({"event_type": "dribble", "team": team, "by": actor.player_id, "target": target.copy()})
            return

        if skill == "trap":
            actor.move_towards(target)
            actor.clamp(self.params.field_size)
            if self.ball.owner_id is None and actor.distance_to(self.ball.position) <= self._ball_interaction_radius(self.params.trap_radius):
                if self._should_lose_ball(actor, action_type="trap", step_events=step_events):
                    self._apply_loose_ball(actor, self.ball.position.copy(), team, step_events)
                else:
                    self.ball.velocity = self.ball.velocity * self.params.trap_velocity_scale
                    speed = float(np.linalg.norm(self.ball.velocity))
                    if speed < self.params.trap_stop_speed:
                        self.ball.velocity[:] = 0.0
                    step_events.append({
                        "event_type": "trap_completed",
                        "team": team,
                        "by": actor.player_id,
                        "ball_speed": float(np.linalg.norm(self.ball.velocity)),
                    })
            else:
                step_events.append({"event_type": "trap_attempt", "team": team, "by": actor.player_id, "target": target.copy()})
            return

        actor.move_towards(target)
        actor.clamp(self.params.field_size)
        # IMPORTANT: move may nudge the ball, but should never "own" it by snapping it onto the player.
        # This keeps possession changes observable as collisions/impulses instead of trivial sticky dribbling.
        if self._can_attempt_move_touch(actor):
            if self._should_lose_ball(actor, action_type="move", step_events=step_events):
                self._apply_loose_ball(actor, target, team, step_events)
            else:
                self._apply_move_touch(actor, target, team, step_events)
        step_events.append({"event_type": "move", "team": team, "by": actor.player_id, "target": target.copy()})

    def _resolve_ball_motion(self, step_events: List[Dict]) -> None:
        damping_rate = self.ball.linear_damping / max(self.ball.mass, 1e-6)
        substeps = max(1, int(self.params.ball_motion_substeps))
        sub_dt = self.params.sim_dt / substeps
        decay = float(np.exp(-damping_rate * sub_dt)) if damping_rate > 1e-8 else 1.0

        for _ in range(substeps):
            self.ball.position = self.ball.position + self.ball.velocity * sub_dt

            scorer = self._check_goal(self.ball.position, self.ball.position)
            if scorer is not None:
                self.winner = scorer
                step_events.append({"event_type": "goal_scored", "team": scorer})
                return

            if self.params.end_on_ball_out:
                # IMPORTANT: with dead-ball enabled, the ball should move freely and only be terminated by
                # the out-of-bounds rule itself. Do not let wall bounce happen first.
                out_info = self._check_ball_out(self.ball.position)
                if out_info is not None:
                    self.winner = "ball_out"
                    step_events.append(out_info)
                    return

            if not self.params.end_on_ball_out:
                self._resolve_wall_collision(step_events)
            self._resolve_ball_player_collisions(step_events)
            self.ball.velocity = self.ball.velocity * decay

        if float(np.linalg.norm(self.ball.velocity)) < 1e-4:
            self.ball.velocity[:] = 0.0

    def _resolve_possession(self, step_events: List[Dict]) -> None:
        # Possession is inferred from loose-ball contact rather than hard attachment.
        owner = self._ball_owner()
        if owner is None or float(np.linalg.norm(self.ball.velocity)) > self.params.trap_safe_speed:
            return

        nearby = [
            player
            for player in self.players
            if player.team != owner.team and player.distance_to(self.ball.position) <= self.params.steal_radius
        ]
        if nearby and self.rng.random() < 0.08 * self.params.defender_press_bias:
            thief = nearby[0]
            self.ball.owner_id = thief.player_id
            step_events.append({"event_type": "steal_attempt_won", "from_player": owner.player_id, "to_player": thief.player_id})

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

    def _clip_ball_position(self, position: np.ndarray) -> np.ndarray:
        width, height = self.params.field_size
        clipped = np.asarray(position, dtype=np.float32).copy()
        clipped[0] = float(np.clip(clipped[0], self.ball.radius, width - self.ball.radius))
        clipped[1] = float(np.clip(clipped[1], self.ball.radius, height - self.ball.radius))
        return clipped

    def _ball_interaction_radius(self, margin: float) -> float:
        return max(
            float(margin),
            float(self.params.player_collision_radius + self.ball.radius + 0.01),
        )

    def _check_goal(self, start: np.ndarray, end: np.ndarray) -> str | None:
        width, height = self.params.field_size
        goal_min_y = height * 0.5 - self.params.goal_half_width
        goal_max_y = height * 0.5 + self.params.goal_half_width

        if end[0] + self.ball.radius >= width and goal_min_y <= end[1] <= goal_max_y:
            return "home"
        if end[0] - self.ball.radius <= 0.0 and goal_min_y <= end[1] <= goal_max_y:
            return "away"
        return None

    def _check_ball_out(self, position: np.ndarray) -> Dict[str, Any] | None:
        width, height = self.params.field_size
        x = float(position[0])
        y = float(position[1])
        r = float(self.ball.radius)
        goal_min_y = height * 0.5 - self.params.goal_half_width
        goal_max_y = height * 0.5 + self.params.goal_half_width
        in_goal_mouth = goal_min_y <= y <= goal_max_y
        if x < -r and not in_goal_mouth:
            return {"event_type": "ball_out_of_bounds", "side": "left", "position": np.asarray(position, dtype=np.float32).copy()}
        if x > width + r and not in_goal_mouth:
            return {"event_type": "ball_out_of_bounds", "side": "right", "position": np.asarray(position, dtype=np.float32).copy()}
        if y < -r:
            return {"event_type": "ball_out_of_bounds", "side": "bottom", "position": np.asarray(position, dtype=np.float32).copy()}
        if y > height + r:
            return {"event_type": "ball_out_of_bounds", "side": "top", "position": np.asarray(position, dtype=np.float32).copy()}
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

                step_events.append({
                    "event_type": "player_collision_resolved",
                    "players": (a.player_id, b.player_id),
                })

    def _resolve_wall_collision(self, step_events: List[Dict]) -> None:
        width, height = self.params.field_size
        min_x = self.ball.radius
        max_x = width - self.ball.radius
        min_y = self.ball.radius
        max_y = height - self.ball.radius
        clip_x = float(np.clip(self.ball.position[0], min_x, max_x))
        clip_y = float(np.clip(self.ball.position[1], min_y, max_y))
        if not np.isclose(clip_x, float(self.ball.position[0])):
            self.ball.position[0] = clip_x
            self.ball.stop_axis(0)
            step_events.append({"event_type": "ball_wall_collision", "axis": 0})
        if not np.isclose(clip_y, float(self.ball.position[1])):
            self.ball.position[1] = clip_y
            self.ball.stop_axis(1)
            step_events.append({"event_type": "ball_wall_collision", "axis": 1})

    def _resolve_ball_player_collisions(self, step_events: List[Dict]) -> None:
        min_dist = self.ball.radius + self.params.player_collision_radius
        for player in self.players:
            delta = self.ball.position - player.position
            dist = float(np.linalg.norm(delta))
            if dist >= min_dist:
                continue

            if dist < 1e-8:
                vel_norm = float(np.linalg.norm(self.ball.velocity))
                normal = self.ball.velocity / vel_norm if vel_norm > 1e-8 else np.asarray([1.0, 0.0], dtype=np.float32)
            else:
                normal = delta / dist

            self.ball.position = player.position + normal * min_dist
            normal_speed = float(np.dot(self.ball.velocity, normal))
            tangential = self.ball.velocity - normal_speed * normal
            if normal_speed < 0.0:
                reflected = -normal_speed * self.params.ball_player_restitution
                self.ball.velocity = tangential * self.params.ball_player_damping + normal * reflected
            else:
                self.ball.velocity = tangential * self.params.ball_player_damping

            if float(np.linalg.norm(self.ball.velocity)) < self.params.trap_stop_speed:
                self.ball.velocity[:] = 0.0

            step_events.append({"event_type": "ball_player_collision", "player": player.player_id})

    def _should_lose_ball(self, actor: Player, action_type: str, step_events: List[Dict]) -> bool:
        risk = 0.0
        team = actor.team
        opponents = [player for player in self.players if player.team != team]
        nearest_opp_dist = min((player.distance_to(self.ball.position) for player in opponents), default=999.0)
        if nearest_opp_dist < self.params.contest_radius:
            risk += self.params.unstable_contest_penalty

        ball_speed = float(np.linalg.norm(self.ball.velocity))
        safe_speed = self.params.pass_safe_speed if action_type == "pass" else self.params.trap_safe_speed
        if ball_speed > safe_speed:
            risk += self.params.unstable_ball_speed_penalty

        last_action = self.prev_actions.get(actor.player_id, {})
        if last_action and last_action.get("skill") not in {action_type, "move"}:
            risk += self.params.unstable_action_switch_penalty

        if any(evt.get("event_type") == "player_collision_resolved" and actor.player_id in evt.get("players", ()) for evt in step_events):
            risk += self.params.unstable_collision_penalty

        risk = float(np.clip(risk, 0.0, 0.92))
        return bool(risk > 0.0 and self.rng.random() < risk)

    def _apply_loose_ball(self, actor: Player, target: np.ndarray, team: str, step_events: List[Dict]) -> None:
        target_dir = np.asarray(target, dtype=np.float32) - actor.position
        norm = float(np.linalg.norm(target_dir))
        if norm < 1e-8:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            direction = target_dir / norm

        if self.rng.random() < self.params.loose_ball_bias:
            angle_jitter = self.rng.uniform(-0.9, 0.9)
            rot = np.asarray(
                [
                    [np.cos(angle_jitter), -np.sin(angle_jitter)],
                    [np.sin(angle_jitter), np.cos(angle_jitter)],
                ],
                dtype=np.float32,
            )
            loose_dir = rot @ direction
            loose_speed = self.rng.uniform(*self.params.loose_ball_speed_range)
            # IMPORTANT: only perform minimal separation before applying new velocity.
            # Large position jumps here look like ball teleportation on possession changes.
            self._separate_ball_from_actor(actor, loose_dir)
            self.ball.velocity = loose_dir * float(loose_speed)
            step_events.append({"event_type": "loose_ball", "team": team, "by": actor.player_id, "ball_speed": float(np.linalg.norm(self.ball.velocity))})
        else:
            speed = float(np.linalg.norm(self.ball.velocity))
            self._separate_ball_from_actor(actor, direction)
            self.ball.velocity = direction * speed * self.params.dead_ball_speed_scale
            if float(np.linalg.norm(self.ball.velocity)) < self.params.trap_stop_speed:
                self.ball.velocity[:] = 0.0
            step_events.append({"event_type": "dead_ball", "team": team, "by": actor.player_id, "ball_speed": float(np.linalg.norm(self.ball.velocity))})

    def _apply_move_touch(self, actor: Player, target: np.ndarray, team: str, step_events: List[Dict]) -> None:
        touch_dir = np.asarray(target, dtype=np.float32) - self.ball.position
        norm = float(np.linalg.norm(touch_dir))
        if norm < 1e-8:
            touch_dir = np.asarray([1.0 if team == "home" else -1.0, 0.0], dtype=np.float32)
        else:
            touch_dir = touch_dir / norm
        self.ball.owner_id = None
        current_speed = float(np.linalg.norm(self.ball.velocity))
        touch_speed = max(current_speed, self.params.move_touch_speed)
        # IMPORTANT: this is an impulse-style touch, not a reposition-to-foot operation.
        # Keep separation minimal and let subsequent physics move the ball.
        self._separate_ball_from_actor(actor, touch_dir)
        self.ball.velocity = touch_dir * float(touch_speed)
        step_events.append({
            "event_type": "move_touch",
            "team": team,
            "by": actor.player_id,
            "ball_speed": float(np.linalg.norm(self.ball.velocity)),
        })

    def _resolve_scramble(self, step_events: List[Dict]) -> None:
        # IMPORTANT: scramble resolution exists to break unrealistic "dogfights" where several players
        # grind around a nearly-static ball forever. Preserve the loose-ball nature of the outcome.
        if float(np.linalg.norm(self.ball.velocity)) > self.params.trap_stop_speed:
            return
        nearby = [
            player
            for player in self.players
            if player.distance_to(self.ball.position) <= self.params.contest_radius
        ]
        if len(nearby) < 2:
            return
        teams = {player.team for player in nearby}
        if len(teams) < 2:
            return
        centroid = np.mean([player.position for player in nearby], axis=0)
        field_center = np.asarray(
            [0.5 * self.params.field_size[0], 0.5 * self.params.field_size[1]],
            dtype=np.float32,
        )
        escape_dir = field_center - centroid
        norm = float(np.linalg.norm(escape_dir))
        if norm < 1e-8:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            escape_dir = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            escape_dir = escape_dir / norm
        angle_jitter = self.rng.uniform(-0.6, 0.6)
        rot = np.asarray(
            [
                [np.cos(angle_jitter), -np.sin(angle_jitter)],
                [np.sin(angle_jitter), np.cos(angle_jitter)],
            ],
            dtype=np.float32,
        )
        escape_dir = rot @ escape_dir
        escape_speed = float(self.rng.uniform(max(self.params.trap_safe_speed, 0.14), max(self.params.trap_safe_speed + 0.08, 0.24)))
        self.ball.owner_id = None
        for player in self.players:
            player.has_ball = False
        anchor = min(nearby, key=lambda player: player.distance_to(self.ball.position))
        self._separate_ball_from_actor(anchor, escape_dir)
        self.ball.velocity = escape_dir * escape_speed
        step_events.append({
            "event_type": "loose_ball_scramble",
            "players": [player.player_id for player in nearby],
            "ball_speed": float(np.linalg.norm(self.ball.velocity)),
        })

    def _can_attempt_move_touch(self, actor: Player) -> bool:
        if actor.distance_to(self.ball.position) > self._ball_interaction_radius(self.params.move_touch_radius):
            return False
        distances = sorted(
            (player.distance_to(self.ball.position), player.player_id)
            for player in self.players
        )
        if not distances:
            return False
        best_dist, best_id = distances[0]
        actor_dist = actor.distance_to(self.ball.position)
        if best_id == actor.player_id:
            return True
        return actor_dist - best_dist <= 0.03 and best_id.startswith(actor.team)

    def _separate_ball_from_actor(self, actor: Player, preferred_dir: np.ndarray) -> None:
        # IMPORTANT: this helper is only for overlap resolution.
        # It must never be turned into a generic "place ball in front of player" utility.
        min_dist = self.ball.radius + self.params.player_collision_radius + 0.01
        delta = self.ball.position - actor.position
        dist = float(np.linalg.norm(delta))
        if dist >= min_dist:
            return
        if dist > 1e-8:
            direction = delta / dist
        else:
            direction = np.asarray(preferred_dir, dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.asarray([1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
        separated = actor.position + direction * min_dist
        if self.params.end_on_ball_out:
            self.ball.position = np.asarray(separated, dtype=np.float32)
        else:
            self.ball.position = self._clip_ball_position(separated)

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

    def _refresh_ball_control(self, step_events: List[Dict], *, emit_events: bool) -> None:
        # IMPORTANT: ball_owner_id / has_ball are weak observational labels for policies and rendering.
        # They are not allowed to drive the ball's physical position, otherwise the simulator collapses back
        # to the old sticky-control regime that invalidated prior experiments.
        previous_owner_id = self.ball.owner_id
        previous_owner = self._ball_owner()
        ball_speed = float(np.linalg.norm(self.ball.velocity))
        if previous_owner is not None:
            release_radius = self._ball_interaction_radius(self.params.control_release_radius)
            if previous_owner.distance_to(self.ball.position) > release_radius or ball_speed > self.params.trap_safe_speed:
                self.ball.owner_id = None

        if self.ball.owner_id is None and ball_speed <= self.params.trap_stop_speed:
            capture_radius = self._ball_interaction_radius(self.params.control_capture_radius)
            candidates = [
                player
                for player in self.players
                if player.distance_to(self.ball.position) <= capture_radius
            ]
            if candidates:
                candidates.sort(key=lambda player: (player.distance_to(self.ball.position), 0 if player.team == "home" else 1))
                best = candidates[0]
                if len(candidates) == 1:
                    self.ball.owner_id = best.player_id
                else:
                    second_dist = candidates[1].distance_to(self.ball.position)
                    if second_dist - best.distance_to(self.ball.position) > 0.08:
                        self.ball.owner_id = best.player_id

        for player in self.players:
            player.has_ball = self.ball.owner_id == player.player_id

        if emit_events and previous_owner_id != self.ball.owner_id:
            if previous_owner_id is not None and self.ball.owner_id is None:
                step_events.append({"event_type": "ball_control_lost", "from_player": previous_owner_id})
            elif previous_owner_id is None and self.ball.owner_id is not None:
                step_events.append({"event_type": "ball_control_gained", "to_player": self.ball.owner_id})
            elif previous_owner_id is not None and self.ball.owner_id is not None:
                step_events.append({
                    "event_type": "turnover",
                    "from_player": previous_owner_id,
                    "to_player": self.ball.owner_id,
                })
