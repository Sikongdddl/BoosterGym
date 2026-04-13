from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Player:
    player_id: str
    team: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    max_speed: float = 0.12
    control_radius: float = 0.18
    has_ball: bool = False
    heading: float = 0.0

    def reset(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.has_ball = False
        self.heading = 0.0 if self.team == "home" else np.pi

    def distance_to(self, target: np.ndarray) -> float:
        delta = np.asarray(target, dtype=np.float32) - self.position
        return float(np.linalg.norm(delta))

    def move_towards(self, target: np.ndarray, dt: float = 1.0, speed_scale: float = 1.0) -> None:
        delta = np.asarray(target, dtype=np.float32) - self.position
        norm = float(np.linalg.norm(delta))
        if norm < 1e-8:
            self.velocity = np.zeros(2, dtype=np.float32)
            return
        direction = delta / norm
        speed = min(self.max_speed * max(0.0, speed_scale), norm / max(dt, 1e-6))
        self.velocity = direction * speed
        if float(np.linalg.norm(self.velocity)) > 1e-8:
            self.heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
        self.position = self.position + self.velocity * dt

    def clamp(self, field_size: tuple[float, float]) -> None:
        width, height = field_size
        self.position[0] = float(np.clip(self.position[0], 0.0, width))
        self.position[1] = float(np.clip(self.position[1], 0.0, height))

    def copy_public_state(self) -> dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "has_ball": self.has_ball,
            "heading": float(self.heading),
        }
