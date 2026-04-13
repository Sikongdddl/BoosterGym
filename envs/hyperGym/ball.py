from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Ball:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    owner_id: Optional[str] = None
    radius: float = 0.11
    density: float = 80.0
    linear_damping: float = 0.015
    angular_damping: float = 0.01
    default_z: float = 0.12
    restitution: float = 0.0

    @property
    def mass(self) -> float:
        volume = (4.0 / 3.0) * np.pi * (self.radius ** 3)
        return float(self.density * volume)

    def reset(self, position: np.ndarray, owner_id: Optional[str] = None) -> None:
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.owner_id = owner_id

    def attach_to(self, owner_id: str, owner_position: np.ndarray) -> None:
        self.owner_id = owner_id
        self.position = np.asarray(owner_position, dtype=np.float32).copy()
        self.velocity = np.zeros(2, dtype=np.float32)

    def release_towards(self, target: np.ndarray, speed: float) -> None:
        delta = np.asarray(target, dtype=np.float32) - self.position
        norm = float(np.linalg.norm(delta))
        if norm < 1e-8:
            self.velocity = np.zeros(2, dtype=np.float32)
        else:
            self.velocity = delta / norm * float(speed)
        self.owner_id = None

    def step_free(self, dt: float = 1.0) -> None:
        if self.owner_id is not None:
            self.velocity = np.zeros(2, dtype=np.float32)
            return

        damping_rate = self.linear_damping / max(self.mass, 1e-6)
        if damping_rate <= 1e-8:
            self.position = self.position + self.velocity * dt
            return

        decay = float(np.exp(-damping_rate * dt))
        displacement_scale = (1.0 - decay) / damping_rate
        self.position = self.position + self.velocity * displacement_scale
        self.velocity = self.velocity * decay

    def stop_axis(self, axis: int) -> None:
        self.velocity[axis] = -self.velocity[axis] * float(self.restitution)

    def get_dynamics(self) -> dict:
        return {
            "radius": float(self.radius),
            "density": float(self.density),
            "linear_damping": float(self.linear_damping),
            "angular_damping": float(self.angular_damping),
            "default_z": float(self.default_z),
            "restitution": float(self.restitution),
            "mass": float(self.mass),
        }
