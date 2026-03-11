from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Ball:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    owner_id: Optional[str] = None
    friction: float = 0.94

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
        self.position = self.position + self.velocity * dt
        self.velocity = self.velocity * self.friction

