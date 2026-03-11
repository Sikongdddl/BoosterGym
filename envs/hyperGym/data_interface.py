from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DataInterface:
    episode_events: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.episode_events.clear()

    def record(self, step: int, event_type: str, **payload: Any) -> None:
        self.episode_events.append({
            "step": int(step),
            "event_type": event_type,
            **payload,
        })

    def build_observation(
        self,
        players: list,
        ball,
        field_size: tuple[float, float],
        controlled_team: str = "home",
    ) -> np.ndarray:
        width, height = field_size
        denom = np.asarray([max(width, 1e-6), max(height, 1e-6)], dtype=np.float32)
        obs: List[float] = []

        for player in players:
            pos = player.position / denom
            vel = player.velocity
            obs.extend([float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])])
            obs.append(1.0 if player.team == controlled_team else 0.0)
            obs.append(1.0 if player.has_ball else 0.0)

        ball_pos = ball.position / denom
        obs.extend([float(ball_pos[0]), float(ball_pos[1]), float(ball.velocity[0]), float(ball.velocity[1])])
        obs.append(-1.0 if ball.owner_id is None else 1.0)
        return np.asarray(obs, dtype=np.float32)

    def export_episode(self) -> List[Dict[str, Any]]:
        return list(self.episode_events)

