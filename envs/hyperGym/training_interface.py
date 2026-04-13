from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TrainingInterface:
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "progress": 0.2,
        "trap": 0.15,
        "pass": 0.4,
        "goal": 2.0,
        "step_cost": -0.01,
    })
    _last_ball_x: float = 0.0

    def reset(self, ball_position: np.ndarray) -> None:
        self._last_ball_x = float(ball_position[0])

    def compute_reward(self, state: Dict, events: List[Dict]) -> float:
        reward = float(self.reward_weights["step_cost"])
        current_ball_x = float(state["ball_position"][0])
        reward += (current_ball_x - self._last_ball_x) * float(self.reward_weights["progress"])
        self._last_ball_x = current_ball_x
        for event in events:
            if event["event_type"] == "trap_completed" and event.get("team") == "home":
                reward += float(self.reward_weights["trap"])
            elif event["event_type"] == "pass_started" and event.get("team") == "home":
                reward += float(self.reward_weights["pass"])
            elif event["event_type"] == "goal_scored":
                reward += float(self.reward_weights["goal"] if event.get("team") == "home" else -self.reward_weights["goal"])
        return reward
