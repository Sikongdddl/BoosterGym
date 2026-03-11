from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TrainingInterface:
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "progress": 0.2,
        "completed_pass": 1.0,
        "intercepted": -1.0,
        "turnover": -0.8,
        "shot": 1.5,
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
            event_type = event["event_type"]
            if event_type == "pass_completed":
                reward += float(self.reward_weights["completed_pass"])
            elif event_type == "intercepted":
                reward += float(self.reward_weights["intercepted"])
            elif event_type == "turnover":
                reward += float(self.reward_weights["turnover"])
            elif event_type == "shot_taken":
                reward += float(self.reward_weights["shot"])
        return reward

    def is_terminal(self, state: Dict) -> bool:
        return bool(state["done"])

