from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from envs.hyperGym.simulation import HyperGymSimulation


class HyperGymController:
    def __init__(self, simulation: HyperGymSimulation | None = None):
        self.simulation = simulation or HyperGymSimulation()

    def rollout(self, actions: Iterable[Dict]) -> List[Dict]:
        records: List[Dict] = []
        obs = self.simulation.reset()
        for action in actions:
            obs, reward, done, info = self.simulation.step(action)
            records.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
            })
            if done:
                break
        return records


if __name__ == "__main__":
    controller = HyperGymController()
    demo_actions = [
        {"type": "move", "target": np.array([2.0, 3.0], dtype=np.float32)},
        {"type": "pass", "receiver": 1},
        {"type": "move", "target": np.array([4.0, 3.0], dtype=np.float32)},
        {"type": "shoot"},
    ]
    rollout = controller.rollout(demo_actions)
    print(f"rollout_len={len(rollout)}")
    if rollout:
        print(rollout[-1]["info"]["state"])
