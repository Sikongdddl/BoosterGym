from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

from envs.hyperGym.policies import SimpleMatchPolicy
from envs.hyperGym.renderer import render_episode_mp4
from envs.hyperGym.simulation import HyperGymSimulation, HyperParams


class HyperGymController:
    def __init__(self, simulation: HyperGymSimulation | None = None):
        self.simulation = simulation or HyperGymSimulation()

    def rollout(self, actions: Iterable[Dict]) -> List[Dict]:
        records: List[Dict] = []
        self.simulation.reset()
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

    def collect_episode(self, actions: Iterable[Dict]) -> List[Dict]:
        records: List[Dict] = [{
            "obs": self.simulation.reset(),
            "reward": 0.0,
            "done": False,
            "action": None,
            "info": {
                "events": [],
                "state": self.simulation.get_state(),
            },
        }]
        for action in actions:
            obs, reward, done, info = self.simulation.step(action)
            records.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "action": dict(action),
                "info": info,
            })
            if done:
                break
        return records

    def export_demo_mp4(self, actions: Iterable[Dict], output_path: str | Path, fps: int = 8) -> Path:
        episode = self.collect_episode(actions)
        return render_episode_mp4(
            episode=episode,
            output_path=output_path,
            field_size=self.simulation.params.field_size,
            fps=fps,
        )

    def collect_match_episode(
        self,
        home_policy: Callable[[Dict], Dict],
        away_policy: Callable[[Dict], Dict] | None = None,
        max_steps: int | None = None,
    ) -> List[Dict]:
        if away_policy is not None:
            self.simulation.opponent_policy = away_policy

        records: List[Dict] = [{
            "obs": self.simulation.reset(),
            "reward": 0.0,
            "done": False,
            "action": None,
            "away_action": None,
            "info": {
                "events": [],
                "state": self.simulation.get_state(),
                "actions": {},
            },
        }]
        max_len = max_steps or self.simulation.params.max_steps
        for _ in range(max_len):
            state = self.simulation.get_state()
            home_action = home_policy(state)
            obs, reward, done, info = self.simulation.step(home_action)
            records.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "action": info["actions"]["home"],
                "away_action": info["actions"]["away"],
                "info": info,
            })
            if done:
                break
        return records

    def export_match_mp4(
        self,
        home_policy: Callable[[Dict], Dict],
        away_policy: Callable[[Dict], Dict],
        output_path: str | Path,
        fps: int = 10,
        max_steps: int | None = None,
    ) -> Path:
        episode = self.collect_match_episode(home_policy, away_policy=away_policy, max_steps=max_steps)
        return render_episode_mp4(
            episode=episode,
            output_path=output_path,
            field_size=self.simulation.params.field_size,
            fps=fps,
        )


def build_match_controller(
    num_home: int,
    num_away: int,
    *,
    seed: int,
    max_steps: int,
    wall_restitution: float = 1.0,
    ball_speed: float = 0.55,
    dribble_speed: float = 0.12,
) -> HyperGymController:
    simulation = HyperGymSimulation(
        num_home=num_home,
        num_away=num_away,
        params=HyperParams(
            max_steps=max_steps,
            wall_restitution=wall_restitution,
            ball_speed=ball_speed,
            dribble_speed=dribble_speed,
        ),
        seed=seed,
    )
    return HyperGymController(simulation)


if __name__ == "__main__":
    controller = build_match_controller(num_home=1, num_away=1, seed=3, max_steps=400)
    demo_actions = [
        {"skill": "move", "target": np.array([2.1, 2.6], dtype=np.float32)},
        {"skill": "move", "target": np.array([2.7, 2.8], dtype=np.float32)},
        {"skill": "move", "target": np.array([3.2, 3.1], dtype=np.float32)},
        {"skill": "pass", "target": np.array([4.8, 4.0], dtype=np.float32)},
        {"skill": "trap", "target": np.array([4.8, 4.0], dtype=np.float32)},
        {"skill": "move", "target": np.array([5.8, 3.7], dtype=np.float32)},
        {"skill": "pass", "target": np.array([8.9, 3.0], dtype=np.float32)},
        {"skill": "trap", "target": np.array([8.9, 3.0], dtype=np.float32)},
        {"skill": "move", "target": np.array([9.7, 3.0], dtype=np.float32)},
    ]
    rollout = controller.rollout(demo_actions)
    print(f"rollout_len={len(rollout)}")
    if rollout:
        print(rollout[-1]["info"]["state"])
    output_path = controller.export_demo_mp4(demo_actions, Path("videos") / "hypergym_demo.mp4")
    print(f"saved_video={output_path}")
    home_policy = SimpleMatchPolicy(team="home", seed=12)
    away_policy = SimpleMatchPolicy(team="away", seed=13)
    match_path = controller.export_match_mp4(home_policy, away_policy, Path("videos") / "hypergym_match_demo.mp4", fps=10, max_steps=400)
    print(f"saved_match={match_path}")
