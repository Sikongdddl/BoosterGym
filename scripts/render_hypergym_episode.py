from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.hyperGym.main import HyperGymController
from envs.hyperGym.main import build_match_controller
from envs.hyperGym.policies import SimpleMatchPolicy


def main() -> None:
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
    output = controller.export_demo_mp4(demo_actions, Path("videos") / "hypergym_demo.mp4")
    print(output)
    match_output = controller.export_match_mp4(
        home_policy=SimpleMatchPolicy(team="home", seed=12),
        away_policy=SimpleMatchPolicy(team="away", seed=13),
        output_path=Path("videos") / "hypergym_match_demo.mp4",
        fps=10,
        max_steps=400,
    )
    print(match_output)

    controller_2v2 = build_match_controller(num_home=2, num_away=2, seed=3, max_steps=500)
    match_2v2_output = controller_2v2.export_match_mp4(
        home_policy=SimpleMatchPolicy(team="home", seed=21),
        away_policy=SimpleMatchPolicy(team="away", seed=32),
        output_path=Path("videos") / "hypergym_match_2v2_demo.mp4",
        fps=10,
        max_steps=500,
    )
    print(match_2v2_output)


if __name__ == "__main__":
    main()
