import isaacgym
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.runner import Runner


def _task_route(task_name: str):
    mapping = {
        "ChaseBallEnv": ("chaseBall", "chaseBall"),
        "DribbleBallEnv": ("dribbleBall", "dribbleBall"),
        "PassBallEnv": ("passBall", "passBall"),
        "TrapBallEnv": ("trapBall", "trapBall"),
    }
    if task_name not in mapping:
        raise ValueError(
            f"Unsupported --task {task_name}. "
            f"Expected one of: {', '.join(mapping.keys())}"
        )
    return mapping[task_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    args, _ = parser.parse_known_args()

    task_dir, method_name = _task_route(args.task)
    runner = Runner(test=False, task_name=task_dir)
    getattr(runner, method_name)()
