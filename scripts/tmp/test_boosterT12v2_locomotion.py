import isaacgym
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.runner import Runner


if __name__ == "__main__":
    runner = Runner(test=True, task_name="boosterT12v2")
    runner.boosterT12v2Locomotion()
