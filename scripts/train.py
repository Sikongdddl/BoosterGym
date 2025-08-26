import isaacgym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.runner import Runner

if __name__ == "__main__":
    runner = Runner(test=False)
    runner.passBall()
