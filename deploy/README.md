# Deploy on Booster Robot

## Installation

Follow these steps to set up your environment:

1. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

2. Install Booster Robotic SDK:

    Refer to the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-WDzedC8AiovU8gxSjeGcQ5CInSf) and ensure you complete the section on [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5JxnCf).

## Usage

1. Prepare the robot:

    - **Simulation:** Set up the simulation by referring to [Development Based on Webots Simulation](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-IsE9d2DrIow8tpxCBUUcogdwn5d) or [Development Based on Isaac Simulation Link](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-Jczjd4UKMou7QlxjvJ4c9NNfnwb).

    - **Real World:** Power on the robot and switch it to PREP Mode. Place the robot to a stable standing position on the ground.

2. Run the deployment script:

    ```sh
    $ python deploy.py --config=T1.yaml
    ```

    - `--config`: Name of the configuration file, located in the `configs/` folder.
    - `--net`: Network interface for SDK communication. Default is `127.0.0.1`.

3. Exit Safely:

    Switch back to PREP Mode before terminating the program to safely release control.
This directory is reserved for the legacy deployment path and real-robot facing assets.

Reusable project checkpoints now live under:

- [core/checkpoints](/home/ubuntu/jrWork/booster_gym/core/checkpoints)
