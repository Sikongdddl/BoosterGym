# BoosterGym

BoosterGym is a research codebase for robot soccer control with a three-layer policy stack:

- Low-level control: robust locomotion and hardware-facing execution
- Mid-level skills: physics-based skill policies such as `chaseBall`, `passBall`, and `trapBall`
- High-level tactics: lightweight multi-agent planning in an abstract soccer environment

The repository is organized around the idea that these three layers should be trained and evaluated in different environments, with different abstractions, but exposed through compatible interfaces.

## Project Status

This codebase is an active research workspace rather than a polished release.

The main implemented directions are:

- Isaac Gym based mid-level skill training
- `hyperGym`, a lightweight 2D soccer simulator for high-level multi-agent reasoning
- VLM-based tactical data collection in `hyperGym`
- behavior cloning from VLM trajectories
- high-level self-play RL initialized from BC

## Three-Layer Architecture

### 1. Low-Level Control

The low-level layer is responsible for stable robot motion and direct deployment to the real platform.

In the current project framing:

- this is the hardware-facing layer
- it should be robust and transferable
- higher layers should depend on commands or subgoals, not raw joint control

### 2. Mid-Level Skills

The mid-level layer operates in physics simulation and learns reusable soccer skills.

Current skill directions include:

- `chaseBall`
- `passBall`
- `trapBall`

This layer is intended to map a skill type plus a target into executable motion behavior in Isaac Gym.

Relevant directories:

- [envs/chaseBall](/home/ubuntu/jrWork/booster_gym/envs/chaseBall)
- [envs/passBall](/home/ubuntu/jrWork/booster_gym/envs/passBall)
- [envs/trapBall](/home/ubuntu/jrWork/booster_gym/envs/trapBall)
- [scripts/train.py](/home/ubuntu/jrWork/booster_gym/scripts/train.py)
- [scripts/runner.py](/home/ubuntu/jrWork/booster_gym/scripts/runner.py)

### 3. High-Level Tactics

The high-level layer reasons over a simplified multi-agent soccer state and decides which skill should be used, and where.

This layer is built around `hyperGym`, a lightweight 2D simulator that removes most low-level contact complexity and focuses on:

- team-level action selection
- passing, trapping, and positioning logic
- self-play and tactical learning
- fast iteration compared with full physics simulation

Current high-level action format:

```python
{
  "skill": "move" | "pass" | "trap",
  "target": np.ndarray(shape=(2,))
}
```

For 2v2 training, the policy takes the full 4-player state and outputs actions for one acting team.

Relevant directories:

- [envs/hyperGym](/home/ubuntu/jrWork/booster_gym/envs/hyperGym)
- [scripts/hlp](/home/ubuntu/jrWork/booster_gym/scripts/hlp)
- [docs/hypergym_vlm_bc_rl](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl)

## Current High-Level Pipeline

The current high-level workflow is:

1. Use VLMs to act in `hyperGym` and collect tactical trajectories.
2. Filter usable VLM decisions and train a state-only BC policy.
3. Use the BC policy as initialization for high-level self-play RL.
4. Improve tactical behavior, especially passing and scoring, through dense reward shaping and FSP-style opponent sampling.

This part of the repository is documented here:

- [dataset.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/dataset.md)
- [bc.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/bc.md)
- [hlp_selfplay.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/hlp_selfplay.md)
- [experiment_log.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/experiment_log.md)

## Repository Layout

Key directories:

- [envs](/home/ubuntu/jrWork/booster_gym/envs): task environments, including Isaac Gym tasks and `hyperGym`
- [core/agents](/home/ubuntu/jrWork/booster_gym/core/agents): reusable RL agents
- [core/imitation](/home/ubuntu/jrWork/booster_gym/core/imitation): BC datasets and models
- [scripts](/home/ubuntu/jrWork/booster_gym/scripts): stable top-level training and inference entrypoints
- [scripts/tmp](/home/ubuntu/jrWork/booster_gym/scripts/tmp): temporary or experiment-specific scripts
- [scripts/hlp](/home/ubuntu/jrWork/booster_gym/scripts/hlp): high-level self-play RL stack
- [docs](/home/ubuntu/jrWork/booster_gym/docs): project notes and experiment documentation
- [deploy](/home/ubuntu/jrWork/booster_gym/deploy): deployment-related code and assets

## Quick Start

### Mid-Level Training

Example:

```bash
python scripts/train.py --checkpoint -1 --task=PassBallEnv
```

### High-Level BC

Example:

```bash
python scripts/tmp/hypergym_vlm_bc_rl/train_bc.py \
  --dataset-roots datasets/vlm_vs_vlm_qwen3vl_2v2_deadball_ep1000 \
  --only-vlm \
  --save-dir core/checkpoints/high_level/bc/run
```

### High-Level Self-Play RL

Example:

```bash
python scripts/hlp/train_selfplay.py \
  --updates 400 \
  --save-dir core/checkpoints/high_level/rl/run \
  --bc-init-checkpoint core/checkpoints/high_level/bc/bc_policy_stage1_gpu_resume40/best.pt
```

## Dependencies

The repository assumes a research environment rather than a fully packaged install.

At minimum:

- Python
- PyTorch
- TensorBoard / WandB depending on workflow
- Isaac Gym for the physics-based tasks

The current [requirements.txt](/home/ubuntu/jrWork/booster_gym/requirements.txt) only covers a small subset of the full environment and should be treated as incomplete.

## Notes

- `datasets/` is ignored by git and is expected to contain large generated artifacts.
- `logs/` contains checkpoints and experiment outputs.
- Many files under [scripts/tmp](/home/ubuntu/jrWork/booster_gym/scripts/tmp) are exploratory and may change quickly.

## Reference Notes

The highest-level project summary and running experiment record are based on:

- [boosterGym-NIPS2026.docx](/home/ubuntu/jrWork/booster_gym/docs/boosterGym-NIPS2026.docx)

For a Chinese summary, see [README-cn.md](/home/ubuntu/jrWork/booster_gym/README-cn.md).
