# HyperGym HLP

This directory contains the high-level multi-agent RL stack for HyperGym.

## Design

- Observation: same 29-dim team-conditioned state used by the BC policy
- Action: same team-level action structure as BC
  - 2 players
  - each outputs `skill in {move, pass, trap}` and normalized `target(x, y)`
- RL algorithm: hybrid PPO
- Self-play: FSP-style opponent pool
  - train the current best-response policy
  - sample opponents from a mixture of scripted baseline and frozen historical snapshots

## Entrypoint

```bash
python scripts/hlp/train_selfplay.py \
  --updates 400 \
  --save-dir core/checkpoints/high_level/rl/run \
  --bc-init-checkpoint core/checkpoints/high_level/bc/bc_policy_stage1_gpu_resume40/best.pt
```

## Reward

Dense reward is team-relative and goal-oriented. It includes:

- forward ball progress toward the opponent goal
- final-third possession bonus
- trap / pass / move-touch bonuses
- possession gain / loss and turnover shaping
- out-of-bounds penalty
- large terminal goal / concede reward
