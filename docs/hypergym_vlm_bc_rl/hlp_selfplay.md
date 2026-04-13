# HLP Self-Play

## Goal

Train a high-level multi-agent policy for `hyperGym` that starts from the BC initialization and improves through adversarial RL.

The target behavior is:

- keep the BC prior for basic positioning, ball approach, and trapping
- learn stronger `pass` usage and near-goal finishing through reward shaping and self-play

## Observation

Use the same team-conditioned state observation as BC.

- dimension: `29`
- input contains all 4 players and the ball
- controlled-team identity is encoded in the observation

Reference:

- [data_interface.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/data_interface.py)
- [hypergym_vlm_dataset.py](/home/ubuntu/jrWork/booster_gym/core/imitation/datasets/hypergym_vlm_dataset.py)

## Action

Use the same team-level action parameterization as BC.

For the acting team:

- 2 players
- each player outputs one discrete skill:
  - `move`
  - `pass`
  - `trap`
- each player also outputs one continuous normalized target:
  - `target_x`
  - `target_y`

The normalized target is decoded back into absolute field coordinates before calling the simulator.

Reference:

- [env.py](/home/ubuntu/jrWork/booster_gym/scripts/hlp/env.py)

## RL Design

Current implementation uses a hybrid PPO actor-critic.

- actor head 1: categorical skill distribution for both team players
- actor head 2: Gaussian target distribution for both team players
- critic head: scalar state value

The actor can be initialized from the BC checkpoint:

- backbone weights are reused
- BC `skill_head` is reused
- BC `target_head` is mapped into the RL actor `target_mean_head`

Reference:

- [policy.py](/home/ubuntu/jrWork/booster_gym/scripts/hlp/policy.py)

## Self-Play

Self-play follows an FSP-style idea.

The training policy is the current best-response policy.

Opponents are sampled from a mixture of:

- a scripted baseline
- frozen historical snapshots of older policies

This avoids training only against the latest self clone and gives a more stable opponent distribution.

Reference:

- [fsp.py](/home/ubuntu/jrWork/booster_gym/scripts/hlp/fsp.py)

## Reward

Reward is dense and team-relative. The design principle is:

- encourage progress toward scoring
- encourage useful possession and ball advancement
- discourage losing the ball or wasting attacks
- keep a large sparse incentive for actual goals

Current shaping terms:

- `step_cost`
- `ball_progress`
- `final_third_possession`
- `trap_completed`
- `pass_started`
- `move_touch`
- `ball_control_gained`
- `ball_control_lost`
- `turnover_won`
- `turnover_lost`
- `steal_won`
- `dead_ball`
- `loose_ball`
- `ball_out_of_bounds`
- `goal_scored`
- `goal_conceded`

Reference:

- [reward.py](/home/ubuntu/jrWork/booster_gym/scripts/hlp/reward.py)

## Entrypoint

Training entrypoint:

- [train_selfplay.py](/home/ubuntu/jrWork/booster_gym/scripts/hlp/train_selfplay.py)

Example:

```bash
python scripts/hlp/train_selfplay.py \
  --updates 400 \
  --save-dir core/checkpoints/high_level/rl/run \
  --bc-init-checkpoint core/checkpoints/high_level/bc/bc_policy_stage1_gpu_resume40/best.pt
```

## Current Status

Implemented:

- team-level environment wrapper
- team-relative dense reward shaping
- hybrid actor-critic
- FSP-style opponent pool
- BC warm start support
- hybrid PPO training loop

Smoke-tested:

- one short run completed successfully
- checkpoints and logs were written to `/tmp/hlp_smoke`
