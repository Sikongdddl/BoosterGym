# Behavior Cloning Policy

## Scope

- First BC version uses state-only input.
- Source environment is `hyperGym`.
- Training data comes from existing `steps.jsonl` rollouts.

## Observation Space

Use the current `hyperGym` state vector built by [data_interface.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/data_interface.py#L16).

### Current Layout

- Per-player features: `6`
- Number of players in `2v2`: `4`
- Ball features: `5`
- Total observation dimension: `29`

### Per-Player Features

For each player, concatenate:

- `x`
- `y`
- `vx`
- `vy`
- `is_controlled_team`
- `has_ball`

Notes:

- `x, y` are normalized by field width and height.
- `vx, vy` are currently unnormalized.
- `is_controlled_team` is `1.0` for `home`, `0.0` otherwise in the current implementation.
- `has_ball` is a binary flag.

### Ball Features

Concatenate:

- `ball_x`
- `ball_y`
- `ball_vx`
- `ball_vy`
- `owner_flag`

Notes:

- `ball_x, ball_y` are normalized by field width and height.
- `owner_flag = -1.0` means free ball.
- `owner_flag = 1.0` means the ball currently has an owner.
- Current `owner_flag` does not distinguish which team owns the ball.

## First BC Policy Plan

First version should predict the current acting team's two actions from this `29`-dim state input.

Recommended first implementation:

- Input: `29`-dim state vector
- Output head 1: discrete skill logits for the acting team's `2` players, each over `move / pass / trap`
- Output head 2: continuous `target_x, target_y` for the acting team's `2` players
- Team conditioning: rebuild the observation with `controlled_team=home` or `controlled_team=away`

## Dataset Wrapper Plan

The dataset wrapper should convert `steps.jsonl` into BC samples of the form:

- `obs`
- `team`
- `action_skill[2]`
- `action_target[2, 2]`
- optional metadata such as `scenario_family`, `source`, `fallback`, `seed`

Suggested first filtering rule:

- Prefer samples where the active decision source is explicitly `vlm`
- Keep fallback metadata so it can be filtered or down-weighted later

## Follow-Up

Current implementation:

- dataset wrapper: [hypergym_vlm_dataset.py](/home/ubuntu/jrWork/booster_gym/core/imitation/datasets/hypergym_vlm_dataset.py)
- BC model: [bc_policy.py](/home/ubuntu/jrWork/booster_gym/core/imitation/models/bc_policy.py)
- training entrypoint: [train_bc.py](/home/ubuntu/jrWork/booster_gym/scripts/tmp/hypergym_vlm_bc_rl/train_bc.py)
