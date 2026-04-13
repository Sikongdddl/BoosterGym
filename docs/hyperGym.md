# hyperGym Notes

## VLM API

- Date added: 2026-04-13
- Base URL: `https://models.sjtu.edu.cn/api/v1/chat/completions`
- API key: `sk-xUnxK-dbuEeusELX9fXPWA`

## Experiment Log

- Placeholder for future hyperGym experiment results and observations.

## Prompt Backup

- Date updated: 2026-04-13
- Goal: clarify that `pass_to_target` is the environment's generic kick action and should also be used for direct shots on goal.

### `scripts/tmp/vlm_policy_poc.py` / `scripts/tmp/vlm_vs_vlm.py`

```text
Policy semantics:
- move_to_target: use when home/this team should run to space, close down a loose ball, or reposition.
- trap_ball: use when the ball is free or moving and home/this team should first secure control near the ball.
- pass_to_target: this means kicking the ball to a target, not only passing to a teammate. Use it for passes, clearances, and direct shots on goal when the shooter/player can plausibly strike the ball now.
Hard constraints:
- If owner starts with '<team>_', do not output trap_ball because this team already controls the ball.
- If owner is free, prefer move_to_target or trap_ball over pass_to_target.
- If this team already controls the ball, prefer move_to_target for ball progression because dribble is not available in this environment.
- If this team has the ball near goal and the shooting lane is open, prefer pass_to_target aimed inside the goal mouth instead of a harmless extra pass.
- Treat pass_to_target as the only available kick action: if a direct shot is best, encode that shot with pass_to_target.
- Treat the touchlines as dangerous: when the ball is near the top or bottom boundary, avoid targets that keep pushing play along or into the sideline.
- Near a sideline, prefer recycling the ball back inward or switching to safer interior space over continuing a risky edge run.
- Favor targets that progress the attack toward the attack direction.
- Avoid meaningless pass_to_target to the current ball location or to a point behind the attack.
```

## Scenario Table

- Goal: add manually generated mid-episode states so BC sees more high-value situations than default free-ball openings.
- Note: sampling ratios do not need to be fixed now; we can add or rebalance scenario families later.

| Scenario Family | Typical Ball / Possession State | Pressure Pattern | What We Want The Policy To Learn | Useful Positive Actions |
| --- | --- | --- | --- | --- |
| Controlled buildup | `home` or `away` already owns the ball in midfield or half-space | low to medium pressure | advance while keeping structure, select support runner, avoid pointless resets | `move`, `pass` |
| Near-goal finishing | ball carrier already controls near the box or open goal mouth | low or one-sided pressure | treat `pass` as kick, finish directly when lane is open, choose far-post or square-ball target | `pass`, `move` |
| Sideline danger | ball or owner starts near top/bottom boundary | low to high pressure | recycle inward, avoid continuing into touchline, decide who traps and who supports | `move`, `trap`, `pass` |
| Pressured possession | one team owns the ball but nearest defender is already close | medium to high pressure | release the ball under pressure, avoid panic traps, choose safe outlet or quick kick | `pass`, `move` |
| Loose-ball scramble | ball is free, moving slowly or bouncing between two teams | contested | assign one player to secure the ball and the other to cover space instead of both collapsing | `trap`, `move` |
| Second-ball continuation | a prior kick has just happened and the next touch matters | medium pressure | trap the second ball, anticipate next support position, convert partial possession into attack | `trap`, `move`, `pass` |
| Transition attack | a turnover has just created open space toward opponent goal | uneven pressure, backtracking defenders | exploit open field quickly, choose direct kick or support run instead of slowing down | `pass`, `move` |
| Emergency defending | opponent owns the ball in a dangerous zone near our goal | high pressure on defense | mark lane, protect central channel, delay rather than blindly chase | `move`, `trap` |
| Box crowding / rebound | ball is near goal with multiple players nearby and no clean owner | chaotic pressure | distinguish immediate kick chance from necessary trap, avoid both attackers choosing same role | `pass`, `trap`, `move` |
| Dead-ball restart analog | ball is stationary after out-of-bounds-like placement or artificial pause | low pressure but structured setup | restart into useful interior space and create clean first receiving shape | `pass`, `move` |

## Generator Sketch

- Existing hook: [vlm_policy_poc.py](/home/ubuntu/jrWork/booster_gym/scripts/tmp/vlm_policy_poc.py#L561) already has `_set_manual_state(simulation, state_spec)`. This is the right base primitive for scenario injection.
- Proposed new script: `scripts/tmp/hypergym_vlm_bc_rl/generate_scenario_dataset.py`
- High-level flow:
  1. Build a normal `hyperGym` controller with `build_match_controller(...)`.
  2. Sample a scenario family name from a registry.
  3. Generate a `state_spec` for that family with randomized geometry, pressure level, and attack direction.
  4. Call `_set_manual_state(simulation, state_spec)` to jump directly into that state.
  5. Optionally prefill `recent_events` to make the situation legible to the VLM, such as `pass_started`, `ball_control_gained`, `ball_out_of_bounds`, or `turnover`.
  6. Run the same VLM-vs-VLM collection loop used in [generate_vlm_dataset.py](/home/ubuntu/jrWork/booster_gym/scripts/tmp/hypergym_vlm_bc_rl/generate_vlm_dataset.py), so output format stays compatible with current BC tooling.

### Scenario Registry Fields

Each scenario family should define:

- `family_id`: stable string name
- `num_home`, `num_away`
- `ball_position`
- `ball_velocity`
- `ball_owner_id`
- `players`: explicit positions, headings optional
- `recent_events`: optional bootstrapping events
- `constraints`: helper rules such as "owner near goal", "nearest defender within 0.8", or "ball within 0.4 of top sideline"
- `tags`: searchable labels for downstream filtering

### Randomization Axes

Within each family, randomize:

- left/right mirror
- top/bottom mirror
- `home` possession vs `away` possession when the family allows it
- pressure level: none / light / heavy
- support geometry: central / wide / back-post / square-pass lane
- ball speed: stationary / rolling / recently kicked
- defender spacing and lane blockage

### Implementation Notes

- Keep family-specific constraints strict enough that the intended tactical question remains clear.
- Prefer explicit construction over fully unconstrained random placement; otherwise many generated states will collapse back to trivial free-ball openings.
- Save the sampled `family_id`, `pressure_level`, and `mirror_flags` into each step record so BC and later analysis can stratify by scenario source.
- Start by generating short rollouts from injected states rather than full matches; this keeps the dataset focused on the valuable decision window.
