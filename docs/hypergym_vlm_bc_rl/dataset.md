# Dataset

## Sources

- Base dataset: VLM-vs-VLM rollouts from normal environment resets
- Scenario dataset: VLM-vs-VLM rollouts from injected mid-episode states

## Format

Both dataset families write compatible `steps.jsonl` files with:

- `state`
- `home_action` / `away_action`
- `active_home_decision` / `active_away_decision`
- `home_should_query` / `away_should_query`
- `reward`, `done`, `winner`

Scenario datasets also include:

- `scenario_family`
- `scenario_description`
- `scenario_tags`

## Filtering

- `fallback` samples should not be used as BC positives
- current BC tooling supports `--only-vlm` to keep only steps whose active decision sources are all `vlm`
