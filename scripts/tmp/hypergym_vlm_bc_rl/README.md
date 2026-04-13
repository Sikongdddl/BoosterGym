# HyperGym VLM -> BC -> RL

This directory contains the thin CLI entrypoints for the HyperGym pipeline:

- `generate_vlm_dataset.py`: collect VLM-vs-VLM rollouts from normal environment resets
- `generate_scenario_dataset.py`: collect VLM-vs-VLM rollouts from injected scenario states
- `eval_vlm_scenarios.py`: define scenario families and compare VLM decisions with scripted policy behavior
- `train_bc.py`: train the state-only behavior cloning policy
