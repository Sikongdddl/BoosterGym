# Low-Level Checkpoints

This directory stores the canonical low-level locomotion checkpoints used by the project.

Why this exists:

- low-level weights are core project assets, not disposable training logs
- `logs/low/` is a poor long-term location because it hides important checkpoints inside experiment output

Current loading behavior:

- `basic.checkpoint: -1` first searches `core/checkpoints/low_level/**/*.pth`
- if nothing is found there, older `logs/low/**/*.pth` files are used as fallback

Recommended layout:

```text
core/checkpoints/low_level/
  2025-06-22-15-47-03/
    config.yaml
    nn/
      model_9600.pth
      model_9700.pth
      model_9800.pth
      model_9900.pth
      model_10000.pth
```
