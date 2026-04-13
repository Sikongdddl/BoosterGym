from __future__ import annotations

import glob
import os
from pathlib import Path


LOW_LEVEL_SEARCH_PATTERNS = (
    "core/checkpoints/low_level/**/*.pth",
    "logs/low/**/*.pth",
)


def find_latest_checkpoint(*patterns: str) -> str | None:
    search_patterns = patterns or LOW_LEVEL_SEARCH_PATTERNS
    candidates: list[str] = []
    for pattern in search_patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_low_level_checkpoint(checkpoint_spec: str | int | None) -> str | None:
    if checkpoint_spec in (None, "", "null"):
        return None
    if checkpoint_spec == "-1" or checkpoint_spec == -1:
        return find_latest_checkpoint(*LOW_LEVEL_SEARCH_PATTERNS)
    return str(checkpoint_spec)


def low_level_model_root() -> Path:
    return Path("core/checkpoints/low_level")
