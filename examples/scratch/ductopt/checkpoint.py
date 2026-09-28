"""Tiny JSON checkpoint helper so the Stage 2 pipeline can run as many small,
resumable stages (this sandbox does not keep background processes alive
between conversation turns, so every stage must finish inside one blocking
call and persist its result to disk immediately)."""

from __future__ import annotations

import json
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "duct_opt" / "checkpoint.json"


def load() -> dict:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open() as fh:
            return json.load(fh)
    return {}


def save(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    tmp.replace(CHECKPOINT_PATH)


def set_key(path: tuple, value) -> None:
    """Merge ``value`` into the checkpoint at nested ``path`` (a tuple of keys), preserving everything else."""
    data = load()
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    save(data)


def has_key(path: tuple) -> bool:
    node = load()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True
