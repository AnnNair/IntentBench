"""Shared helpers: seeding and config loading.

Every script reads its parameters from a versioned YAML config (see
configs/) instead of hard-coding values -- this is part of how the project
stays reproducible without Docker (see README.md, section "Reproducibility
without Docker").
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load and shallow-merge one or more YAML config files, in order."""
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int) -> None:
    """Fix the random seed across every library involved in a run."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
