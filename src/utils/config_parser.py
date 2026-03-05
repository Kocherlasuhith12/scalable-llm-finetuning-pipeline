"""YAML config loading and merging."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config; resolve _base if present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base", None)
    if base:
        base_path = path.parent / base
        base_config = load_config(base_path)
        data = merge_configs(base_config, data)
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out
