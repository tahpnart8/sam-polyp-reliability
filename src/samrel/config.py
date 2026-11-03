"""YAML configs with single-level ``inherit`` (configs/*.yaml)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load ``a0``/``a1``/``a2`` (or a path) and resolve ``inherit``."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def config_sha256(cfg: dict[str, Any]) -> str:
    """Hash of the resolved config, key order independent (manifest, PROTOCOL §9.1)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
