"""run_id, manifest and environment records (PROTOCOL §9)."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import MANIFEST_KEYS
from .paths import REPO_ROOT

PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "scikit_learn": "scikit-learn", "pillow": "pillow",
    "opencv": "opencv-python-headless", "segment_anything": "segment-anything", "imagehash": "ImageHash",
    "pandas": "pandas",
}


def make_run_id(config: str, seed: int | str, commit: str, when: datetime | None = None) -> str:
    """``<config>_s<seed>_<commit7>_<yyyymmdd-hhmm>`` in UTC."""
    when = when or datetime.now(timezone.utc)
    return f"{config}_s{seed}_{commit[:7]}_{when.astimezone(timezone.utc):%Y%m%d-%H%M}"


def git_commit(repo: Path = REPO_ROOT, paths: tuple[str, ...] = ()) -> tuple[str, bool]:
    """(commit hash, dirty). ``paths`` limits the dirty check to what is actually shipped, so the
    writing stream's uncommitted paper files do not block a code job. On Kaggle there is no .git;
    the caller reads COMMIT.txt instead."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all",
                                              "--", *paths], cwd=repo, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Software and hardware description (PROTOCOL §9.2). The hostname is hashed, not stored."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
    }
    env.update({k: _version(v) for k, v in PACKAGES.items()})
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env.update(gpu_name=props.name, gpu_count=torch.cuda.device_count(),
                       gpu_mem_total_mb=int(props.total_memory // 2**20))
        else:
            env.update(gpu_name=None, gpu_count=0, gpu_mem_total_mb=None)
    except ImportError:
        env["torch"] = None
    try:
        env["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        env["driver"] = None
    return env


def new_manifest(**fields: Any) -> dict[str, Any]:
    """Manifest with every key of PROTOCOL §9.1; unknown keys are rejected."""
    unknown = set(fields) - set(MANIFEST_KEYS)
    if unknown:
        raise KeyError(f"not manifest keys: {sorted(unknown)}")
    return {k: fields.get(k) for k in MANIFEST_KEYS}


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
