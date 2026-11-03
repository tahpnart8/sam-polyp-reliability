"""Repository paths and the results-directory guard (PROTOCOL §11)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results"


def results_dir() -> Path:
    """Return the results root, honouring ``SAMREL_RESULTS_DIR`` (tests redirect it)."""
    return Path(os.environ.get("SAMREL_RESULTS_DIR", str(DEFAULT_RESULTS))).resolve()


def data_root() -> Path | None:
    """Return ``SAMREL_DATA_ROOT`` if set and existing, else None (real-data tests skip)."""
    value = os.environ.get("SAMREL_DATA_ROOT")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None
