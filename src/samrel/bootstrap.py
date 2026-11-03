"""Image-level bootstrap, single and paired (PROTOCOL §7.4)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .seeding import derive_seed


def bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """Index matrix ``[B, n]``. Reusing it across configurations makes comparisons paired."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def indices_for_dataset(dataset: str, n: int, n_resamples: int, root_seed: int) -> np.ndarray:
    return bootstrap_indices(n, n_resamples, derive_seed(root_seed, "bootstrap", dataset))


def ci_percentile(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, NaN resamples ignored (e.g. AUROC without failures in a resample)."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [a, 1.0 - a])
    return float(lo), float(hi)


def resample_statistic(stat: Callable[[np.ndarray], float], idx: np.ndarray) -> np.ndarray:
    """Evaluate ``stat(rows)`` on every resample; ``stat`` receives the resampled row indices."""
    return np.array([stat(row) for row in idx], dtype=np.float64)


def paired_difference(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                      idx: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Point difference on the full sample and its paired bootstrap interval."""
    full = np.arange(idx.shape[1])
    diff = stat_a(full) - stat_b(full)
    samples = resample_statistic(stat_a, idx) - resample_statistic(stat_b, idx)
    lo, hi = ci_percentile(samples, level)
    return {"diff": float(diff), "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "B": int(idx.shape[0])}
