"""Prompt-perturbation analysis (PROTOCOL §14.5, §14.8): per-run values, Δ to P0, DiD to A0, and
the RQ2 comparison families at every level.

Every run of one dataset is resampled with the same bootstrap index matrix (§7.4), so a paired
difference is the difference of the cached per-resample values; nothing is recomputed per pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregate import SECONDARY_FAILURE_DICE, metric_functions, prepare
from .bootstrap import ci_percentile, indices_for_dataset

AUROC_MIN_CASES = 20  # PROTOCOL §14.8: both failure and success groups need at least this many images
LEVELS = ("pp00", "pp10", "pp20", "pp30")
# metric -> +1 if larger is better (used to read DiD as more / less sensitive), -1 if smaller is better
DIRECTION = {"dice": 1, "iou": 1, "spearman": 1, "auroc": 1, "aurc": -1}
FAMILIES = ("a1_vs_a0", "a2_vs_a0", "a2_vs_a1", "a3_vs_a2")
RQ2_METRICS = ("aurc_iou_pred", "spearman_iou_pred", "auroc_iou_pred", "auroc07_iou_pred")


def direction(metric: str) -> int | None:
    for prefix, d in DIRECTION.items():
        if metric == prefix or metric.startswith(prefix + "_") or (prefix == "auroc" and metric.startswith("auroc07_")):
            return d
    return None


def threshold_of(metric: str) -> str | None:
    if metric.startswith("auroc07_"):
        return "07"
    if metric.startswith("auroc_"):
        return "05"
    return None


@dataclass
class RunValues:
    """Full-sample values and bootstrap samples of one run on one dataset."""
    value: dict[str, float]
    samples: dict[str, np.ndarray]
    counts: dict[str, int]  # n_fail_05, n_success_05, n_fail_07, n_success_07, n_images

    def eligible(self, metric: str) -> bool:
        t = threshold_of(metric)
        if t is None:
            return True
        return min(self.counts[f"n_fail_{t}"], self.counts[f"n_success_{t}"]) >= AUROC_MIN_CASES


def run_values(df: pd.DataFrame, bins: dict, n_resamples: int, root_seed: int) -> dict[str, RunValues]:
    out = {}
    df, stats = prepare(df, bins)
    for dataset, g in df.groupby("dataset", sort=True):
        rows = g.index.to_numpy()
        sub = g.reset_index(drop=True)
        fns = metric_functions(sub, stats[rows])
        boot = indices_for_dataset(dataset, len(sub), n_resamples, root_seed)
        full = np.arange(len(sub))
        dice = sub["dice"].to_numpy(float)
        counts = {"n_images": len(sub), "n_fail_05": int(sub["failed"].sum()),
                  "n_success_05": int((~sub["failed"].astype(bool)).sum()),
                  "n_fail_07": int((dice < SECONDARY_FAILURE_DICE).sum()),
                  "n_success_07": int((dice >= SECONDARY_FAILURE_DICE).sum())}
        out[dataset] = RunValues({m: f(full) for m, f in fns.items()},
                                 {m: np.array([f(r) for r in boot], dtype=float) for m, f in fns.items()}, counts)
    return out


def sign_of(lo: float, hi: float) -> int:
    if np.isnan(lo) or np.isnan(hi):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def diff_row(a: np.ndarray, b: np.ndarray, va: float, vb: float, level: float) -> dict:
    d = a - b
    lo, hi = ci_percentile(d, level)
    return {"diff": va - vb, "ci_low": lo, "ci_high": hi, "sign": sign_of(lo, hi),
            "n_boot_undefined": int(np.isnan(d).sum())}


def consistency(signs: list[int], eligible: list[bool], n_required: int) -> str:
    """PROTOCOL §7.8 over seeds: every seed eligible and its CI excludes 0 with the same sign."""
    if not all(eligible) or len(signs) < n_required:
        return "insufficient_cases"
    if all(s == 1 for s in signs):
        return "consistent_+"
    if all(s == -1 for s in signs):
        return "consistent_-"
    return "not_consistent"


def reading(cls: str, metric: str) -> str:
    """DiD class -> less / more sensitive than A0 (§14.5 rule 2)."""
    if cls in ("insufficient_cases", "not_consistent"):
        return {"insufficient_cases": "insufficient_cases", "not_consistent": "not_distinguishable"}[cls]
    up = (cls == "consistent_+") == (direction(metric) == 1)
    return "less_sensitive" if up else "more_sensitive"
