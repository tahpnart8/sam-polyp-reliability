"""Dataset-level metrics, bootstrap CIs, paired comparisons and seed stability (PROTOCOL §7.4, §8).

Inputs are only the per-image CSV and ECE-bin files of each run; nothing is recomputed from
predictions here. Excluded images (PROTOCOL §1.5) are dropped before any statistic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from .bootstrap import ci_percentile, indices_for_dataset
from .constants import REGIONS
from .metrics import aurc, auroc_failure, ece_from_bins

SCORES = {"iou_pred": "conf_iou_pred", "mean_inside": "conf_mean_inside", "pairwise_dice": "conf_pairwise_dice"}
# Failure threshold 0.7 is a sensitivity analysis (§7.6). Spearman rho was moved to the main
# table, so only the threshold-0.7 metrics are secondary.
SECONDARY_FAILURE_DICE = 0.7
SECONDARY_PREFIXES = ("auroc07_", "n_failures_07")


def is_secondary(metric: str) -> bool:
    """True for secondary metrics (PROTOCOL §7.6-§7.7); the main table must not mix them in."""
    return metric.startswith(SECONDARY_PREFIXES)


def _spearman(conf: np.ndarray, dice: np.ndarray) -> float:
    if len(conf) < 3 or np.all(conf == conf[0]) or np.all(dice == dice[0]):
        return float("nan")
    return float(spearmanr(conf, dice).statistic)


def metric_functions(df: pd.DataFrame, stats: np.ndarray) -> dict[str, Callable[[np.ndarray], float]]:
    """Each metric as a function of resampled row indices into ``df`` (rows aligned with ``stats``)."""
    dice = df["dice"].to_numpy(float)
    iou = df["iou"].to_numpy(float)
    failed = df["failed"].to_numpy(bool)
    failed07 = dice < SECONDARY_FAILURE_DICE
    ids = df["image_id"].to_numpy(object)
    fns: dict[str, Callable[[np.ndarray], float]] = {
        "dice": lambda r: float(dice[r].mean()),
        "iou": lambda r: float(iou[r].mean()),
        "n_failures": lambda r: float(failed[r].sum()),
        "n_failures_07": lambda r: float(failed07[r].sum()),
    }
    for k, region in enumerate(REGIONS):
        fns[f"ece_{region}"] = lambda r, k=k: ece_from_bins(stats[r, k])
    for name, col in SCORES.items():
        conf = df[col].to_numpy(float)
        if np.isnan(conf).all():
            continue
        fns[f"auroc_{name}"] = lambda r, c=conf: auroc_failure(c[r], failed[r])
        # Resampled rows repeat images; suffixing the position keeps the tie-break deterministic.
        fns[f"aurc_{name}"] = lambda r, c=conf: aurc(c[r], dice[r], [f"{ids[i]}#{j}" for j, i in enumerate(r)])
        fns[f"auroc07_{name}"] = lambda r, c=conf: auroc_failure(c[r], failed07[r])
        fns[f"spearman_{name}"] = lambda r, c=conf: _spearman(c[r], dice[r])
    return fns


def prepare(df: pd.DataFrame, bins: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop excluded images and align bin stats with rows sorted by image_id."""
    pos = {i: k for k, i in enumerate(bins["image_ids"].tolist())}
    df = df[~df["excluded"].astype(bool)].sort_values("image_id").reset_index(drop=True)
    return df, bins["stats"][[pos[i] for i in df["image_id"]]]


def summarise_run(df: pd.DataFrame, bins: dict, n_resamples: int, level: float, root_seed: int) -> pd.DataFrame:
    """One row per (dataset, metric) with value and percentile CI."""
    rows = []
    df, stats = prepare(df, bins)
    for dataset, g in df.groupby("dataset", sort=True):
        idx_rows = g.index.to_numpy()
        sub, sub_stats = g.reset_index(drop=True), stats[idx_rows]
        fns = metric_functions(sub, sub_stats)
        boot = indices_for_dataset(dataset, len(sub), n_resamples, root_seed)
        full = np.arange(len(sub))
        for metric, f in fns.items():
            samples = np.array([f(r) for r in boot])
            lo, hi = ci_percentile(samples, level)
            rows.append({"dataset": dataset, "metric": metric, "value": f(full), "ci_low": lo, "ci_high": hi,
                         "n_images": len(sub), "n_failures": int(sub["failed"].sum())})
    return pd.DataFrame(rows)


def paired(df_a: pd.DataFrame, bins_a: dict, df_b: pd.DataFrame, bins_b: dict,
           n_resamples: int, level: float, root_seed: int) -> pd.DataFrame:
    """Paired bootstrap of ``A - B`` per dataset: the same resampled image indices for both."""
    a, sa = prepare(df_a, bins_a)
    b, sb = prepare(df_b, bins_b)
    if a["image_id"].tolist() != b["image_id"].tolist():
        raise ValueError("paired comparison needs the same images in both runs")
    rows = []
    for dataset in sorted(a["dataset"].unique()):
        m = (a["dataset"] == dataset).to_numpy()
        ga, gb = a[m].reset_index(drop=True), b[m].reset_index(drop=True)
        fa, fb = metric_functions(ga, sa[m]), metric_functions(gb, sb[m])
        boot = indices_for_dataset(dataset, len(ga), n_resamples, root_seed)
        full = np.arange(len(ga))
        for metric in sorted(set(fa) & set(fb)):
            diffs = np.array([fa[metric](r) - fb[metric](r) for r in boot])
            lo, hi = ci_percentile(diffs, level)
            rows.append({"dataset": dataset, "metric": metric, "diff": fa[metric](full) - fb[metric](full),
                         "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0), "B": n_resamples})
    return pd.DataFrame(rows)


def seed_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD (ddof=1) over seeds of each (config, dataset, metric); A1 and A2 only."""
    s = summary[summary["config"].isin(["a1", "a2"])]
    g = s.groupby(["config", "dataset", "metric"])
    out = g["value"].agg(mean="mean", sd=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                         n_seeds="count").reset_index()
    out["run_ids"] = g["run_ids"].agg(lambda v: ";".join(sorted(v))).to_numpy()
    return out
