"""Accuracy, region ECE and failure-detection metrics (PROTOCOL §7)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .constants import REGIONS


# ---------------------------------------------------------------- §7.1 accuracy
def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Dice and IoU of two binary masks; both empty gives 1, one empty gives 0."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    s = pred.sum(dtype=np.int64) + gt.sum(dtype=np.int64)
    union = s - inter
    if s == 0:
        return 1.0, 1.0
    return float(2 * inter / s), float(inter / union)


def confidence(prob: np.ndarray) -> np.ndarray:
    """Pixel confidence ``c = max(p, 1 - p)``."""
    prob = prob.astype(np.float64)
    return np.maximum(prob, 1.0 - prob)


def mean_inside_confidence(prob: np.ndarray, threshold: float = 0.5, empty_value: float = 0.0) -> float:
    """Mean ``c`` over predicted-foreground pixels; ``empty_value`` if none (Q-B6)."""
    inside = prob >= threshold
    if not inside.any():
        return float(empty_value)
    return float(confidence(prob[inside]).mean())


# ---------------------------------------------------------------- §7.2 regions and ECE
def roi_mask(shape: tuple[int, int], tight_boxes: Sequence[Sequence[int]], expand_per_side: float) -> np.ndarray:
    """Union of ground-truth boxes, each widened by ``expand_per_side`` of its size on every side."""
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tight_boxes:
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        ax0 = int(max(0, np.floor(x0 - expand_per_side * bw)))
        ay0 = int(max(0, np.floor(y0 - expand_per_side * bh)))
        ax1 = int(min(w - 1, np.ceil(x1 + expand_per_side * bw)))
        ay1 = int(min(h - 1, np.ceil(y1 + expand_per_side * bh)))
        out[ay0:ay1 + 1, ax0:ax1 + 1] = True
    return out


def boundary_mask(gt: np.ndarray, frac_diag: float) -> np.ndarray:
    """Pixels within ``frac_diag`` of the image diagonal from the ground-truth boundary.

    Foreground pixels use their distance to the nearest background pixel and vice versa, so the
    band is symmetric across the contour.
    """
    gt = gt.astype(bool)
    h, w = gt.shape
    r = frac_diag * float(np.hypot(h, w))
    if gt.all() or not gt.any():
        return np.zeros_like(gt)
    d_in = ndimage.distance_transform_edt(gt)
    d_out = ndimage.distance_transform_edt(~gt)
    return (gt & (d_in <= r)) | (~gt & (d_out <= r))


def region_masks(gt: np.ndarray, tight_boxes, expand_per_side: float, frac_diag: float) -> dict[str, np.ndarray]:
    return {
        "img": np.ones(gt.shape, dtype=bool),
        "roi": roi_mask(gt.shape, tight_boxes, expand_per_side),
        "bnd": boundary_mask(gt, frac_diag),
    }


def bin_edges(n_bins: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.linspace(lo, hi, n_bins + 1)


def _bin_index(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Bins are (lo, hi]; the lowest edge is folded into the first bin.
    idx = np.searchsorted(edges, conf, side="left") - 1
    return np.clip(idx, 0, len(edges) - 2)


def ece_bin_stats(prob: np.ndarray, gt: np.ndarray, region: np.ndarray, n_bins: int = 15,
                  lo: float = 0.0, hi: float = 1.0, threshold: float = 0.5) -> np.ndarray:
    """Per-bin ``[count, sum_conf, sum_correct]`` of one image restricted to ``region``."""
    sel = region.astype(bool)
    p = prob[sel].astype(np.float64)
    y = gt[sel].astype(bool)
    c = np.maximum(p, 1.0 - p)
    correct = ((p >= threshold) == y).astype(np.float64)
    idx = _bin_index(c, bin_edges(n_bins, lo, hi))
    out = np.zeros((n_bins, 3), dtype=np.float64)
    out[:, 0] = np.bincount(idx, minlength=n_bins)
    out[:, 1] = np.bincount(idx, weights=c, minlength=n_bins)
    out[:, 2] = np.bincount(idx, weights=correct, minlength=n_bins)
    return out


def image_bin_stats(prob, gt, tight_boxes, cfg_metrics: dict) -> np.ndarray:
    """``stats[3 regions, n_bins, 3]`` for one image, regions ordered as ``REGIONS``."""
    lo, hi = cfg_metrics["ece_range"]
    regions = region_masks(gt, tight_boxes, cfg_metrics["roi_expand_per_side"], cfg_metrics["boundary_frac_diag"])
    return np.stack([ece_bin_stats(prob, gt, regions[r], cfg_metrics["ece_bins"], lo, hi) for r in REGIONS])


def ece_from_bins(stats: np.ndarray) -> float:
    """ECE from pooled bin stats ``[..., n_bins, 3]`` (leading axes are summed first)."""
    s = stats.reshape(-1, stats.shape[-2], 3).sum(axis=0)
    total = s[:, 0].sum()
    if total == 0:
        return float("nan")
    nz = s[:, 0] > 0
    gap = np.abs(s[nz, 2] / s[nz, 0] - s[nz, 1] / s[nz, 0])
    return float((s[nz, 0] / total * gap).sum())


def ece_direct(prob: np.ndarray, gt: np.ndarray, n_bins: int = 15, lo: float = 0.0, hi: float = 1.0) -> float:
    """Independent loop implementation used only to check :func:`ece_from_bins`."""
    p = prob.ravel().astype(np.float64)
    y = gt.ravel().astype(bool)
    c = np.maximum(p, 1 - p)
    yhat = p >= 0.5
    edges = np.linspace(lo, hi, n_bins + 1)
    total, ece = len(p), 0.0
    for m in range(n_bins):
        in_bin = (c > edges[m]) & (c <= edges[m + 1])
        if m == 0:
            in_bin |= c <= edges[0]
        if in_bin.any():
            ece += in_bin.sum() / total * abs((yhat[in_bin] == y[in_bin]).mean() - c[in_bin].mean())
    return float(ece)


# ---------------------------------------------------------------- §7.3 failure detection
def auroc_failure(conf: np.ndarray, failed: np.ndarray) -> float:
    """AUROC with failures as positives and ``-conf`` as score; NaN if one class is absent."""
    failed = np.asarray(failed, dtype=bool)
    conf = np.asarray(conf, dtype=np.float64)
    if failed.all() or not failed.any():
        return float("nan")
    return float(roc_auc_score(failed, -conf))


def _order(conf, ids) -> np.ndarray:
    # Descending confidence, ties broken by ascending image_id so AURC is deterministic.
    ids = np.asarray(ids, dtype=object)
    return np.array(sorted(range(len(conf)), key=lambda i: (-float(conf[i]), str(ids[i]))), dtype=np.int64)


def risk_coverage_curve(conf, dice, ids) -> tuple[np.ndarray, np.ndarray]:
    """Coverage ``k/N`` and risk ``mean(1 - dice)`` of the ``k`` most confident images."""
    dice = np.asarray(dice, dtype=np.float64)
    order = _order(np.asarray(conf, dtype=np.float64), ids)
    loss = 1.0 - dice[order]
    k = np.arange(1, len(loss) + 1)
    return k / len(loss), np.cumsum(loss) / k


def aurc(conf, dice, ids) -> float:
    if len(dice) == 0:
        return float("nan")
    return float(risk_coverage_curve(conf, dice, ids)[1].mean())
