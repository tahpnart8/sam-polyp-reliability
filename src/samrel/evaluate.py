"""Per-image metrics and ECE bin statistics from saved predictions (PROTOCOL §7, §8).

Everything here reads ``results/predictions/<run_id>/`` (or the Kaggle output copy); nothing is
retrained. A3 is built here from the saved A2 members (PROTOCOL §5.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import PER_IMAGE_COLUMNS, REGIONS
from .data import Record, read_mask
from .ensemble import combine_probs, pairwise_dice
from .infer import load_prediction
from .metrics import dice_iou, image_bin_stats, mean_inside_confidence


def _pred_path(pred_dir: Path, r: Record) -> Path:
    return pred_dir / r.dataset / f"{r.image_id.split('/', 1)[1]}.npz"


def evaluate_image(prob: np.ndarray, gt: np.ndarray, tight_boxes, iou_pred: float, cfg: dict,
                   pairwise: float = float("nan")) -> tuple[dict[str, Any], np.ndarray]:
    """Metric row (without run identifiers) and ``stats[3, n_bins, 3]`` for one image."""
    m = cfg["metrics"]
    thr = cfg["model"]["threshold"]
    if prob.shape != gt.shape:
        raise ValueError(f"prediction {prob.shape} and mask {gt.shape} differ")
    pred = prob >= thr
    dice, iou = dice_iou(pred, gt)
    row = {
        "img_h": gt.shape[0], "img_w": gt.shape[1], "n_boxes": len(tight_boxes),
        "gt_area": int(gt.sum()), "pred_area": int(pred.sum()),
        "dice": dice, "iou": iou, "failed": bool(dice < m["failure_dice"]),
        "conf_iou_pred": float(iou_pred),
        "conf_mean_inside": mean_inside_confidence(prob, thr, m["empty_mask_conf"]),
        "conf_pairwise_dice": pairwise,
    }
    return row, image_bin_stats(prob, gt, tight_boxes, m)


def evaluate_run(run_id: str, config: str, seed: str | int, member_dirs: Sequence[Path],
                 records: Iterable[Record], prompts: dict, cfg: dict,
                 excluded: Iterable[str] = ()) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate one run. One member dir = A0/A1/A2; several = A3 ensemble of those members.

    Returns the per-image table (PROTOCOL §8 columns) and the ``ece_bins`` arrays. Excluded images
    are still evaluated and flagged, so the main analysis can drop them and a sensitivity check
    can keep them.
    """
    excluded = set(excluded)
    rows, stats, ids = [], [], []
    thr = cfg["model"]["threshold"]
    for r in sorted(records, key=lambda r: r.image_id):
        gt = read_mask(r.mask_path, cfg["data"]["mask_threshold"])
        tight = prompts["images"][r.image_id]["tight_boxes"]
        preds = [load_prediction(_pred_path(d, r)) for d in member_dirs]
        probs = [p["prob"].astype(np.float32) for p in preds]
        iou_each = [float(np.mean(p["iou_pred"])) for p in preds]  # mean over boxes (§5.1)
        if len(probs) == 1:
            prob, pairwise = probs[0], float("nan")
        else:
            prob = combine_probs(probs)
            pairwise = pairwise_dice([p >= thr for p in probs], cfg["metrics"]["empty_pair_dice"])
        row, s = evaluate_image(prob, gt, tight, float(np.mean(iou_each)), cfg, pairwise)
        row.update(run_id=run_id, config=config, seed=str(seed), dataset=r.dataset,
                   image_id=r.image_id, excluded=r.image_id in excluded)
        rows.append(row)
        stats.append(s)
        ids.append(r.image_id)
    df = pd.DataFrame(rows, columns=list(PER_IMAGE_COLUMNS))
    bins = {"stats": np.stack(stats) if stats else np.zeros((0, len(REGIONS), cfg["metrics"]["ece_bins"], 3)),
            "image_ids": np.asarray(ids), "regions": np.asarray(REGIONS),
            "columns": np.asarray(["count", "sum_conf", "sum_correct"])}
    return df, bins


def write_run_metrics(df: pd.DataFrame, bins: dict[str, np.ndarray], run_id: str, out_dir: Path) -> None:
    """Write ``per_image/<run_id>.csv`` and ``ece_bins/<run_id>.npz`` under ``out_dir``."""
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)
    (out_dir / "ece_bins").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_image" / f"{run_id}.csv", index=False)
    np.savez_compressed(out_dir / "ece_bins" / f"{run_id}.npz", **bins)
