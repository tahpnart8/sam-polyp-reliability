"""Recompute Dice, IoU and the predicted IoU of selected images from the raw predictions and the masks.

The per-image tables are the raw outputs of each run; this ties them back to the saved probability maps. It needs
the `.npz` prediction files of those images (`<pred-root>/<run_id>/<dataset>/<stem>.npz`) and the dataset masks.
A3 is rebuilt here as the pixel-wise mean of its three members. Tolerance: 1e-6 on stored float16 maps.

    python scripts/checks/raw_prediction_check.py --runs results/metrics/runs.csv --metrics-dir results/metrics \
        --selected results/figures/examples_selected.csv --pred-root <dir> --data-root data/raw \
        --out-csv results/data_checks/raw_prediction_check.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.config import load_config
from samrel.data import discover, read_mask
from samrel.infer import load_prediction


def overlap(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return (2 * inter / total if total else 1.0), (inter / union if union else 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--pred-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = load_config("base")["data"]
    records = {r.image_id: r for r in discover(args.data_root, cfg)}
    runs = pd.read_csv(args.runs, dtype=str)
    run = {(r.config, r.seed): r.run_id for r in runs.itertuples()}
    members = [run[("a2", s)] for s in ("0", "1", "2")]
    shown = {"a0": run[("a0", "0")], "a1": run[("a1", "0")], "a2_s0": run[("a2", "0")], "a2_s1": run[("a2", "1")], "a2_s2": run[("a2", "2")]}
    a3 = run[("a3", "012")]
    tables = {i: pd.read_csv(args.metrics_dir / "per_image" / f"{i}.csv").set_index("image_id") for i in [*shown.values(), a3]}
    rows = []
    for image_id in pd.read_csv(args.selected)["image_id"]:
        ds, stem = image_id.split("/", 1)
        gt = read_mask(records[image_id].mask_path, cfg["mask_threshold"])

        def load(rid):
            return load_prediction(args.pred_root / rid / ds / f"{stem}.npz")

        def record(name, rid, prob, iou_pred):
            dice, iou = overlap(prob >= 0.5, gt)
            t = tables[rid].loc[image_id]
            rows.append({"image_id": image_id, "run": name, "run_id": rid, "dice_npz": dice, "dice_table": t["dice"], "iou_npz": iou, "iou_table": t["iou"],
                         "iou_pred_npz": iou_pred, "iou_pred_table": t["conf_iou_pred"]})

        for name, rid in shown.items():
            p = load(rid)
            record(name, rid, p["prob"].astype(np.float32), float(np.mean(p["iou_pred"])))
        loaded = [load(r) for r in members]
        record("a3", a3, np.mean([x["prob"].astype(np.float32) for x in loaded], axis=0), float(np.mean([np.mean(x["iou_pred"]) for x in loaded])))
    res = pd.DataFrame(rows)
    res["max_gap"] = np.maximum.reduce([(res.dice_npz - res.dice_table).abs(), (res.iou_npz - res.iou_table).abs(), (res.iou_pred_npz - res.iou_pred_table).abs()])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    print(f"rows {len(res)}, images {res.image_id.nunique()}, largest gap {res.max_gap.max():.2e}, rows above 1e-6: {int((res.max_gap > 1e-6).sum())}")
    print(res[res.image_id == "colondb/204"][["run", "dice_npz", "iou_npz", "iou_pred_npz", "max_gap"]].round(4).to_string(index=False))
    return int((res.max_gap > 1e-6).any())


if __name__ == "__main__":
    raise SystemExit(main())
