"""Overconfidence of A2 seed 0 as predicted IoU minus actual IoU, against the old Dice-based rule.

Writes the five largest gaps of each external set, the distribution of the gap, and the values of the
typical and overconfident example images under both rules. Reads per-image tables only.

Usage:
    python scripts/checks/overconfidence_check.py --metrics-dir results/metrics --out-csv results/data_checks/overconfidence_a2_s0.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = pd.read_csv(args.metrics_dir / "runs.csv", dtype=str)
    run = runs[(runs.config == "a2") & (runs.seed == "0")].run_id.item()
    df = pd.read_csv(args.metrics_dir / "per_image" / f"{run}.csv")
    rows, marks = [], {"colondb/235": "typical", "etis/100": "typical", "colondb/204": "old_rule", "etis/192": "old_rule"}
    for ds in ("colondb", "etis"):
        g = df[(df.dataset == ds) & ~df.excluded.astype(bool)].copy()
        g["gap_iou"] = g.conf_iou_pred - g.iou
        g["gap_dice"] = g.conf_iou_pred - g.dice
        g["rank_iou"] = g.gap_iou.rank(ascending=False, method="first").astype(int)
        g["rank_dice"] = g.gap_dice.rank(ascending=False, method="first").astype(int)
        keep = g[(g.rank_iou <= 5) | g.image_id.isin(marks)]
        for r in keep.sort_values("rank_iou").itertuples():
            rows.append({"dataset": ds, "image_id": r.image_id, "conf_iou_pred": r.conf_iou_pred, "iou": r.iou, "dice": r.dice,
                         "gap_iou": r.gap_iou, "gap_dice": r.gap_dice, "rank_iou": r.rank_iou, "rank_dice": r.rank_dice,
                         "note": marks.get(r.image_id, "")})
        print(ds, f"n={len(g)} share_gap_iou>0={(g.gap_iou > 0).mean():.3f} mean={g.gap_iou.mean():.4f} "
                  f"median={g.gap_iou.median():.4f} rule_pick={g.sort_values(['gap_iou', 'image_id'], ascending=[False, True]).image_id.iloc[0]}")
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
