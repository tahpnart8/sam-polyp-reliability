"""Overconfidence of A2 seed 0 as predicted IoU minus actual IoU, for every external-test image.

Writes the full ranked list (one row per image, largest overconfidence first per dataset, ties to the
smaller image_id) and a short file with the five largest gaps and the example images, next to the old
Dice-based gap. Prints the top of each list and checks it against results/figures/examples_selected.csv;
exits with code 1 if the example image is not rank 1. Reads per-image tables only.

Usage:
    python scripts/checks/overconfidence_check.py --metrics-dir results/metrics \
        --selected results/figures/examples_selected.csv \
        --ranked-csv results/data_checks/overconfidence_a2_s0_ranked.csv \
        --out-csv results/data_checks/overconfidence_a2_s0.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--ranked-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = pd.read_csv(args.metrics_dir / "runs.csv", dtype=str)
    run = runs[(runs.config == "a2") & (runs.seed == "0")].run_id.item()
    df = pd.read_csv(args.metrics_dir / "per_image" / f"{run}.csv")
    selected = pd.read_csv(args.selected)
    marks = {"colondb/235": "typical", "etis/100": "typical", "colondb/204": "old_rule", "etis/192": "old_rule"}
    ranked, short, ok = [], [], True
    for ds in ("colondb", "etis"):
        g = df[(df.dataset == ds) & ~df.excluded.astype(bool)].copy()
        g["overconfidence"] = g.conf_iou_pred - g.iou
        g = g.sort_values(["overconfidence", "image_id"], ascending=[False, True]).reset_index(drop=True)
        g["rank"] = g.index + 1
        g["gap_iou"], g["gap_dice"] = g.overconfidence, g.conf_iou_pred - g.dice
        g["rank_iou"] = g["rank"]
        g["rank_dice"] = g.gap_dice.rank(ascending=False, method="first").astype(int)
        ranked.append(g[["dataset", "rank", "image_id", "conf_iou_pred", "iou", "overconfidence", "dice", "n_boxes", "run_id"]])
        keep = g[(g.rank_iou <= 5) | g.image_id.isin(marks)]
        for r in keep.itertuples():
            short.append({"dataset": ds, "image_id": r.image_id, "conf_iou_pred": r.conf_iou_pred, "iou": r.iou, "dice": r.dice,
                          "gap_iou": r.gap_iou, "gap_dice": r.gap_dice, "rank_iou": r.rank_iou, "rank_dice": r.rank_dice,
                          "note": marks.get(r.image_id, "")})
        pick = selected[(selected.dataset == ds) & (selected.rule == "overconfident")].image_id.item()
        top, second = g.iloc[0], g.iloc[1]
        match = pick == top.image_id
        ok &= match
        print(f"{ds}: n={len(g)}, rank 1 {top.image_id} predicted {top.conf_iou_pred:.3f} actual {top.iou:.3f} "
              f"overconfidence {top.overconfidence:.3f}; rank 2 {second.image_id} {second.overconfidence:.3f}; "
              f"selected {pick}: {'matches rank 1' if match else 'DOES NOT MATCH'}; "
              f"share > 0 {(g.overconfidence > 0).mean():.3f}, median {g.overconfidence.median():.4f}, mean {g.overconfidence.mean():.4f}")
    args.ranked_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(ranked, ignore_index=True).to_csv(args.ranked_csv, index=False)
    pd.DataFrame(short).to_csv(args.out_csv, index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
