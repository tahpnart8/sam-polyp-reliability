"""Spearman rho of the A3 pairwise-Dice score against its IoU-head score, with a paired bootstrap interval.

Exploratory: added after the results were seen. Uses the per-image tables of the A3 runs and the same bootstrap
index matrix as every other comparison (§7.4). Oracle-box values must equal summary.csv; the script stops if not.

    python scripts/checks/a3_score_comparison.py --metrics-dir results/metrics \
        --out-csv results/metrics/a3_pairwise_vs_iou_head.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from samrel.bootstrap import ci_percentile, indices_for_dataset
from samrel.config import load_config
from samrel.perturb import sign_of


def spearman(conf: np.ndarray, dice: np.ndarray) -> np.ndarray:
    a, b = rankdata(conf, axis=1), rankdata(dice, axis=1)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args(argv)
    m, p = args.metrics_dir, args.metrics_dir / "perturb"
    cfg = load_config("base")
    B, level, root = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]
    runs = pd.read_csv(m / "runs.csv", dtype=str)
    a3 = runs[runs.config == "a3"].run_id.item()
    summary = pd.read_csv(m / "summary.csv", dtype={"seed": str})
    summary = summary[summary.config == "a3"].set_index(["dataset", "metric"])
    pruns = pd.concat([pd.read_csv(f, dtype={"seed": str}) for f in sorted(p.glob("runs_*.csv"))], ignore_index=True)
    sources = [("P0", m / "per_image" / f"{a3}.csv")]
    for r in pruns[(pruns.config == "a3") & (pruns.level != "pp00")].sort_values("level").itertuples():
        sources.append((f"P{int(r.level[2:]) // 10}", p / "per_image" / f"{r.run_id}.csv"))
    rows = []
    for name, path in sources:
        df = pd.read_csv(path).sort_values("image_id")
        for ds in ("colondb", "etis"):
            g = df[(df.dataset == ds) & ~df.excluded.astype(bool)]
            dice, iou, pair = g.dice.to_numpy(), g.conf_iou_pred.to_numpy(), g.conf_pairwise_dice.to_numpy()
            full = np.arange(len(g))[None, :]
            rho_pair, rho_iou = spearman(pair[full], dice[full])[0], spearman(iou[full], dice[full])[0]
            if name == "P0":
                for key, value in (("spearman_pairwise_dice", rho_pair), ("spearman_iou_pred", rho_iou)):
                    if abs(summary.loc[(ds, key), "value"] - value) > 1e-9:
                        raise SystemExit(f"STOP: {ds} {key} differs from summary.csv")
            idx = indices_for_dataset(ds, len(g), B, root)
            lo, hi = ci_percentile(spearman(pair[idx], dice[idx]) - spearman(iou[idx], dice[idx]), level)
            reading = {1: "pairwise_dice_higher", -1: "pairwise_dice_lower", 0: "undetermined"}[sign_of(lo, hi)]
            rows.append({"level": name, "dataset": ds, "rho_pairwise_dice": rho_pair, "rho_iou_head": rho_iou, "diff": rho_pair - rho_iou,
                         "ci_low": lo, "ci_high": hi, "reading": reading, "B": B, "note": "exploratory: added after results were seen"})
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.drop(columns=["B", "note"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
