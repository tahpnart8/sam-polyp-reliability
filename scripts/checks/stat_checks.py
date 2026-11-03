"""Two descriptive checks (pooled-ECE pixel weights; seed vs bootstrap variation) from files in results/.

ece_pixel_weights.csv: how unequal the per-image pixel weights of pooled ECE are, per region and
  dataset. Region sizes depend only on the image and the ground-truth mask (identical in every run),
  so this uses no model output. Columns: share of the region's pixels held by the 10% largest
  images, effective number of images 1/sum(w^2), largest single-image share, largest/smallest.
variance_sources.csv: for A1 and A2, the seed SD of each metric (seed_stability.csv) next to the
  mean half-width of the per-seed bootstrap 95% CI (summary.csv), i.e. training-seed variation vs
  image-sampling uncertainty. Numbers already reported; only put side by side.

    python scripts/checks/stat_checks.py --metrics-dir results/metrics --out-dir results/data_checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.constants import PRIMARY_DATASETS, REGIONS

METRICS = ("dice", "iou", "ece_img", "ece_roi", "ece_bnd", "aurc_iou_pred", "spearman_iou_pred",
           "aurc_mean_inside", "spearman_mean_inside")


def pixel_weights(bins_file: Path) -> pd.DataFrame:
    with np.load(bins_file) as z:
        counts = z["stats"][..., 0].sum(-1)  # n_images x regions
        ds = np.array([i.split("/", 1)[0] for i in z["image_ids"]])
    rows = []
    for d in PRIMARY_DATASETS:
        for k, reg in enumerate(REGIONS):
            n = np.sort(counts[ds == d, k])[::-1]
            w = n / n.sum()
            top = int(np.ceil(0.1 * len(n)))
            rows.append({"dataset": d, "region": reg, "n_images": len(n), "top10pct_share": round(w[:top].sum(), 4),
                         "effective_n_images": round(1 / (w ** 2).sum(), 1), "max_share": round(w[0], 4),
                         "max_over_min": round(n[0] / n[-1], 1)})
    return pd.DataFrame(rows)


def variance_sources(metrics_dir: Path) -> pd.DataFrame:
    s = pd.read_csv(metrics_dir / "summary.csv", dtype={"seed": str})
    ss = pd.read_csv(metrics_dir / "seed_stability.csv")
    rows = []
    for c in ("a1", "a2"):
        for d in PRIMARY_DATASETS:
            for m in METRICS:
                x = s[(s.config == c) & (s.dataset == d) & (s.metric == m) & s.seed.isin(["0", "1", "2"])]
                y = ss[(ss.config == c) & (ss.dataset == d) & (ss.metric == m)]
                if len(x) != 3 or len(y) != 1:
                    continue
                half = float(((x.ci_high - x.ci_low) / 2).mean())
                sd = float(y["sd"].iloc[0])
                rows.append({"config": c, "dataset": d, "metric": m, "seed_mean": round(float(y["mean"].iloc[0]), 4),
                             "seed_sd": round(sd, 4), "bootstrap_ci_halfwidth_mean": round(half, 4),
                             "sd_over_halfwidth": round(sd / half, 2) if half else np.nan})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = pd.read_csv(args.metrics_dir / "runs.csv")
    first = runs.run_id.iloc[0]
    pw = pixel_weights(args.metrics_dir / "ece_bins" / f"{first}.npz")
    # region sizes come from the ground truth only, so every run must give the same counts
    ref = np.load(args.metrics_dir / "ece_bins" / f"{first}.npz")["stats"][..., 0].sum(-1)
    for r in runs.run_id[1:]:
        if not np.array_equal(ref, np.load(args.metrics_dir / "ece_bins" / f"{r}.npz")["stats"][..., 0].sum(-1)):
            raise SystemExit(f"STOP: region pixel counts of {r} differ from {first}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pw.to_csv(args.out_dir / "ece_pixel_weights.csv", index=False)
    vs = variance_sources(args.metrics_dir)
    vs.to_csv(args.out_dir / "variance_sources.csv", index=False)
    print(pw.to_string(index=False))
    print(vs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
