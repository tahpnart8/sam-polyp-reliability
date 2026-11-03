"""Per-image ECE sensitivity check (sensitivity analysis, PROTOCOL §7.9): every image weighs the same.

For R_roi and R_bnd on CVC-ColonDB and ETIS, the ECE of each image is computed from its own bin
statistics (results/metrics/ece_bins/<run_id>.npz, the same bins as the pooled ECE). Reported:
  ece_per_image_summary.csv      per run: median, IQR (q1, q3), mean of per-image ECE, pooled ECE
  ece_per_image_paired.csv       paired bootstrap of the MEAN per-image ECE, same families and seed
                                 pairing as paired_consistency.csv (B = 1000, percentile, §7.4 indices)
  ece_per_image_consistency.csv  §7.8 reading over seeds, next to the pooled-ECE reading
The pooled ECE stays the main number; this file is only a sensitivity analysis.

    python scripts/checks/ece_per_image.py --metrics-dir results/metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.bootstrap import ci_percentile, indices_for_dataset
from samrel.config import load_config
from samrel.constants import PRIMARY_DATASETS, REGIONS
from samrel.metrics import ece_from_bins

FAMILIES = {"a1_vs_a0": [("a1", s, "a0", "0") for s in "012"], "a2_vs_a0": [("a2", s, "a0", "0") for s in "012"],
            "a2_vs_a1": [("a2", s, "a1", s) for s in "012"], "a3_vs_a2": [("a3", "012", "a2", s) for s in "012"]}


def per_image_ece(metrics_dir: Path, run_id: str) -> dict[tuple[str, str], np.ndarray]:
    df = pd.read_csv(metrics_dir / "per_image" / f"{run_id}.csv")
    keep = set(df.loc[~df.excluded.astype(bool), "image_id"])
    with np.load(metrics_dir / "ece_bins" / f"{run_id}.npz") as z:
        ids, stats = z["image_ids"].tolist(), z["stats"]
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    out = {}
    for ds in PRIMARY_DATASETS:
        rows = [i for i in order if ids[i].split("/", 1)[0] == ds and ids[i] in keep]
        for region in ("roi", "bnd"):
            k = REGIONS.index(region)
            out[(ds, region)] = (np.array([ece_from_bins(stats[i, k]) for i in rows]), ece_from_bins(stats[rows, k]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = load_config("base")
    B, level, root = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]
    runs = pd.read_csv(args.metrics_dir / "runs.csv", dtype={"seed": str})
    vals = {(r.config, r.seed): per_image_ece(args.metrics_dir, r.run_id) for r in runs.itertuples()}

    summ = []
    for (c, s), v in vals.items():
        for (ds, region), (e, pooled) in v.items():
            q1, med, q3 = np.percentile(e, [25, 50, 75])
            summ.append({"config": c, "seed": s, "dataset": ds, "region": region, "n_images": len(e),
                         "median": med, "q1": q1, "q3": q3, "mean": e.mean(), "pooled": pooled})
    pd.DataFrame(summ).to_csv(args.metrics_dir / "ece_per_image_summary.csv", index=False)

    paired, cons = [], []
    pooled_cons = pd.read_csv(args.metrics_dir / "paired_consistency.csv")
    for fam, pairs in FAMILIES.items():
        for ds in PRIMARY_DATASETS:
            for region in ("roi", "bnd"):
                signs = []
                for ca, sa, cb, sb in pairs:
                    ea, eb = vals[(ca, sa)][(ds, region)][0], vals[(cb, sb)][(ds, region)][0]
                    idx = indices_for_dataset(ds, len(ea), B, root)
                    d = ea[idx].mean(1) - eb[idx].mean(1)
                    lo, hi = ci_percentile(d, level)
                    sign = 1 if lo > 0 else (-1 if hi < 0 else 0)
                    signs.append(sign)
                    paired.append({"family": fam, "pair": f"{ca}_s{sa}-{cb}_s{sb}", "dataset": ds, "region": region,
                                   "diff_mean_per_image": ea.mean() - eb.mean(), "ci_low": lo, "ci_high": hi, "sign": sign})
                cls = ("consistent_+" if all(x == 1 for x in signs) else
                       "consistent_-" if all(x == -1 for x in signs) else "not_consistent")
                p = pooled_cons[(pooled_cons.family == fam) & (pooled_cons.dataset == ds) &
                                (pooled_cons.metric == f"ece_{region}")].iloc[0]
                pooled_cls = ("not_consistent" if not p.consistent else
                              "consistent_+" if int(p.signs_excluding_zero) > 0 else "consistent_-")
                cons.append({"family": fam, "dataset": ds, "region": region, "per_image_class": cls,
                             "n_excluding_zero": sum(x != 0 for x in signs), "pooled_class": pooled_cls,
                             "same_reading": cls == pooled_cls})
    pd.DataFrame(paired).to_csv(args.metrics_dir / "ece_per_image_paired.csv", index=False)
    cons = pd.DataFrame(cons)
    cons.to_csv(args.metrics_dir / "ece_per_image_consistency.csv", index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False))
    print(cons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
