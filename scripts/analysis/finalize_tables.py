"""Derived tables for the paper, read from aggregate.py outputs only.

- paired_consistency.csv: per comparison family x dataset x metric, how many seeds have a paired
  95% CI excluding 0, their signs and the diff range. ``consistent`` follows PROTOCOL §7.8: all three seeds
  exclude 0 with the same sign. paired.csv itself is not changed.
- cost.csv: every per-run cost file stacked into one table.

Usage:
    python scripts/analysis/finalize_tables.py --metrics-dir results/metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def family(row) -> str:
    return f"{row.config_a.split('_')[0]}_vs_{row.config_b.split('_')[0]}"


def consistency(paired: pd.DataFrame) -> pd.DataFrame:
    p = paired.dropna(subset=["diff"]).copy()
    p["family"] = p.apply(family, axis=1)
    p["sign"] = np.sign(p["diff"]).astype(int)
    rows = []
    for (fam, ds, metric), g in p.groupby(["family", "dataset", "metric"], sort=True):
        ex = g[g["excludes_zero"].astype(bool)]
        signs = sorted(set(ex["sign"]))
        rows.append({"family": fam, "dataset": ds, "metric": metric, "n_pairs": len(g),
                     "n_excluding_zero": len(ex), "signs_excluding_zero": ";".join(map(str, signs)),
                     "diff_min": g["diff"].min(), "diff_max": g["diff"].max(),
                     "consistent": len(g) == 3 and len(ex) == 3 and len(signs) == 1,
                     "pairs": ";".join(g["config_a"] + "-" + g["config_b"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    for name in ("paired", "paired_secondary"):
        src = args.metrics_dir / f"{name}.csv"
        out = args.metrics_dir / f"{name.replace('paired', 'paired_consistency')}.csv"
        consistency(pd.read_csv(src)).to_csv(out, index=False)
        print(out)
    files = sorted((args.metrics_dir / "cost").glob("*.csv"))
    pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(args.metrics_dir / "cost.csv", index=False)
    print(args.metrics_dir / "cost.csv", len(files), "files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
