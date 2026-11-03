"""Build summary, paired and seed_stability tables (main + _secondary) from per-run files (PROTOCOL §7.4-§7.6, §8).

Usage:
    python scripts/analysis/aggregate.py --metrics-dir results/metrics --runs results/metrics/runs.csv

runs.csv lists the runs to include, one row per run: run_id, config, seed, commit. Comparisons are
fixed in advance (COMPARISONS below); A3 is compared with every single A2 seed (RQ3, PROTOCOL §7.4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.aggregate import is_secondary, paired, seed_stability, summarise_run
from samrel.config import load_config
from samrel.constants import PAIRED_COLUMNS, SEED_STABILITY_COLUMNS, SUMMARY_COLUMNS

# (A, B) pairs: difference A - B. Configs with seeds are compared seed by seed.
COMPARISONS = [("a1", "a0"), ("a2", "a0"), ("a2", "a1"), ("a3", "a2")]


def write_split(df: pd.DataFrame, out: Path, stem: str) -> None:
    """Main metrics to ``<stem>.csv``; secondary ones (threshold 0.7, PROTOCOL §7.6) to
    ``<stem>_secondary.csv`` so the main tables never mix them in."""
    sec = df["metric"].map(is_secondary)
    df[~sec].to_csv(out / f"{stem}.csv", index=False)
    df[sec].to_csv(out / f"{stem}_secondary.csv", index=False)


def load_run(metrics_dir: Path, run_id: str):
    df = pd.read_csv(metrics_dir / "per_image" / f"{run_id}.csv")
    with np.load(metrics_dir / "ece_bins" / f"{run_id}.npz", allow_pickle=False) as z:
        bins = {k: z[k] for k in z.files}
    return df, bins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = load_config("base")
    B, level, root = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]
    runs = pd.read_csv(args.runs, dtype=str)
    loaded = {r.run_id: load_run(args.metrics_dir, r.run_id) for r in runs.itertuples()}

    summ = []
    for r in runs.itertuples():
        s = summarise_run(*loaded[r.run_id], B, level, root)
        s.insert(0, "seed", r.seed); s.insert(0, "config", r.config)
        s["run_ids"], s["commit"] = r.run_id, r.commit
        summ.append(s)
    summary = pd.concat(summ, ignore_index=True)[list(SUMMARY_COLUMNS)]
    write_split(summary, args.metrics_dir, "summary")
    write_split(seed_stability(summary)[list(SEED_STABILITY_COLUMNS)], args.metrics_dir, "seed_stability")

    pairs = []
    by_cfg = {c: g for c, g in runs.groupby("config")}
    for a, b in COMPARISONS:
        if a not in by_cfg or b not in by_cfg:
            continue
        for ra in by_cfg[a].itertuples():
            for rb in by_cfg[b].itertuples():
                # Same-seed pairing for A2 vs A1; every seed otherwise.
                if {a, b} == {"a1", "a2"} and ra.seed != rb.seed:
                    continue
                p = paired(*loaded[ra.run_id], *loaded[rb.run_id], B, level, root)
                p.insert(0, "config_b", f"{b}_s{rb.seed}"); p.insert(0, "config_a", f"{a}_s{ra.seed}")
                pairs.append(p)
    if pairs:
        write_split(pd.concat(pairs, ignore_index=True)[list(PAIRED_COLUMNS)], args.metrics_dir, "paired")
    print(f"summary rows {len(summary)}, paired rows {sum(len(p) for p in pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
