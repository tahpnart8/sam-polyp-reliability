"""Spearman rho of A0-A3 per dataset, perturbation level, score and seed, each paired with A0.

A3 versus A0 was absent from paired_consistency.csv and rq2_consistency.csv; adding it here is an
analysis made after the results were seen and is labelled exploratory. Rules are the existing ones:
same bootstrap index matrix per dataset (PROTOCOL §7.4), percentile 95%, A1/A2 consistent only when
all three seeds exclude 0 with one sign (PROTOCOL §7.8), A0/A3 a single run (§14.5).

Writes to --out-dir:
  rq2_spearman_by_seed.csv  one row per dataset x level x score x run: rho, CI, paired diff with A0, CI, class
  rq2_spearman_cells.csv    consistency reading per dataset x level x score x config (seed range, k of 3)
  rq2_spearman_counts.csv   counts of better / worse / undetermined, per config, per score, per dataset
  a3_vs_a0.csv              A3 - A0 for every metric of RQ1 and RQ2 on the external sets, P0-P3
Usage:
    python scripts/checks/rq2_spearman_audit.py --perturb-dir results/metrics/perturb --out-dir results/metrics/perturb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.bootstrap import ci_percentile
from samrel.config import load_config
from samrel.perturb import LEVELS, run_values, sign_of

SCORES = ("iou_pred", "mean_inside")
A3_METRICS = ("dice", "iou", "ece_img", "ece_roi", "ece_bnd", "aurc_iou_pred", "aurc_mean_inside",
              "spearman_iou_pred", "spearman_mean_inside")
LABEL = {1: "higher_than_a0", -1: "lower_than_a0", 0: "undetermined"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-resamples", type=int, help="tests only; default bootstrap.n_resamples (1000)")
    args = ap.parse_args(argv)
    cfg = load_config("base")
    B, level, root = args.n_resamples or cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]
    pdir = args.perturb_dir
    runs = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(pdir.glob("runs_*.csv"))], ignore_index=True)

    vals = {}
    for r in runs.itertuples():
        df = pd.read_csv(pdir / "per_image" / f"{r.run_id}.csv")
        with np.load(pdir / "ece_bins" / f"{r.run_id}.npz") as z:
            bins = {k: z[k] for k in z.files}
        vals[(r.config, r.seed, r.level)] = run_values(df, bins, B, root)
        print(r.run_id, flush=True)

    rows = []
    for (c, s, lv), per_ds in vals.items():
        for ds, v in per_ds.items():
            a0 = vals[("a0", "0", lv)][ds]
            for score in SCORES:
                m = f"spearman_{score}"
                lo, hi = ci_percentile(v.samples[m], level)
                row = {"dataset": ds, "level": lv, "score": score, "config": c, "seed": s,
                       "rho": v.value[m], "ci_low": lo, "ci_high": hi}
                if c != "a0":
                    dlo, dhi = ci_percentile(v.samples[m] - a0.samples[m], level)
                    sign = sign_of(dlo, dhi)
                    row.update({"diff_vs_a0": v.value[m] - a0.value[m], "diff_ci_low": dlo, "diff_ci_high": dhi,
                                "sign": sign, "reading": LABEL[sign]})
                rows.append(row)
    by_seed = pd.DataFrame(rows).sort_values(["dataset", "score", "level", "config", "seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(args.out_dir / "rq2_spearman_by_seed.csv", index=False)

    cells = []
    for (ds, lv, score, c), g in by_seed[by_seed.config != "a0"].groupby(["dataset", "level", "score", "config"]):
        signs = g["sign"].astype(int).tolist()
        n_up, n_down = signs.count(1), signs.count(-1)
        if all(s == 1 for s in signs):
            cls = "consistent_higher"
        elif all(s == -1 for s in signs):
            cls = "consistent_lower"
        else:
            cls = "not_consistent"
        cells.append({"dataset": ds, "level": lv, "score": score, "config": c, "n_seeds": len(signs),
                      "n_higher": n_up, "n_lower": n_down, "n_undetermined": signs.count(0), "class": cls,
                      "diff_min": g["diff_vs_a0"].min(), "diff_max": g["diff_vs_a0"].max(),
                      "n_point_higher": int((g["diff_vs_a0"] > 0).sum())})
    cells = pd.DataFrame(cells)
    cells.to_csv(args.out_dir / "rq2_spearman_cells.csv", index=False)

    counts = []
    for keys, sub in [(("all", "all"), cells), *[((ds, "all"), g) for ds, g in cells.groupby("dataset")],
                      *[(("all", sc), g) for sc, g in cells.groupby("score")],
                      *[(k, g) for k, g in cells.groupby(["dataset", "score"])]]:
        for c, g in sub.groupby("config"):
            counts.append({"dataset": keys[0], "score": keys[1], "config": c, "n_cells": len(g),
                           "cells_consistent_higher": int((g["class"] == "consistent_higher").sum()),
                           "cells_consistent_lower": int((g["class"] == "consistent_lower").sum()),
                           "cells_not_consistent": int((g["class"] == "not_consistent").sum()),
                           "seed_runs": int(g["n_seeds"].sum()), "seed_runs_higher": int(g["n_higher"].sum()),
                           "seed_runs_lower": int(g["n_lower"].sum()),
                           "seed_runs_undetermined": int(g["n_undetermined"].sum()),
                           "seed_runs_point_higher": int(g["n_point_higher"].sum())})
    pd.DataFrame(counts).to_csv(args.out_dir / "rq2_spearman_counts.csv", index=False)

    a3 = []
    for lv in LEVELS:
        for ds, v in vals[("a3", "012", lv)].items():
            a0 = vals[("a0", "0", lv)][ds]
            for m in A3_METRICS:
                lo, hi = ci_percentile(v.samples[m] - a0.samples[m], level)
                a3.append({"level": lv, "dataset": ds, "metric": m, "a3": v.value[m], "a0": a0.value[m],
                           "diff": v.value[m] - a0.value[m], "ci_low": lo, "ci_high": hi, "sign": sign_of(lo, hi),
                           "note": "exploratory: added after results were seen"})
    pd.DataFrame(a3).to_csv(args.out_dir / "a3_vs_a0.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
