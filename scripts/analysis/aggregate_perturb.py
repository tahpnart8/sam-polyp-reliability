"""Prompt-perturbation tables (PROTOCOL §14.5, §14.7, §14.8) from results/metrics/perturb/.

Reads every runs_*.csv index, the per-image and ECE-bin files of each run, then writes:
  summary.csv       value + CI per run x dataset x metric, failure/success counts, auroc_eligible
  delta.csv         Δ = level k − pp00 within a run family (same config and seed)
  did.csv           DiD = Δ(config) − Δ(A0); A1/A2 per seed, A3 directly
  rq2_pairs.csv     the four comparison families of paired_consistency.csv, at every level
  delta_consistency.csv, did_consistency.csv, rq2_consistency.csv   consistency reading over seeds (§7.8)
Usage:
    python scripts/analysis/aggregate_perturb.py --perturb-dir results/metrics/perturb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.bootstrap import ci_percentile
from samrel.config import load_config
from samrel.perturb import (FAMILIES, LEVELS, RQ2_METRICS, consistency, diff_row, reading, run_values,
                            threshold_of)


def load_runs(pdir: Path) -> pd.DataFrame:
    runs = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(pdir.glob("runs_*.csv"))], ignore_index=True)
    if runs.run_id.duplicated().any():
        raise SystemExit("STOP: duplicated run_id across runs_*.csv")
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--n-resamples", type=int, help="tests only; default bootstrap.n_resamples (1000)")
    args = ap.parse_args(argv)
    cfg = load_config("base")
    B, level, root = args.n_resamples or cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]
    pdir = args.perturb_dir
    runs = load_runs(pdir)

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
            for m, x in v.value.items():
                lo, hi = ci_percentile(v.samples[m], level)
                rows.append({"config": c, "seed": s, "level": lv, "dataset": ds, "metric": m, "value": x,
                             "ci_low": lo, "ci_high": hi, **v.counts,
                             "auroc_eligible": v.eligible(m) if threshold_of(m) else np.nan,
                             "n_boot_undefined": int(np.isnan(v.samples[m]).sum())})
    pd.DataFrame(rows).to_csv(pdir / "summary.csv", index=False)

    def pair(a, b, ds, m):
        va, vb = vals[a][ds], vals[b][ds]
        return diff_row(va.samples[m], vb.samples[m], va.value[m], vb.value[m], level), va.eligible(m) and vb.eligible(m)

    seeds = {c: sorted({s for (cc, s, _) in vals if cc == c}) for c in ("a0", "a1", "a2", "a3")}
    datasets = sorted(next(iter(vals.values())))
    metrics = sorted(next(iter(next(iter(vals.values())).values())).value)
    delta, did, rq2 = [], [], []
    for c, ss in seeds.items():
        for s in ss:
            for lv in LEVELS[1:]:
                for ds in datasets:
                    for m in metrics:
                        d, ok = pair((c, s, lv), (c, s, "pp00"), ds, m)
                        delta.append({"config": c, "seed": s, "level": lv, "dataset": ds, "metric": m, **d, "eligible": ok})
                        if c == "a0" or m.endswith("pairwise_dice"):
                            continue
                        a0 = (vals[("a0", "0", lv)][ds], vals[("a0", "0", "pp00")][ds])
                        cc = (vals[(c, s, lv)][ds], vals[(c, s, "pp00")][ds])
                        dc = cc[0].samples[m] - cc[1].samples[m]
                        da = a0[0].samples[m] - a0[1].samples[m]
                        r = diff_row(dc, da, cc[0].value[m] - cc[1].value[m], a0[0].value[m] - a0[1].value[m], level)
                        ok4 = all(x.eligible(m) for x in (*a0, *cc))
                        did.append({"config": c, "seed": s, "level": lv, "dataset": ds, "metric": m, **r, "eligible": ok4})
    fam_pairs = {"a1_vs_a0": [(("a1", s), ("a0", "0")) for s in seeds["a1"]],
                 "a2_vs_a0": [(("a2", s), ("a0", "0")) for s in seeds["a2"]],
                 "a2_vs_a1": [(("a2", s), ("a1", s)) for s in seeds["a2"]],
                 "a3_vs_a2": [(("a3", "012"), ("a2", s)) for s in seeds["a2"]]}
    for fam in FAMILIES:
        for (ca, sa), (cb, sb) in fam_pairs[fam]:
            for lv in LEVELS:
                for ds in datasets:
                    for m in RQ2_METRICS:
                        d, ok = pair((ca, sa, lv), (cb, sb, lv), ds, m)
                        rq2.append({"family": fam, "seed_a": sa, "seed_b": sb, "level": lv, "dataset": ds, "metric": m,
                                    **d, "eligible": ok})
    delta, did, rq2 = pd.DataFrame(delta), pd.DataFrame(did), pd.DataFrame(rq2)
    delta.to_csv(pdir / "delta.csv", index=False)
    did.to_csv(pdir / "did.csv", index=False)
    rq2.to_csv(pdir / "rq2_pairs.csv", index=False)

    def summarise(df, keys, n_required):
        out = []
        for k, g in df.groupby(keys, sort=True):
            cls = consistency(g["sign"].tolist(), g["eligible"].tolist(), n_required)
            out.append({**dict(zip(keys, k)), "class": cls, "n_seeds": len(g),
                        "n_excluding_zero": int((g["sign"] != 0).sum()), "diff_min": g["diff"].min(),
                        "diff_max": g["diff"].max()})
        return pd.DataFrame(out)

    req = lambda c: 1 if c in ("a0", "a3") else 3  # noqa: E731
    dc = pd.concat([summarise(g, ["config", "level", "dataset", "metric"], req(c)) for c, g in delta.groupby("config")])
    dc.to_csv(pdir / "delta_consistency.csv", index=False)
    dd = pd.concat([summarise(g, ["config", "level", "dataset", "metric"], req(c)) for c, g in did.groupby("config")])
    dd["reading"] = [reading(c, m) for c, m in zip(dd["class"], dd["metric"])]
    dd.to_csv(pdir / "did_consistency.csv", index=False)
    rc = summarise(rq2, ["family", "level", "dataset", "metric"], 3)
    base = rc[rc.level == "pp00"].set_index(["family", "dataset", "metric"])["class"]
    rc["class_pp00"] = [base.get((f, d, m)) for f, d, m in zip(rc.family, rc.dataset, rc.metric)]
    rc["same_as_pp00"] = rc["class"] == rc["class_pp00"]
    rc.to_csv(pdir / "rq2_consistency.csv", index=False)
    print(dd[dd.metric.isin(["dice", "spearman_iou_pred", "aurc_iou_pred"])].to_string(index=False))
    print(rc[rc.metric.isin(["aurc_iou_pred", "spearman_iou_pred"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
