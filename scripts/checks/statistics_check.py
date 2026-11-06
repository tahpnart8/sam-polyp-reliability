"""Check how the paired comparisons are built: pairing, shared runs, consistency reading. Writes a report.

Facts checked at run time, from the code and the stored files:
- the bootstrap index matrix depends only on dataset and size, so all runs of a dataset use the same resamples;
- every run holds the same images in the same order, so an index means the same image in every run;
- consistency flags in paired_consistency.csv follow from paired.csv;
- the sign of each point difference agrees with its interval when the interval excludes 0;
- which Table 5 comparisons reach k of 3 with mixed directions.

    python scripts/checks/statistics_check.py --metrics-dir results/metrics --report results/logs/statistics_check.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.bootstrap import indices_for_dataset
from samrel.config import load_config

TABLE5 = ("dice", "ece_roi", "aurc_iou_pred", "spearman_iou_pred")
EXTERNAL = ("colondb", "etis")


def family(name_a: str, name_b: str) -> str:
    return f"{name_a.split('_')[0]}_vs_{name_b.split('_')[0]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)
    m, pdir = args.metrics_dir, args.metrics_dir / "perturb"
    cfg = load_config("base")
    B, root = cfg["bootstrap"]["n_resamples"], cfg["root_seed"]
    runs = pd.read_csv(m / "runs.csv", dtype=str)
    pruns = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(pdir.glob("runs_*.csv"))], ignore_index=True)

    def table(path):
        df = pd.read_csv(path, usecols=["dataset", "image_id", "excluded"])
        return df[~df.excluded.astype(bool)].sort_values("image_id")

    tables = {r.run_id: table(m / "per_image" / f"{r.run_id}.csv") for r in runs.itertuples()}
    ptables = {r.run_id: table(pdir / "per_image" / f"{r.run_id}.csv") for r in pruns.itertuples()}
    first = next(iter(tables.values()))
    same_main = all(t.image_id.tolist() == first.image_id.tolist() for t in tables.values())
    ext_ids = first[first.dataset.isin(EXTERNAL)].image_id.tolist()
    pfirst = next(iter(ptables.values()))
    same_pert = all(t.image_id.tolist() == pfirst.image_id.tolist() for t in ptables.values())
    same_cross = sorted(pfirst.image_id) == sorted(ext_ids)

    a = {ds: indices_for_dataset(ds, int((first.dataset == ds).sum()), B, root) for ds in EXTERNAL}
    b = {ds: indices_for_dataset(ds, int((pfirst.dataset == ds).sum()), B, root) for ds in EXTERNAL}
    same_idx = all(np.array_equal(a[ds], b[ds]) for ds in EXTERNAL)

    paired = pd.read_csv(m / "paired.csv").dropna(subset=["diff"])
    paired["family"] = [family(x, y) for x, y in zip(paired.config_a, paired.config_b)]
    pairs = paired.groupby("family").apply(lambda g: sorted({f"{x}-{y}" for x, y in zip(g.config_a, g.config_b)}), include_groups=False).to_dict()
    ex = paired[paired.excludes_zero.astype(bool)]
    sign_ci = np.where(ex.ci_low > 0, 1, -1)
    n_disagree = int((np.sign(ex["diff"]) != sign_ci).sum())
    disagree = ex[np.sign(ex["diff"]) != sign_ci]

    pc = pd.read_csv(m / "paired_consistency.csv")
    n_bad = 0
    for r in pc.itertuples():
        g = paired[(paired.family == r.family) & (paired.dataset == r.dataset) & (paired.metric == r.metric)]
        e = g[g.excludes_zero.astype(bool)]
        n_bad += int(bool(r.consistent) != (len(g) == 3 and len(e) == 3 and len(set(np.sign(e["diff"]))) == 1))
    t5 = paired[paired.dataset.isin(EXTERNAL) & paired.metric.isin(TABLE5)]
    mixed = []
    for (fam, ds, metric), g in t5.groupby(["family", "dataset", "metric"]):
        e = g[g.excludes_zero.astype(bool)]
        if len(e) and len(e) < 3:
            ups, downs = int((e.ci_low > 0).sum()), int((e.ci_high < 0).sum())
            mixed.append((fam, ds, metric, len(e), ups, downs))
    a3 = paired[paired.family == "a3_vs_a2"]
    lines = ["# How the paired comparisons are built", "",
             "## Pairing", "",
             f"- Image order: all {len(tables)} main runs hold the same images in the same order: {same_main}. All {len(ptables)} perturbation runs: {same_pert}. "
             f"Perturbation images are exactly the ColonDB and ETIS images of the main runs: {same_cross}.",
             f"- Index matrix: `samrel.bootstrap.indices_for_dataset` seeds on the root seed, the label `bootstrap` and the dataset only, so every run of a dataset, at every perturbation level, uses one matrix ({B} x n). Main and perturbation matrices identical: {same_idx}.",
             "- A paired difference is computed per resample on the same image indices, in `samrel.aggregate.paired` and `samrel.perturb.diff_row`. Images stay paired.", "",
             "## Which runs enter each comparison", ""]
    for fam, p in pairs.items():
        lines.append(f"- `{fam}`: {'; '.join(p)}")
    lines += ["",
              "Consequences for reading k of 3:",
              "- A1 - A0 and A2 - A0 use one A0 run, which is deterministic. The three differences share A0 and vary only through the fine-tuned model.",
              "- A3 - A2 uses one A3 and each of its three members. A3 is the pixel-wise mean of those members, so each difference compares a model with an ensemble that contains it. The three comparisons share A3 and are not three independent repetitions.",
              "- A2 - A1 pairs runs with the same seed label. The seed sets minibatch order and, for A2, the LoRA initialisation; the pairing is a convention and the two runs share no trained weights.",
              "- All comparisons on a dataset use the same resampled images, so intervals in different cells are correlated. No interval is adjusted for multiple comparisons. Images are resampled, not videos.", "",
              "## Consistency rule", "",
              f"- `paired_consistency.csv` flags recomputed from `paired.csv`: {n_bad} of {len(pc)} differ.",
              f"- Comparisons whose interval excludes 0 while the point difference has the opposite sign: {n_disagree} of {len(ex)}" + (
                  " (" + "; ".join(f"{r.config_a}-{r.config_b} {r.dataset} {r.metric}" for r in disagree.itertuples()) + ")." if n_disagree else "."),
              "- `consistent` needs all three intervals to exclude 0 with one sign. k counts the intervals that exclude 0 and does not require one sign when k is below 3.", "",
              "## Table 5 cells with k of 1 or 2", "",
              "| Comparison | Dataset | Metric | k of 3 | Above 0 | Below 0 |", "|---|---|---|---|---|---|"]
    lines += [f"| {f} | {d} | {me} | {k} | {u} | {dn} |" for f, d, me, k, u, dn in mixed]
    lines += ["", f"`a3_vs_a2` rows checked: {len(a3)}."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
