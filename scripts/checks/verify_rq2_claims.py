"""Check each statement about RQ2 and prompt perturbation in the manuscript against the output files.

Every claim is recomputed from a file, not copied from a table of the paper. Verdict: match, or
differs with what the file shows. Writes a CSV and a Markdown report; exits with code 1 on any difference.

    python scripts/checks/verify_rq2_claims.py --metrics-dir results/metrics \
        --out-csv results/data_checks/rq2_claims_check.csv --report results/logs/rq2_claims_check.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ("pp00", "pp10", "pp20", "pp30")
PERTURBED = LEVELS[1:]


def f3(x: float) -> str:
    return f"{x:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)
    m, p = args.metrics_dir, args.metrics_dir / "perturb"
    summary = pd.read_csv(m / "summary.csv", dtype={"seed": str})
    secondary = pd.read_csv(m / "summary_secondary.csv", dtype={"seed": str})
    stab = pd.read_csv(m / "seed_stability.csv")
    pc = pd.read_csv(m / "paired_consistency.csv")
    psum = pd.read_csv(p / "summary.csv", dtype={"seed": str})
    delta = pd.read_csv(p / "delta_consistency.csv")
    delta_raw = pd.read_csv(p / "delta.csv", dtype={"seed": str})
    did = pd.read_csv(p / "did_consistency.csv")
    rq2 = pd.read_csv(p / "rq2_consistency.csv")
    levels = pd.read_csv(p / "prompt_levels.csv")
    cells = pd.read_csv(p / "rq2_spearman_cells.csv")
    seeds = pd.read_csv(p / "rq2_spearman_by_seed.csv", dtype={"seed": str})
    auroc = pd.read_csv(p / "auroc_cells.csv", dtype={"seed": str, "threshold": str})
    comp = pd.read_csv(p / "auroc_comparisons.csv", dtype={"threshold": str})
    rows = []

    def add(claim, source, observed, ok):
        rows.append({"id": len(rows) + 1, "claim": claim, "source": source, "observed": observed, "verdict": "match" if ok else "differs"})

    def val(df, **kw):
        g = df
        for k, v in kw.items():
            g = g[g[k] == v]
        return g

    def fails(frame, metric, ds, cfg):
        return val(frame, metric=metric, dataset=ds, config=cfg).sort_values("seed")["value"].astype(int).tolist()

    f = {(ds, c): fails(summary, "n_failures", ds, c) for ds in ("colondb", "etis") for c in ("a0", "a1", "a2", "a3")}
    add("Failures (Dice < 0.5): zero-shot SAM 40 ColonDB images, 1 ETIS image", "metrics/summary.csv, n_failures",
        f"{f[('colondb', 'a0')]}, {f[('etis', 'a0')]}", f[("colondb", "a0")] == [40] and f[("etis", "a0")] == [1])
    add("A1 failed on one ColonDB image per seed", "metrics/summary.csv, n_failures", str(f[("colondb", "a1")]), f[("colondb", "a1")] == [1, 1, 1])
    none = all(sum(f[(d, c)]) == 0 for d in ("colondb", "etis") for c in ("a2", "a3"))
    add("A2 and A3 failed on no image", "metrics/summary.csv, n_failures", str({k: v for k, v in f.items() if k[1] in ("a2", "a3")}), none)
    s7 = {(d, c): fails(secondary, "n_failures_07", d, c) for d in ("colondb", "etis") for c in ("a0", "a1", "a2", "a3")}
    mx = max(max(v) for k, v in s7.items() if k[1] != "a0")
    add("Threshold 0.7: A0 43 and 6 failures, no fine-tuned model more than 2", "metrics/summary_secondary.csv, n_failures_07",
        f"A0 {s7[('colondb', 'a0')]}, {s7[('etis', 'a0')]}; fine-tuned max {mx}", s7[("colondb", "a0")] == [43] and s7[("etis", "a0")] == [6] and mx <= 2)
    e0 = val(auroc, level="pp00", eligible=True)
    v = val(summary, config="a0", dataset="colondb", metric="auroc_iou_pred").iloc[0]
    add("Oracle boxes: AUROC usable only for zero-shot SAM on ColonDB; IoU head 0.951 [0.922, 0.976]", "metrics/perturb/auroc_cells.csv; metrics/summary.csv",
        f"eligible P0 cells: {sorted(set(zip(e0.config, e0.dataset)))}; {f3(v.value)} [{f3(v.ci_low)}, {f3(v.ci_high)}]",
        set(zip(e0.config, e0.dataset)) == {("a0", "colondb")} and (f3(v.value), f3(v.ci_low), f3(v.ci_high)) == ("0.951", "0.922", "0.976"))

    def cons(fam, ds, metric):
        return val(pc, family=fam, dataset=ds, metric=metric).iloc[0]

    add("Fine-tuning lowered AURC consistently for A1 and A2 on both sets", "metrics/paired_consistency.csv, aurc_iou_pred",
        "; ".join(f"{a} {d} max {cons(a, d, 'aurc_iou_pred').diff_max:+.3f}" for a in ("a1_vs_a0", "a2_vs_a0") for d in ("colondb", "etis")),
        all(cons(a, d, "aurc_iou_pred").consistent and cons(a, d, "aurc_iou_pred").diff_max < 0 for a in ("a1_vs_a0", "a2_vs_a0") for d in ("colondb", "etis")))
    r0 = {d: val(summary, config="a0", dataset=d, metric="spearman_iou_pred").iloc[0].value for d in ("colondb", "etis")}
    add("IoU head: rho 0.724 and 0.743 for zero-shot SAM", "metrics/summary.csv", f"{f3(r0['colondb'])}, {f3(r0['etis'])}", (f3(r0["colondb"]), f3(r0["etis"])) == ("0.724", "0.743"))
    a1c, a1e = cons("a1_vs_a0", "colondb", "spearman_iou_pred"), cons("a1_vs_a0", "etis", "spearman_iou_pred")
    add("rho fell consistently for A1, by 0.08 to 0.13 on both sets", "metrics/paired_consistency.csv, spearman_iou_pred",
        f"ColonDB {a1c.diff_min:+.4f} to {a1c.diff_max:+.4f}; ETIS {a1e.diff_min:+.4f} to {a1e.diff_max:+.4f}",
        bool(a1c.consistent and a1e.consistent) and all(0.075 <= abs(x) < 0.135 for x in (a1c.diff_min, a1c.diff_max, a1e.diff_min, a1e.diff_max)))
    a2c, a2e = cons("a2_vs_a0", "colondb", "spearman_iou_pred"), cons("a2_vs_a0", "etis", "spearman_iou_pred")
    add("rho of A2 did not differ consistently from A0 (IoU head)", "metrics/paired_consistency.csv", f"k {a2c.n_excluding_zero}/3, {a2e.n_excluding_zero}/3", not a2c.consistent and not a2e.consistent)
    mi = cons("a2_vs_a0", "etis", "spearman_mean_inside")
    st = val(stab, config="a2", dataset="etis", metric="spearman_mean_inside").iloc[0]
    a0mi = val(summary, config="a0", dataset="etis", metric="spearman_mean_inside").iloc[0].value
    add("Mean pixel confidence: A2 ranked ETIS consistently worse, rho 0.231 +- 0.098 versus 0.489", "metrics/paired_consistency.csv; seed_stability.csv; summary.csv",
        f"consistent {bool(mi.consistent)}, {mi.diff_min:+.3f} to {mi.diff_max:+.3f}; {f3(st['mean'])} +- {f3(st['sd'])} versus {f3(a0mi)}",
        bool(mi.consistent) and mi.diff_max < 0 and (f3(st["mean"]), f3(st["sd"]), f3(a0mi)) == ("0.231", "0.098", "0.489"))
    med = val(levels, dataset="colondb").set_index("level")["box_iou_median"]
    add("Median IoU with oracle boxes 0.81, 0.67, 0.54 at P1-P3 (ColonDB)", "metrics/perturb/prompt_levels.csv", ", ".join(f"{med[l]:.2f}" for l in PERTURBED),
        [f"{med[l]:.2f}" for l in PERTURBED] == ["0.81", "0.67", "0.54"])
    dd = val(delta, metric="dice")
    exceptions = dd[dd["class"] != "consistent_-"]
    add("Dice fell with the level in every configuration, consistently except A0 on ColonDB at P1", "metrics/perturb/delta_consistency.csv, dice",
        f"{len(dd) - len(exceptions)}/{len(dd)} consistent; exceptions {[(r.config, r.dataset, r.level) for r in exceptions.itertuples()]}",
        [(r.config, r.dataset, r.level) for r in exceptions.itertuples()] == [("a0", "colondb", "pp10")])
    d3 = {c: val(psum, config=c, level="pp30", dataset="colondb", metric="dice").iloc[0].value for c in ("a0", "a3")}
    add("Dice 0.623 (A0) and 0.747 (A3) on ColonDB at P3", "metrics/perturb/summary.csv", f"{f3(d3['a0'])}, {f3(d3['a3'])}", (f3(d3["a0"]), f3(d3["a3"])) == ("0.623", "0.747"))
    dice_did = val(did, metric="dice")
    more = sorted({(r.dataset, r.level) for r in dice_did.itertuples() if r.reading == "more_sensitive"})
    less = sorted({(r.dataset, r.level) for r in dice_did.itertuples() if r.reading == "less_sensitive"})
    add("Dice loss of fine-tuned models against A0: more sensitive only on ETIS at P2, otherwise not distinguishable", "metrics/perturb/did_consistency.csv, dice",
        f"more sensitive {more}; less sensitive {less}; others not_distinguishable", more == [("etis", "pp20")] and not less)
    worse = 0
    total = 0
    for ds in ("colondb", "etis"):
        for lv in PERTURBED:
            base = val(delta_raw, config="a0", dataset=ds, level=lv, metric="dice").iloc[0]["diff"]
            g = val(delta_raw, dataset=ds, level=lv, metric="dice")
            g = g[g.config != "a0"]
            total += len(g)
            worse += int((g["diff"] < base).sum())
    add("Point estimates only: most fine-tuned runs lost at least as much Dice as A0, so \"not distinguishable\" must not be read as \"lost no more\"", "metrics/perturb/delta.csv, dice",
        f"{worse} of {total} fine-tuned runs lost more Dice than A0 in point estimate", worse > total / 2)
    iou_did = val(did, metric="iou")
    more_i = sorted({(r.dataset, r.level) for r in iou_did.itertuples() if r.reading == "more_sensitive"})
    add("Fine-tuned models lost more IoU from P2 on (P1 on ETIS)", "metrics/perturb/did_consistency.csv, iou", str(more_i),
        more_i == [("colondb", "pp20"), ("colondb", "pp30"), ("etis", "pp10"), ("etis", "pp20"), ("etis", "pp30")])
    high = cells[cells.config.isin(["a1", "a2"]) & (cells["class"] == "consistent_higher")]
    add("A1 and A2 never had consistently higher rho than A0, with either score", "metrics/perturb/rq2_spearman_cells.csv", f"{len(high)} of {int(cells.config.isin(['a1', 'a2']).sum())} cells", high.empty)
    low = cells[(cells.score == "iou_pred") & (cells["class"] == "consistent_lower") & cells.config.isin(["a1", "a2"])]
    got = sorted((r.config, r.dataset, r.level) for r in low.itertuples())
    want = sorted([("a1", "colondb", "pp20"), ("a2", "colondb", "pp20"), ("a1", "colondb", "pp00")]
                  + [("a1", "etis", v) for v in LEVELS] + [("a2", "etis", v) for v in PERTURBED])
    add("IoU head: both consistently lower on ColonDB at P2; on ETIS A1 at every level, A2 from P1 (A1 on ColonDB at P0 is stated separately)", "metrics/perturb/rq2_spearman_cells.csv", str(got), got == want)
    above = seeds[seeds["sign"] == 1]
    listed = sorted((r.config, r.seed, r.dataset, r.score, r.level, f"{r.diff_vs_a0:+.3f}") for r in above.itertuples())
    add("Paired intervals above A0 occurred only for A2 seed 1 at P0 (+0.113) and A3 at P3 (+0.111), mean confidence on ColonDB", "metrics/perturb/rq2_spearman_by_seed.csv", str(listed),
        listed == [("a2", "1", "colondb", "mean_inside", "pp00", "+0.113"), ("a3", "012", "colondb", "mean_inside", "pp30", "+0.111")])
    r = val(rq2, metric="aurc_iou_pred", dataset="etis")
    cl = {(fam, lv): val(r, family=fam, level=lv).iloc[0]["class"] for fam in ("a1_vs_a0", "a2_vs_a0") for lv in ("pp20", "pp30")}
    dice_ok = True
    obs = []
    for lv in ("pp20", "pp30"):
        base = val(psum, config="a0", level=lv, dataset="etis", metric="dice").iloc[0].value
        g = val(psum, level=lv, dataset="etis", metric="dice")
        g = g[g.config.isin(["a1", "a2"])]
        dice_ok &= bool((g.value >= base).all())
        obs.append(f"{lv}: A0 {f3(base)}, A1/A2 min {f3(g.value.min())}")
    add("ETIS P2-P3: A1 and A2 consistently higher AURC than A0 despite no lower mean Dice", "metrics/perturb/rq2_consistency.csv; summary.csv", f"{set(cl.values())}; {'; '.join(obs)}",
        set(cl.values()) == {"consistent_+"} and dice_ok)
    e = val(auroc, dataset="etis", threshold="0.7", level="pp30", score="iou_pred")
    a0e = e[e.config == "a0"].iloc[0]
    fine = e[e.config.isin(["a1", "a2"])]
    fc = {x["comparison"]: x["class"] for _, x in val(comp, source="rq2_consistency.csv", dataset="etis", threshold="0.7", level="pp30").iterrows()}
    add("AUROC at threshold 0.7, ETIS, P3: A0 0.914 [0.854, 0.960]; A1 and A2 consistently lower (0.785 to 0.822)", "metrics/perturb/auroc_cells.csv; auroc_comparisons.csv",
        f"{f3(a0e.value)} [{f3(a0e.ci_low)}, {f3(a0e.ci_high)}]; {f3(fine.value.min())} to {f3(fine.value.max())}; {fc}; eligible {bool(e.eligible.all())}",
        (f3(a0e.value), f3(a0e.ci_low), f3(a0e.ci_high)) == ("0.914", "0.854", "0.960") and (f3(fine.value.min()), f3(fine.value.max())) == ("0.785", "0.822")
        and fc.get("a1_vs_a0") == "consistent_-" and fc.get("a2_vs_a0") == "consistent_-" and bool(e.eligible.all()))
    others = comp[(comp.source.isin(["rq2_consistency.csv", "added: A3 versus A0, exploratory"])) & (comp["class"] != "insufficient_cases")]
    others = others[~((others.dataset == "etis") & (others.threshold == "0.7") & (others.level == "pp30") & others.comparison.isin(["a1_vs_a0", "a2_vs_a0"]))]
    add("Other eligible AUROC comparisons between configurations were inconsistent", "metrics/perturb/auroc_comparisons.csv", f"{len(others)} comparisons, classes {sorted(set(others['class']))}",
        set(others["class"]) == {"not_consistent"})
    dl = comp[(comp.source == "delta_consistency.csv") & (comp["class"] != "insufficient_cases")]
    did_ok = comp[(comp.source == "did_consistency.csv") & (comp["class"] != "insufficient_cases")]
    add("Changes in AUROC from P0 are computable only for A0 on ColonDB (few P0 failures elsewhere); no difference from A0 is computable", "metrics/perturb/auroc_comparisons.csv",
        f"{len(did_ok)} differences from A0 usable; changes from P0 usable for {sorted(set(dl.comparison))} on {sorted(set(dl.dataset))} ({len(dl)} rows, classes {sorted(set(dl['class']))})",
        did_ok.empty and set(dl.comparison) == {"a0"} and set(dl.dataset) == {"colondb"})
    higher_a3 = cells[(cells.config == "a3") & (cells["class"] == "consistent_higher")]
    add("Under perturbed boxes fine-tuned models did not consistently rank better than zero-shot SAM (A1, A2 all cells; A3 is one run)", "metrics/perturb/rq2_spearman_cells.csv",
        f"A1, A2: 0 cells higher. A3: {[(x.dataset, x.score, x.level) for x in higher_a3.itertuples()]} of 16 cells, one run, exploratory",
        high.empty and len(higher_a3) == 1)
    res = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    lines = ["# Statements on RQ2 and prompt perturbation against the output files", "", "| # | Statement | File | What the file shows | Verdict |", "|---|---|---|---|---|"]
    lines += [f"| {r.id} | {r.claim} | {r.source} | {r.observed} | {r.verdict} |" for r in res.itertuples()]
    lines += ["", f"{int((res.verdict == 'match').sum())} match, {int((res.verdict == 'differs').sum())} differ, of {len(res)}."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return int((res.verdict != "match").any())


if __name__ == "__main__":
    raise SystemExit(main())
