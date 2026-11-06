"""Every AUROC cell of the perturbation analysis with its failure and success counts and eligibility.

A cell is config x seed x level x dataset x threshold x score; it may support a conclusion only with at
least 20 failures and 20 successes (§14.8). Reads perturb/summary.csv and the three consistency tables, and adds
A3 versus A0 (exploratory, not in those tables) with the same bootstrap indices (§7.4).

Writes auroc_cells.csv (all cells) and auroc_comparisons.csv (every AUROC comparison and its reading).

    python scripts/checks/auroc_cells.py --perturb-dir results/metrics/perturb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from samrel.bootstrap import ci_percentile, indices_for_dataset
from samrel.config import load_config
from samrel.perturb import AUROC_MIN_CASES, LEVELS, sign_of


def parse(metric: str) -> tuple[str, str]:
    head, score = metric.split("_", 1)
    return ("0.7" if head == "auroc07" else "0.5"), score


def auroc_rows(conf: np.ndarray, failed: np.ndarray) -> np.ndarray:
    """AUROC with failure as the positive class and -confidence as the score, row by row (Mann-Whitney)."""
    ranks = rankdata(-conf, axis=1)
    n_pos = failed.sum(1)
    n_neg = failed.shape[1] - n_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((ranks * failed).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    out[(n_pos == 0) | (n_neg == 0)] = np.nan
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, help="default: --perturb-dir")
    args = ap.parse_args(argv)
    pdir, out_dir = args.perturb_dir, args.out_dir or args.perturb_dir
    cfg = load_config("base")
    B, level, root = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci_level"], cfg["root_seed"]

    s = pd.read_csv(pdir / "summary.csv", dtype={"seed": str})
    s = s[s.metric.str.startswith("auroc")].copy()
    parsed = [parse(m) for m in s.metric]
    s["threshold"], s["score"] = [p[0] for p in parsed], [p[1] for p in parsed]
    s["n_fail"] = np.where(s.threshold == "0.5", s.n_fail_05, s.n_fail_07)
    s["n_success"] = np.where(s.threshold == "0.5", s.n_success_05, s.n_success_07)
    s["eligible"] = (s.n_fail >= AUROC_MIN_CASES) & (s.n_success >= AUROC_MIN_CASES)
    if (s["eligible"] != s["auroc_eligible"].astype(bool)).any():
        raise SystemExit("STOP: eligibility differs from summary.csv")
    cells = s[["dataset", "threshold", "score", "level", "config", "seed", "n_images", "n_fail", "n_success", "eligible",
               "value", "ci_low", "ci_high", "n_boot_undefined"]].sort_values(["dataset", "threshold", "score", "level", "config", "seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "auroc_cells.csv", index=False)

    rows = []
    for name, keys in (("rq2_consistency.csv", "family"), ("delta_consistency.csv", "config"), ("did_consistency.csv", "config")):
        t = pd.read_csv(pdir / name)
        t = t[t.metric.str.startswith("auroc")]
        for r in t.itertuples():
            th, score = parse(r.metric)
            rows.append({"source": name, "comparison": getattr(r, keys), "level": r.level, "dataset": r.dataset, "threshold": th, "score": score,
                         "class": t.loc[r.Index, "class"], "n_seeds": r.n_seeds, "n_excluding_zero": r.n_excluding_zero,
                         "diff_min": r.diff_min, "diff_max": r.diff_max})

    runs = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(pdir.glob("runs_*.csv"))], ignore_index=True)
    ids = {(r.config, r.level): r.run_id for r in runs.itertuples() if r.config in ("a0", "a3")}
    for lv in LEVELS:
        a0 = pd.read_csv(pdir / "per_image" / f"{ids[('a0', lv)]}.csv").sort_values("image_id")
        a3 = pd.read_csv(pdir / "per_image" / f"{ids[('a3', lv)]}.csv").sort_values("image_id")
        for ds in sorted(a0.dataset.unique()):
            g0, g3 = a0[a0.dataset == ds].reset_index(drop=True), a3[a3.dataset == ds].reset_index(drop=True)
            if g0.image_id.tolist() != g3.image_id.tolist():
                raise SystemExit("STOP: A0 and A3 hold different images")
            idx = indices_for_dataset(ds, len(g0), B, root)
            for th, limit in (("0.5", 0.5), ("0.7", 0.7)):
                f0, f3 = g0.dice.to_numpy() < limit, g3.dice.to_numpy() < limit
                ok = all(min(f.sum(), (~f).sum()) >= AUROC_MIN_CASES for f in (f0, f3))
                c0, c3 = g0.conf_iou_pred.to_numpy(), g3.conf_iou_pred.to_numpy()
                full = np.arange(len(g0))[None, :]
                d = auroc_rows(c3[full], f3[full])[0] - auroc_rows(c0[full], f0[full])[0]
                lo, hi = ci_percentile(auroc_rows(c3[idx], f3[idx]) - auroc_rows(c0[idx], f0[idx]), level)
                sign = sign_of(lo, hi)
                cls = "insufficient_cases" if not ok else {1: "higher", -1: "lower", 0: "not_consistent"}[sign]
                rows.append({"source": "added: A3 versus A0, exploratory", "comparison": "a3_vs_a0", "level": lv, "dataset": ds, "threshold": th,
                             "score": "iou_pred", "class": cls, "n_seeds": 1, "n_excluding_zero": int(sign != 0),
                             "diff_min": d, "diff_max": d})
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "auroc_comparisons.csv", index=False)
    print("cells", len(cells), "eligible", int(cells.eligible.sum()))
    print(cells[cells.eligible].groupby(["dataset", "threshold", "level"]).size().to_string())
    print(comp[~comp["class"].eq("insufficient_cases")].drop(columns=["diff_min", "diff_max"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
