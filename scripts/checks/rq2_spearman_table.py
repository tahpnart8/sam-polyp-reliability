"""Readable version of the per-seed Spearman tables: one table per dataset and score.

Reads rq2_spearman_by_seed.csv, rq2_spearman_cells.csv and rq2_spearman_counts.csv and writes one Markdown file.
Each cell gives rho, the paired difference with A0 and its 95% interval, and a mark:
  [higher] interval above 0, [lower] interval below 0, [undetermined] interval contains 0.

    python scripts/checks/rq2_spearman_table.py --perturb-dir results/metrics/perturb \
        --out results/metrics/perturb/rq2_spearman_table.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LEVELS = ("pp00", "pp10", "pp20", "pp30")
NAMES = {"pp00": "P0 (oracle)", "pp10": "P1 (0.10)", "pp20": "P2 (0.20)", "pp30": "P3 (0.30)"}
DATASETS = {"colondb": "CVC-ColonDB", "etis": "ETIS-LaribPolypDB"}
SCORES = {"iou_pred": "IoU-head score", "mean_inside": "mean pixel confidence"}
MARK = {"higher_than_a0": "higher", "lower_than_a0": "lower", "undetermined": "undetermined"}
CLASS = {"consistent_higher": "higher", "consistent_lower": "lower", "not_consistent": "not consistent"}


def run_cell(r) -> str:
    if r.config == "a0":
        return f"{r.rho:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    return f"{r.rho:.3f}; {r.diff_vs_a0:+.3f} [{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}] {MARK[r.reading]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    seeds = pd.read_csv(args.perturb_dir / "rq2_spearman_by_seed.csv", dtype={"seed": str})
    cells = pd.read_csv(args.perturb_dir / "rq2_spearman_cells.csv")
    counts = pd.read_csv(args.perturb_dir / "rq2_spearman_counts.csv")
    lines = ["# Spearman rho between confidence and Dice, by dataset, perturbation level, score and seed", "",
             "A0: rho [95% bootstrap interval]. Other rows: rho; paired difference from A0 [95% paired bootstrap interval], then a mark: "
             "**higher** or **lower** when the interval excludes 0, **undetermined** when it contains 0. Undetermined does not mean equivalent. "
             "Intervals are per image, not adjusted for multiple comparisons, and do not account for frames from the same video.",
             "", "A larger rho means better ranking of images by segmentation quality. A3 versus A0 and mean pixel confidence under perturbed boxes "
             "are exploratory analyses added after the results were seen.", ""]
    for ds, ds_name in DATASETS.items():
        for score, score_name in SCORES.items():
            lines += [f"## {ds_name}, {score_name}", "", "| Run | " + " | ".join(NAMES[v] for v in LEVELS) + " |", "|---|---|---|---|---|"]
            sub = seeds[(seeds.dataset == ds) & (seeds.score == score)]
            for (cfg, seed), g in sub.groupby(["config", "seed"], sort=True):
                by_level = {r.level: run_cell(r) for r in g.itertuples()}
                label = cfg.upper() if cfg in ("a0", "a3") else f"{cfg.upper()} seed {seed}"
                lines.append(f"| {label} | " + " | ".join(by_level[v] for v in LEVELS) + " |")
            for cfg in ("a1", "a2", "a3"):
                row = cells[(cells.dataset == ds) & (cells.score == score) & (cells.config == cfg)].set_index("level")
                note = "" if cfg != "a3" else " (one run)"
                lines.append(f"| **{cfg.upper()} reading{note}** | " + " | ".join(
                    f"{CLASS[row.loc[v, 'class']]} ({int(row.loc[v, 'n_higher'])} higher, {int(row.loc[v, 'n_lower'])} lower of {int(row.loc[v, 'n_seeds'])})"
                    for v in LEVELS) + " |")
            lines.append("")
    lines += ["## Counts over all cells", "",
              "A cell is one dataset x score x level. A1, A2: higher or lower only when all three seeds agree. A3: a single paired interval.", "",
              "| Dataset | Score | Config. | Cells | Cells higher | Cells lower | Cells not consistent | Runs higher | Runs lower | Runs undetermined | Point estimates above A0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in counts.itertuples():
        lines.append(f"| {r.dataset} | {r.score} | {r.config.upper()} | {r.n_cells} | {r.cells_consistent_higher} | {r.cells_consistent_lower} | "
                     f"{r.cells_not_consistent} | {r.seed_runs_higher} | {r.seed_runs_lower} | {r.seed_runs_undetermined} | {r.seed_runs_point_higher}/{r.seed_runs} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out, len(lines), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
