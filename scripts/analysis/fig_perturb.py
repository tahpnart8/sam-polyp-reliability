"""Prompt-perturbation figure (PROTOCOL §14.7): one 1 x 4 strip, 122 mm x 44 mm:
Dice ColonDB, Dice ETIS, Spearman rho (conf_iou_pred) ColonDB, rho ETIS; x = P0..P3
(--layout grid gives the earlier 2 x 2). A1, A2: mean of 3 seeds with ±SD bars;
A0, A3: value with bootstrap 95% CI. Reads only results/metrics/perturb/summary.csv.

    python scripts/analysis/fig_perturb.py --perturb-dir results/metrics/perturb --out-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figures import COLOR, LABEL, MM, NAME, WIDTH, plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from samrel.perturb import LEVELS  # noqa: E402

ROWS = (("dice", "Dice"), ("spearman_iou_pred", r"Spearman $\rho$ (predicted IoU)"))
DATASETS = ("colondb", "etis")
XLABELS = ("P0\n(oracle)", "P1\n$\\sigma$=0.1", "P2\n$\\sigma$=0.2", "P3\n$\\sigma$=0.3")
SHORT = {"colondb": "ColonDB", "etis": "ETIS"}
OFFSET = {"a0": -0.09, "a1": -0.03, "a2": 0.03, "a3": 0.09}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perturb-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", default="fig_perturb")
    ap.add_argument("--layout", default="strip", choices=["strip", "grid"])
    args = ap.parse_args(argv)
    s = pd.read_csv(args.perturb_dir / "summary.csv", dtype={"seed": str})
    x = np.arange(len(LEVELS))
    strip = args.layout == "strip"
    if strip:
        fig, flat = plt.subplots(1, 4, figsize=(WIDTH, 44 * MM))
        axes = np.array(flat).reshape(2, 2)  # [metric, dataset], panels in reading order
    else:
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 82 * MM), sharex=True)
    for i, (metric, ylabel) in enumerate(ROWS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            g = s[(s.metric == metric) & (s.dataset == ds)]
            for c in ("a0", "a1", "a2", "a3"):
                gc = g[g.config == c].set_index(["level", "seed"])
                if c in ("a1", "a2"):
                    v = np.array([gc.xs(lv, level="level")["value"].mean() for lv in LEVELS])
                    sd = np.array([gc.xs(lv, level="level")["value"].std(ddof=1) for lv in LEVELS])
                    err = np.vstack([sd, sd])
                else:
                    row = [gc.xs(lv, level="level").iloc[0] for lv in LEVELS]
                    v = np.array([r["value"] for r in row])
                    err = np.vstack([v - [r["ci_low"] for r in row], [r["ci_high"] for r in row] - v])
                ax.errorbar(x + OFFSET[c], v, yerr=err, color=COLOR[c], marker="o", ms=2.5, lw=0.8, elinewidth=0.6,
                            capsize=1.5, label=LABEL[c])
            ax.tick_params(length=2, pad=1)
            if strip:
                ax.set_xticks(x, ["P0", "P1", "P2", "P3"])
                ax.set_title(("Dice, " if i == 0 else r"Spearman $\rho$, ") + SHORT[ds], pad=3)
            else:
                ax.set_xticks(x, XLABELS)
                if i == 0:
                    ax.set_title(NAME[ds], pad=3)
                if j == 0:
                    ax.set_ylabel(ylabel, labelpad=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5, rect=(0, 0.11 if strip else 0.06, 1, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.name}.{ext}", facecolor="white")
    print(args.out_dir / f"{args.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
