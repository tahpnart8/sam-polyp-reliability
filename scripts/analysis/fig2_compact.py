"""Compact Fig. 2 of the paper: 2 rows (CVC-ColonDB, ETIS) x 3 panels (reliability R_roi,
reliability R_bnd, risk-coverage by conf_iou_pred), one shared legend, 122 mm x about 90 mm, 600 dpi.

Same data and the same drawing functions as figures.py; only the layout changes (no ECE text in
the panels, the values are in Table 4). A1 and A2 are seed 0, as in the earlier figures.

    python scripts/analysis/fig2_compact.py --metrics-dir results/metrics \
        --runs results/figures/runs_shown.csv --out-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figures import COLOR, LABEL, MM, NAME, WIDTH, load, plt, reliability  # noqa: E402

import pandas as pd  # noqa: E402

from samrel.config import load_config  # noqa: E402
from samrel.constants import REGIONS  # noqa: E402
from samrel.metrics import bin_edges, risk_coverage_curve  # noqa: E402

HEIGHT = 90 * MM
DATASETS = ("colondb", "etis")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--name", default="fig2_compact")
    args = ap.parse_args(argv)
    cfg = load_config("base")["metrics"]
    lo, hi = cfg["ece_range"]
    edges = bin_edges(cfg["ece_bins"], lo, hi)
    runs = pd.read_csv(args.runs, dtype=str)
    data = {r.run_id: load(args.metrics_dir, r.run_id) for r in runs.itertuples()}

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, HEIGHT))
    for i, ds in enumerate(DATASETS):
        for j, region in enumerate(("roi", "bnd")):
            ax = axes[i, j]
            ax.plot([lo, hi], [lo, hi], ls="--", lw=0.6, color="0.6")
            for r in runs.itertuples():
                df, stats = data[r.run_id]
                m = (df["dataset"] == ds).to_numpy()
                reliability(ax, stats[m, REGIONS.index(region)], edges, COLOR[r.config], LABEL[r.config])
            ax.set(xlim=(lo, hi), ylim=(lo, hi))
            ax.set_xlabel("Confidence", labelpad=1)
            ax.set_ylabel("Pixel accuracy", labelpad=1)
        ax = axes[i, 2]
        for r in runs.itertuples():
            df, _ = data[r.run_id]
            m = (df["dataset"] == ds).to_numpy()
            cov, risk = risk_coverage_curve(df.loc[m, "conf_iou_pred"].to_numpy(), df.loc[m, "dice"].to_numpy(),
                                            df.loc[m, "image_id"].to_numpy())
            ax.plot(cov, risk, lw=0.8, color=COLOR[r.config])
        ax.set(xlim=(0, 1), ylim=(0, None))
        ax.set_xlabel("Coverage", labelpad=1)
        ax.set_ylabel("Risk (1 - Dice)", labelpad=1)
        axes[i, 0].annotate(NAME[ds], xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=7, fontweight="bold")
    for j, title in enumerate(("(a) Reliability, $R_{roi}$", "(b) Reliability, $R_{bnd}$", "(c) Risk-coverage")):
        axes[0, j].set_title(title, pad=3)
    for ax in axes.flat:
        ax.tick_params(length=2, pad=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6, rect=(0.02, 0.05, 1, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.name}.{ext}", facecolor="white")
    print(args.out_dir / f"{args.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
