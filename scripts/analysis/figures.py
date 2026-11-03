"""Reliability diagrams and risk-coverage curves from per-run files (paper Sect. 4).

Figures are 122 mm wide (LNCS text width), 600 dpi, fonts >= 6 pt. They read
only results/metrics/{per_image,ece_bins}; nothing is recomputed from predictions.

Usage:
    python scripts/analysis/figures.py --metrics-dir results/metrics --runs results/metrics/runs.csv \
        --out-dir results/figures --datasets colondb,etis --region roi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from samrel.config import load_config  # noqa: E402
from samrel.constants import REGIONS  # noqa: E402
from samrel.metrics import bin_edges, ece_from_bins, risk_coverage_curve  # noqa: E402

MM = 1 / 25.4
WIDTH = 122 * MM
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "font.family": "serif", "savefig.dpi": 600})
LABEL = {"a0": "A0 zero-shot", "a1": "A1 decoder", "a2": "A2 LoRA", "a3": "A3 ensemble"}
NAME = {"kvasir": "Kvasir", "clinicdb": "CVC-ClinicDB", "colondb": "CVC-ColonDB", "etis": "ETIS-LaribPolypDB",
        "cvc300": "CVC-300"}
COLOR = {"a0": "#4d4d4d", "a1": "#1b9e77", "a2": "#d95f02", "a3": "#7570b3"}


def load(metrics_dir: Path, run_id: str):
    df = pd.read_csv(metrics_dir / "per_image" / f"{run_id}.csv")
    with np.load(metrics_dir / "ece_bins" / f"{run_id}.npz") as z:
        bins = {k: z[k] for k in z.files}
    keep = ~df["excluded"].astype(bool).to_numpy()
    pos = {i: k for k, i in enumerate(bins["image_ids"].tolist())}
    return df[keep].reset_index(drop=True), bins["stats"][[pos[i] for i in df.loc[keep, "image_id"]]]


def reliability(ax, stats: np.ndarray, edges: np.ndarray, color: str, label: str) -> float:
    s = stats.sum(axis=0)
    nz = s[:, 0] > 0
    conf, acc = s[nz, 1] / s[nz, 0], s[nz, 2] / s[nz, 0]
    ax.plot(conf, acc, "o-", ms=2, lw=0.8, color=color, label=label)
    return ece_from_bins(stats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path, help="CSV: run_id, config, seed (one run per config shown)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--datasets", default="colondb,etis")
    ap.add_argument("--region", default="roi", choices=REGIONS)
    args = ap.parse_args(argv)
    cfg = load_config("base")["metrics"]
    lo, hi = cfg["ece_range"]
    edges = bin_edges(cfg["ece_bins"], lo, hi)
    runs = pd.read_csv(args.runs, dtype=str)
    datasets = args.datasets.split(",")
    k = REGIONS.index(args.region)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=(WIDTH, WIDTH / len(datasets) * 0.95), squeeze=False)
    fig2, axes2 = plt.subplots(1, len(datasets), figsize=(WIDTH, WIDTH / len(datasets) * 0.8), squeeze=False)
    for j, ds in enumerate(datasets):
        ax, ax2 = axes[0, j], axes2[0, j]
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.6, color="0.6")
        for r in runs.itertuples():
            df, stats = load(args.metrics_dir, r.run_id)
            m = (df["dataset"] == ds).to_numpy()
            e = reliability(ax, stats[m, k], edges, COLOR[r.config], f"{LABEL[r.config]}")
            ax.text(0.98, 0.03 + 0.07 * list(runs.config).index(r.config), f"ECE {e:.3f}",
                    transform=ax.transAxes, ha="right", color=COLOR[r.config], fontsize=6)
            cov, risk = risk_coverage_curve(df.loc[m, "conf_iou_pred"].to_numpy(), df.loc[m, "dice"].to_numpy(),
                                            df.loc[m, "image_id"].to_numpy())
            ax2.plot(cov, risk, lw=0.8, color=COLOR[r.config], label=LABEL[r.config])
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Confidence", title=NAME[ds])
        ax2.set(xlim=(0, 1), ylim=(0, None), xlabel="Coverage", title=NAME[ds])
        if j == 0:
            ax.set_ylabel("Pixel accuracy")
            ax2.set_ylabel("Risk (1 - Dice)")
    axes[0, 0].legend(loc="upper left", frameon=False)
    axes2[0, 0].legend(loc="upper left", frameon=False)
    for f, name in ((fig, f"reliability_{args.region}"), (fig2, "risk_coverage_iou_pred")):
        f.tight_layout(pad=0.4)
        f.savefig(args.out_dir / f"{name}.png")
        f.savefig(args.out_dir / f"{name}.pdf")
        print(args.out_dir / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
