"""Success/failure example figure (PROTOCOL §13): rule-based selection, then drawing.

Step 1 writes the selected images (by rule, not by eye; the overconfident rule was corrected after results):
    python scripts/analysis/example_figure.py select --runs results/metrics/runs.csv \
        --out results/figures/examples_selected.csv
Step 2 needs the .npz of those images only, downloaded per run into <pred-root>/<run_id>/<dataset>/<stem>.npz:
    python scripts/analysis/example_figure.py draw --runs results/metrics/runs.csv \
        --selected results/figures/examples_selected.csv --pred-root <dir> --data-root data/raw \
        --out-dir results/figures
A3 maps are the pixel-wise mean of the three A2 members (samrel.ensemble.combine_probs), as in the
A3 job; the actual and predicted IoU printed in each cell come from the per-image tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from samrel.config import load_config  # noqa: E402
from samrel.data import discover, read_image, read_mask  # noqa: E402
from samrel.ensemble import combine_probs  # noqa: E402
from samrel.examples import select_examples  # noqa: E402
from samrel.infer import load_prediction  # noqa: E402
from samrel.paths import REPO_ROOT  # noqa: E402

MM = 1 / 25.4
WIDTH = 122 * MM
plt.rcParams.update({"font.size": 6, "font.family": "serif", "savefig.dpi": 600})
COLUMNS = ("a0", "a1", "a2", "a3")
HEAD = {"a0": "A0 zero-shot", "a1": "A1 decoder", "a2": "A2 LoRA", "a3": "A3 ensemble"}
MAIN = [("colondb", "typical"), ("colondb", "overconfident"), ("etis", "typical"), ("etis", "overconfident")]
GT_COLOR, PRED_COLOR, BOX_COLOR = "#00ff00", "#ff00ff", "#ffff00"


def shown_runs(runs: pd.DataFrame) -> dict[str, list[str]]:
    """Seed 0 for A0-A2 (fixed in advance, §13); A3 needs the three A2 members for its maps."""
    out = {c: runs.loc[(runs.config == c) & (runs.seed == "0"), "run_id"].tolist() for c in ("a0", "a1", "a2")}
    out["a3"] = runs.loc[runs.config == "a3", "run_id"].tolist()
    out["a2_members"] = sorted(runs.loc[runs.config == "a2", "run_id"])
    for k, v in out.items():
        if not v or (k != "a2_members" and len(v) != 1) or (k == "a2_members" and len(v) != 3):
            raise SystemExit(f"runs.csv: unexpected runs for {k}: {v}")
    return out


def per_image(run_id: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "metrics" / "per_image" / f"{run_id}.csv")


def cmd_select(args) -> int:
    runs = pd.read_csv(args.runs, dtype=str)
    r = shown_runs(runs)
    sel = select_examples(per_image(r["a0"][0]), per_image(r["a2"][0]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args.out, index=False)
    print(sel.to_string(index=False))
    # file pattern for `kaggle kernels output --file-pattern` (only the selected images)
    stems = "|".join(sorted({i.split("/", 1)[1] for i in sel["image_id"]}))
    print(f"pattern: .*/predictions/.*/({'|'.join(sorted(set(sel['dataset'])))})/({stems})\\.npz$")
    return 0


def prob_map(pred_root: Path, run_id: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    ds, stem = image_id.split("/", 1)
    p = load_prediction(pred_root / run_id / ds / f"{stem}.npz")
    return p["prob"].astype(np.float32), p["boxes"]


def outline(ax, mask: np.ndarray, color: str) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.5)


def cmd_draw(args) -> int:
    runs = pd.read_csv(args.runs, dtype=str)
    r = shown_runs(runs)
    sel = pd.read_csv(args.selected)
    cfg = load_config("base")["data"]
    records = {rec.image_id: rec for rec in discover(args.data_root, cfg)}
    tables = {c: per_image(r[c][0]).set_index("image_id") for c in COLUMNS}

    def draw(rows, name):
        # one row gets a fixed page budget (under 620 px at 600 dpi) with explicit margins, see below
        height = 26.2 * MM if len(rows) == 1 else WIDTH / 5 * len(rows) * 0.95
        fig, axes = plt.subplots(len(rows), 1 + len(COLUMNS), figsize=(WIDTH, height), squeeze=False)
        for i, (ds, rule) in enumerate(rows):
            image_id = sel.loc[(sel.dataset == ds) & (sel.rule == rule), "image_id"].item()
            rec = records[image_id]
            img, gt = read_image(rec.image_path), read_mask(rec.mask_path, cfg["mask_threshold"])
            ax = axes[i, 0]
            ax.imshow(img)
            outline(ax, gt, GT_COLOR)
            _, boxes = prob_map(args.pred_root, r["a0"][0], image_id)
            for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4):
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BOX_COLOR, lw=0.4))
            ax.set_ylabel(f"{ds}\n{rule}", fontsize=6)
            for j, c in enumerate(COLUMNS, start=1):
                if c == "a3":
                    prob = combine_probs([prob_map(args.pred_root, m, image_id)[0] for m in r["a2_members"]])
                else:
                    prob = prob_map(args.pred_root, r[c][0], image_id)[0]
                row = tables[c].loc[image_id]
                ax = axes[i, j]
                ax.imshow(img)
                outline(ax, gt, GT_COLOR)
                outline(ax, prob >= 0.5, PRED_COLOR)
                ax.set_xlabel(f"IoU {row['iou']:.2f}  IoU$_p$ {row['conf_iou_pred']:.2f}", fontsize=6, labelpad=1)
        for j, title in enumerate(["Image, GT, box"] + [HEAD[c] for c in COLUMNS]):
            axes[0, j].set_title(title, fontsize=6, pad=2 if len(rows) == 1 else 6)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
        if len(rows) == 1:  # tight_layout leaves slack around equal-aspect images and clips the titles
            fig.subplots_adjust(top=1 - 3.1 * MM / height, bottom=2.6 * MM / height)
        for ext in ("png", "pdf"):
            fig.savefig(args.out_dir / f"{name}.{ext}")
        plt.close(fig)
        png = args.out_dir / f"{name}.png"
        Image.open(png).convert("RGB").save(png, dpi=(600, 600))  # white background, no alpha channel for Word
        print(args.out_dir / f"{name}.png")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw(MAIN, "examples_main")
    draw([("colondb", "overconfident")], "fig3_one_row")
    draw([(ds, rule) for ds in ("colondb", "etis") for rule in ("typical", "overconfident", "a0_worst", "a2_worst")],
         "examples_supplementary")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--runs", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    d = sub.add_parser("draw")
    d.add_argument("--runs", required=True, type=Path)
    d.add_argument("--selected", required=True, type=Path)
    d.add_argument("--pred-root", required=True, type=Path)
    d.add_argument("--data-root", required=True, type=Path)
    d.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    return cmd_select(args) if args.cmd == "select" else cmd_draw(args)


if __name__ == "__main__":
    raise SystemExit(main())
