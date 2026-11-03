"""Rebuild every derived table and figure from one git commit and compare with results/.

Exports the analysis code of --commit with ``git archive`` into a scratch folder, copies only the
per-run inputs (per_image/, ece_bins/, cost/, runs.csv, trainable_params.csv), reruns
aggregate.py, finalize_tables.py, figures.py (roi, bnd, img) and example_figure.py select there,
then compares with the committed files: CSV by sha256 of the bytes (fallback: numeric equality),
PNG by decoded pixels. PDFs carry a creation date, so they are reported but not compared.
The example-figure drawing step needs downloaded predictions and is not rerun here.

    python scripts/checks/reproduce_outputs.py --commit HEAD --work <scratch dir> --report <md file>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ("per_image", "ece_bins", "cost", "runs.csv", "trainable_params.csv")
TABLES = ("summary.csv", "summary_secondary.csv", "seed_stability.csv", "seed_stability_secondary.csv",
          "paired.csv", "paired_secondary.csv", "paired_consistency.csv", "paired_consistency_secondary.csv",
          "cost.csv")
FIGURES = ("reliability_roi.png", "reliability_bnd.png", "reliability_img.png", "risk_coverage_iou_pred.png")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export(commit: str, dest: Path) -> str:
    full = subprocess.run(["git", "rev-parse", commit], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    blob = subprocess.run(["git", "archive", full, "src", "scripts", "configs"], cwd=ROOT, capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)
    return full


def run(code: Path, script: str, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(code / "src")}
    subprocess.run([sys.executable, str(code / "scripts" / "analysis" / script), *args], cwd=code, env=env,
                   check=True, capture_output=True)


def same_csv(a: Path, b: Path) -> str:
    if sha(a) == sha(b):
        return "identical bytes"
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return "DIFFERENT shape/columns"
    num = x.select_dtypes("number").columns
    other = [c for c in x.columns if c not in num]
    ok = x[other].astype(str).equals(y[other].astype(str)) and np.allclose(
        x[num].to_numpy(float), y[num].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True)
    return "equal values (bytes differ)" if ok else "DIFFERENT values"


def same_png(a: Path, b: Path) -> str:
    x, y = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    return "identical pixels" if x.shape == y.shape and np.array_equal(x, y) else "DIFFERENT pixels"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)

    if args.work.exists():
        shutil.rmtree(args.work)
    # example_figure.py reads <repo>/results/metrics, so the inputs live inside the exported tree
    code = args.work / "code"
    metrics, figs = code / "results" / "metrics", args.work / "figures"
    code.mkdir(parents=True)
    full = export(args.commit, code)
    src = ROOT / "results" / "metrics"
    for name in INPUTS:
        (shutil.copytree if (src / name).is_dir() else shutil.copy2)(src / name, metrics / name)
    runs = str(metrics / "runs.csv")
    run(code, "aggregate.py", "--metrics-dir", str(metrics), "--runs", runs)
    run(code, "finalize_tables.py", "--metrics-dir", str(metrics))
    shown = str(ROOT / "results" / "figures" / "runs_shown.csv")
    for region in ("roi", "bnd", "img"):
        run(code, "figures.py", "--metrics-dir", str(metrics), "--runs", shown, "--out-dir", str(figs),
            "--region", region)
    run(code, "example_figure.py", "select", "--runs", runs, "--out", str(figs / "examples_selected.csv"))

    lines = ["# Rebuilding tables and figures from one commit", "",
             f"- Analysis code commit: `{full}`", f"- Inputs: `results/metrics/{{{','.join(INPUTS)}}}` (repository copy)",
             "- Compared with the files committed in `results/`.", "", "| File | Result |", "|---|---|"]
    bad = 0
    for t in TABLES:
        r = same_csv(src / t, metrics / t)
        bad += r.startswith("DIFFERENT")
        lines.append(f"| `results/metrics/{t}` | {r} |")
    for f in FIGURES:
        r = same_png(ROOT / "results" / "figures" / f, figs / f)
        bad += r.startswith("DIFFERENT")
        lines.append(f"| `results/figures/{f}` | {r} |")
    r = same_csv(ROOT / "results" / "figures" / "examples_selected.csv", figs / "examples_selected.csv")
    bad += r.startswith("DIFFERENT")
    lines += [f"| `results/figures/examples_selected.csv` | {r} |", "",
              "PDFs are not compared (they embed a creation date). `example_figure.py draw` needs the prediction `.npz` files and is not rerun.",
              "", f"**{'0 differences' if not bad else f'{bad} files differ'}**"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
