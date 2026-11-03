"""Integrity checks for the prompt-perturbation runs (PROTOCOL §14), independent of the analysis code.

  1. index: 8 configs/seeds x 4 levels, one commit, prompt sha256 of each level = file in data/prompts
  2. per-run: 576 images (380 ColonDB + 196 ETIS), no NaN outside conf_pairwise_dice of non-A3 runs,
     n_boxes and region pixel counts identical to the original runs (same ground truth, same ROI)
  3. pp00 reproduces the original runs (p0_check files and a direct per-image comparison here)
  4. pp00 summary values (perturb/summary.csv) equal the original summary.csv for ColonDB/ETIS
  5. A3 rows = mean of the three A2 members' predicted IoU at every level
  6. perturbed prompt files: tight_boxes equal the oracle file, sigma matches the level

    python scripts/checks/integrity_perturb.py --report results/logs/integrity_perturb.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "results" / "metrics"
P = M / "perturb"
TOL = 1e-4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)
    res: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    runs = pd.concat([pd.read_csv(p, dtype={"seed": str}) for p in sorted(P.glob("runs_*.csv"))], ignore_index=True)
    orig = pd.read_csv(M / "runs.csv", dtype={"seed": str})
    check("index: 32 runs", len(runs) == 32, f"{len(runs)}")
    check("index: one commit", runs.commit.nunique() == 1, ",".join(runs.commit.unique()))
    check("index: 8 runs per level", (runs.groupby("level").size() == 8).all(), runs.groupby("level").size().to_dict())
    files = {"pp00": ROOT / "data/prompts/prompts.json", **{f"pp{k}": ROOT / f"data/prompts/prompts_pp{k}.json" for k in ("10", "20", "30")}}
    shas = {lv: hashlib.sha256(f.read_bytes()).hexdigest() for lv, f in files.items()}
    check("index: prompt sha256 = files", all(runs.prompts_sha256 == runs.level.map(shas)))

    oracle = json.loads(files["pp00"].read_text())
    for lv in ("pp10", "pp20", "pp30"):
        d = json.loads(files[lv].read_text())
        same = all(e["tight_boxes"] == oracle["images"][i]["tight_boxes"] and e["oracle_boxes"] == oracle["images"][i]["boxes"]
                   for i, e in d["images"].items())
        check(f"prompts {lv}: tight/oracle boxes unchanged, sigma", same and abs(d["perturb"]["sigma"] - int(lv[2:]) / 100) < 1e-12)

    ref_cols = ["image_id", "n_boxes", "gt_area"]
    for r in runs.itertuples():
        df = pd.read_csv(P / "per_image" / f"{r.run_id}.csv")
        n_ok = len(df) == 576 and (df.dataset == "colondb").sum() == 380 and (df.dataset == "etis").sum() == 196
        cols = [c for c in df.columns if not (c == "conf_pairwise_dice" and r.config != "a3")]
        nan_ok = not df[cols].drop(columns=["excluded"]).isna().any().any()
        src = orig[(orig.config == r.config) & (orig.seed == r.seed)].run_id.iloc[0]
        o = pd.read_csv(M / "per_image" / f"{src}.csv")
        o = o[o.dataset.isin(["colondb", "etis"])].sort_values("image_id").reset_index(drop=True)
        d = df.sort_values("image_id").reset_index(drop=True)
        gt_ok = d[ref_cols].equals(o[ref_cols])
        with np.load(P / "ece_bins" / f"{r.run_id}.npz") as z, np.load(M / "ece_bins" / f"{src}.npz") as zo:
            pos = {i: k for k, i in enumerate(zo["image_ids"].tolist())}
            co = zo["stats"][[pos[i] for i in z["image_ids"]]][..., 0].sum(-1)
            px_ok = np.array_equal(z["stats"][..., 0].sum(-1), co)
        check(f"{r.run_id}: 576 images, no NaN, GT/ROI same as original", n_ok and nan_ok and gt_ok and px_ok,
              f"n={len(df)} nan_ok={nan_ok} gt={gt_ok} px={px_ok}")
        if r.level == "pp00":
            worst = max(float(np.abs(d[c] - o[c]).max()) for c in ("dice", "iou", "conf_iou_pred", "conf_mean_inside"))
            check(f"{r.run_id}: pp00 = original {src} (max |diff| <= {TOL})", worst <= TOL, f"{worst:.2e}")

    pc = pd.concat([pd.read_csv(p) for p in sorted(P.glob("p0_check_*.csv"))], ignore_index=True)
    check("p0_check files: 8 runs checked in the kernels", len(pc) == 8, f"{len(pc)}")

    a3 = runs[runs.config == "a3"]
    for r in a3.itertuples():
        df = pd.read_csv(P / "per_image" / f"{r.run_id}.csv").set_index("image_id")
        mem = runs[(runs.config == "a2") & (runs.level == r.level)]
        iou = np.mean([pd.read_csv(P / "per_image" / f"{m}.csv").set_index("image_id").loc[df.index, "conf_iou_pred"]
                       for m in mem.run_id], axis=0)
        check(f"{r.run_id}: predicted IoU = mean of 3 A2 members", np.allclose(df.conf_iou_pred, iou, atol=1e-6))

    ps = P / "summary.csv"
    if ps.exists():
        s = pd.read_csv(ps, dtype={"seed": str})
        s = s[s.level == "pp00"]
        so = pd.read_csv(M / "summary.csv", dtype={"seed": str})
        so = so[so.dataset.isin(["colondb", "etis"]) & so.seed.isin(["0", "1", "2", "012"])]
        mg = s.merge(so, on=["config", "seed", "dataset", "metric"], suffixes=("", "_o"))
        diff = float(np.nanmax(np.abs(mg[["value", "ci_low", "ci_high"]].to_numpy() - mg[["value_o", "ci_low_o", "ci_high_o"]].to_numpy())))
        check("pp00 summary (value, CI) = original summary.csv", len(mg) > 0 and diff <= TOL, f"{len(mg)} rows, max |diff| {diff:.2e}")

    n_fail = sum(not ok for _, ok, _ in res)
    lines = ["# Integrity checks: prompt perturbation (PROTOCOL §14)", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in res]
    lines += ["", f"**{len(res) - n_fail} PASS, {n_fail} FAIL**"]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-1:]))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
