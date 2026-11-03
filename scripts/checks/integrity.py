"""Integrity checks over finished runs (leakage, equal conditions, training, data, metrics, A3).

Independent of samrel.metrics where possible: AUROC by Mann-Whitney pair counting, AURC by an
explicit loop, Spearman by ranking + Pearson. Writes a Markdown report; exits 1 on any FAIL.

Usage:
    python scripts/checks/integrity.py --runs a0_s0_x,a1_s0_x,... --report results/logs/integrity_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samrel.aggregate import metric_functions, prepare
from samrel.config import load_config
from samrel.constants import DATASETS, TEST_DATASETS, TRAIN_DATASETS
from samrel.paths import REPO_ROOT

R = REPO_ROOT / "results"
EXPECTED_TEST = {ds: DATASETS[ds][1] for ds in TEST_DATASETS}
NAN_OK = {"conf_pairwise_dice"}  # NaN except for A3 (PROTOCOL §7.1)


class Report:
    def __init__(self):
        self.lines, self.fails = [], 0

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        self.fails += not ok
        self.lines.append(f"- {'PASS' if ok else '**FAIL**'}: {what}" + (f" ({detail})" if detail else ""))

    def note(self, text: str) -> None:
        self.lines.append(f"- NOTE: {text}")

    def head(self, text: str) -> None:
        self.lines.append(f"\n## {text}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auroc_mw(conf, failed) -> float:
    """P(score of a failure > score of a success) with score = -conf, ties 1/2."""
    pos, neg = -conf[failed], -conf[~failed]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def aurc_loop(conf, dice, ids) -> float:
    order = sorted(range(len(conf)), key=lambda i: (-conf[i], ids[i]))
    total, risks = 0.0, []
    for k, i in enumerate(order, start=1):
        total += 1 - dice[i]
        risks.append(total / k)
    return float(np.mean(risks))


def spearman_rank(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--final-commit", help="commit every final run must share")
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = [r for r in args.runs.split(",") if r]
    cfg = load_config("base")
    rep = Report()

    rep.head("1. Data leakage")
    split = json.loads((REPO_ROOT / cfg["data"]["val_split_file"]).read_text(encoding="utf-8"))
    tr = {i for ids in split["train"].values() for i in ids}
    va = {i for ids in split["val"].values() for i in ids}
    rep.check(not (tr & va), "train and val are disjoint", f"{len(tr)} + {len(va)}")
    rep.check(all(i.split("/")[0] in TRAIN_DATASETS for i in tr | va), "train/val contain only images of the two training sets")
    rep.check(len(tr | va) == 1450, "train + val = the 1,450 PraNet training images")
    excluded = json.loads((REPO_ROOT / cfg["data"]["excluded_file"]).read_text(encoding="utf-8"))["excluded"]
    rep.note(f"excluded_images.json lists {len(excluded)} images")

    rep.head("2. Same conditions")
    prompts_sha = sha(REPO_ROOT / cfg["data"]["prompts_file"])
    split_sha = sha(REPO_ROOT / cfg["data"]["val_split_file"])
    lock = json.loads((REPO_ROOT / "configs/env_kaggle.lock.json").read_text())["versions"]
    commits = {}
    for r in runs:
        mf = R / "logs" / "manifests" / f"{r}.json"
        if not mf.exists():
            rep.check(False, f"{r}: manifest present")
            continue
        m = json.loads(mf.read_text())
        commits[r] = m["commit"]
        rep.check(m["status"] == "ok", f"{r}: manifest status ok", m["status"])
        if m.get("prompts_sha256"):
            rep.check(m["prompts_sha256"] == prompts_sha, f"{r}: prompts.json sha256 matches the repository file")
            rep.check(m["val_split_sha256"] == split_sha, f"{r}: val_split.json sha256 matches")
            rep.check(m["checkpoint_sha256"] == cfg["model"]["checkpoint_sha256"], f"{r}: SAM checkpoint sha256 matches")
        else:
            rep.note(f"{r}: manifest has no sha fields (run made before commit 22ff673)")
        env = json.loads((R / "logs" / "environment" / f"{r}.json").read_text())
        diff = {k: (v, env.get(k)) for k, v in lock.items() if env.get(k) != v}
        rep.check(not diff, f"{r}: environment matches the env lock", str(diff) if diff else "")
    if args.final_commit:
        bad = {r: c[:7] for r, c in commits.items() if not c.startswith(args.final_commit)}
        rep.check(not bad, f"all runs at commit {args.final_commit}", str(bad) if bad else "")
    else:
        rep.note("commit theo run: " + ", ".join(f"{r}={c[:7]}" for r, c in commits.items()))

    rep.head("3. Training of A1, A2")
    for r in runs:
        cfg_name = r.split("_")[0]
        if cfg_name not in ("a1", "a2"):
            continue
        log = pd.read_csv(R / "logs" / "train" / f"{r}.csv")
        rep.check(list(log["epoch"]) == list(range(12)), f"{r}: all 12 epochs")
        best = json.loads((R / "checkpoints" / r / "best.json").read_text())
        exp = int(log["val_dice"].idxmax())  # idxmax returns the first max: ties keep the earlier epoch
        rep.check(best["epoch"] == int(log.loc[exp, "epoch"]), f"{r}: selected epoch = highest validation Dice (earliest on ties)",
                  f"epoch {best['epoch']}, val {best['val_dice']:.4f}")
        rep.check(log["train_loss"].iloc[-1] < log["train_loss"].iloc[0], f"{r}: training loss decreases",
                  f"{log['train_loss'].iloc[0]:.3f} → {log['train_loss'].iloc[-1]:.3f}")
        state = torch.load(R / "checkpoints" / r / "best_trainable.pt", map_location="cpu", weights_only=True)
        n_lora = sum(v.numel() for k, v in state.items() if ".lora_" in k)
        n_dec = sum(v.numel() for k, v in state.items() if k.startswith("mask_decoder."))
        other = [k for k in state if ".lora_" not in k and not k.startswith("mask_decoder.")]
        params = pd.read_csv(R / "metrics" / "trainable_params.csv").set_index("config")
        rep.check(not other, f"{r}: checkpoint holds only LoRA and mask decoder", str(other[:3]) if other else "")
        rep.check(n_dec == params.loc[cfg_name, "n_trainable_decoder"], f"{r}: decoder parameter count = trainable_params.csv", str(n_dec))
        rep.check(n_lora == params.loc[cfg_name, "n_trainable_lora"], f"{r}: LoRA parameter count = trainable_params.csv", str(n_lora))
        if cfg_name == "a2":
            moved = max(float(v.abs().max()) for k, v in state.items() if ".lora_b." in k)
            rep.check(moved > 0, f"{r}: LoRA B is non-zero after training (LoRA was learned)", f"max |B| {moved:.2e}")

    rep.head("4-6. Per-run data, metrics checked independently")
    ref_ids = None
    for r in runs:
        df = pd.read_csv(R / "metrics" / "per_image" / f"{r}.csv")
        counts = df["dataset"].value_counts().to_dict()
        rep.check(len(df) == 798 and counts == EXPECTED_TEST, f"{r}: 798 images, correct count per source")
        ids = sorted(df["image_id"])
        rep.check(len(set(ids)) == len(ids), f"{r}: image_id unique")
        if ref_ids is None:
            ref_ids = ids
        rep.check(ids == ref_ids, f"{r}: same image_id set as the first run")
        nan_cols = [c for c in df.columns if df[c].isna().any() and c not in NAN_OK]
        rep.check(not nan_cols, f"{r}: no NaN outside the allowed column", str(nan_cols) if nan_cols else "")
        if r.startswith("a3"):
            rep.check(df["conf_pairwise_dice"].notna().all(), f"{r}: A3 has pairwise Dice for every image")
        empty = df["pred_area"] == 0
        rep.check(bool((df.loc[empty, "dice"] == 0).all() and (df.loc[empty, "conf_mean_inside"] == 0).all()),
                  f"{r}: empty mask -> Dice 0 and conf_mean_inside 0", f"{int(empty.sum())} empty")
        rep.check(bool((df["failed"] == (df["dice"] < 0.5)).all()), f"{r}: column failed = (Dice < 0.5)")
        with np.load(R / "metrics" / "ece_bins" / f"{r}.npz") as z:
            bins = {k: z[k] for k in z.files}
        d2, stats = prepare(df, bins)
        worst = 0.0
        for ds in TEST_DATASETS:
            m = (d2["dataset"] == ds).to_numpy()
            g = d2[m].reset_index(drop=True)
            fns = metric_functions(g, stats[m])
            full = np.arange(len(g))
            c, dice, fail, gid = (g["conf_iou_pred"].to_numpy(), g["dice"].to_numpy(),
                                  g["failed"].to_numpy(bool), g["image_id"].tolist())
            pairs = [(fns["aurc_iou_pred"](full), aurc_loop(c, dice, gid)),
                     (fns["spearman_iou_pred"](full), spearman_rank(c, dice))]
            a1, a2 = fns["auroc_iou_pred"](full), auroc_mw(c, fail)
            if not (np.isnan(a1) and np.isnan(a2)):
                pairs.append((a1, a2))
            worst = max([worst] + [abs(x - y) for x, y in pairs])
            counts_bins = stats[m][:, 0, :, 0].sum(axis=1)
            px = (g["img_h"] * g["img_w"]).to_numpy()
            rep.check(bool(np.all(counts_bins == px)), f"{r}/{ds}: whole-image ECE bin pixel count = H x W of each image")
        rep.check(worst < 1e-9, f"{r}: AURC, Spearman, AUROC match an independent implementation", f"max deviation {worst:.1e}")

    rep.head("A3")
    for r in runs:
        if not r.startswith("a3"):
            continue
        members = json.loads((R / "logs" / "manifests" / f"{r}_members.json").read_text())["members"]
        seeds = sorted(m.split("_")[1] for m in members)
        mc = {commits.get(m) or json.loads((R / "logs/manifests" / f"{m}.json").read_text())["commit"] for m in members}
        rep.check(seeds == ["s0", "s1", "s2"] and all(m.startswith("a2_") for m in members), f"{r}: 3 members are A2 seeds 0/1/2")
        rep.check(len(mc) == 1, f"{r}: 3 members at one commit", str(mc))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = "# Integrity report\n\nRuns: " + ", ".join(runs) + "\n" + "\n".join(rep.lines) + \
           f"\n\n**Result: {rep.fails} FAIL.**\n"
    args.report.write_text(text, encoding="utf-8")
    print(f"{rep.fails} FAIL; report {args.report}")  # report is UTF-8; the Windows console is not
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
