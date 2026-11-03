"""Run one experiment task (called by kaggle/jobs/<run_id>/run.py; also runs locally on CPU).

Tasks:
  infer  A0, or A1/A2 with --trainable: predict every test image, then per-image metrics.
  train  A1/A2 with seed: (A1) cache embeddings, train, keep best epoch by validation Dice,
         then infer the test sets with the best weights.

Outputs under --out mirror results/: predictions/<run_id>/, metrics/per_image/, metrics/ece_bins/,
metrics/cost/<run_id>.csv, logs/train/<run_id>.csv, checkpoints/<run_id>/best_trainable.pt.
--limit-images and --max-steps exist only for the smoke run; their outputs are never analysed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from samrel.config import load_config
from samrel.constants import COST_COLUMNS, TEST_DATASETS
from samrel.data import discover
from samrel.evaluate import evaluate_run, write_run_metrics
from samrel.infer import run_inference
from samrel.models import build_model
from samrel.train import SampleLoader, cache_embeddings, load_trainable, train


def gpu_info() -> tuple[str | None, int]:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0), torch.cuda.device_count()
    return None, 0


def cost_row(run_id, cfg, seed, phase, n_images, wall, per_image, peak_mb, commit) -> dict:
    name, count = gpu_info()
    return {"run_id": run_id, "config": cfg["name"], "seed": seed, "phase": phase, "gpu_name": name,
            "gpu_count": count, "amp": cfg["train"]["amp"], "batch_size": cfg["train"]["batch_size"],
            "n_images": n_images, "wall_seconds": round(wall, 2),
            # The code uses one GPU even when Kaggle provides two, so GPU-hours = wall-clock hours.
            "gpu_hours": round(wall / 3600, 4) if count else 0.0,
            "sec_per_image_mean": round(statistics.mean(per_image), 4) if per_image else None,
            "sec_per_image_median": round(statistics.median(per_image), 4) if per_image else None,
            "peak_gpu_mem_mb": round(peak_mb, 1), "commit": commit}


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def infer_and_evaluate(model, cfg, seed, run_id, records, prompts, excluded, out: Path, device, commit, limit):
    tests = sorted((r for r in records if r.dataset in TEST_DATASETS), key=lambda r: r.image_id)
    if limit:
        tests = [r for ds in TEST_DATASETS for r in [x for x in tests if x.dataset == ds][:limit]]
    pred_dir = out / "predictions" / run_id
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    timing = run_inference(model, tests, prompts, pred_dir, device=device)
    row = cost_row(run_id, cfg, seed, "infer", len(tests), time.perf_counter() - t0,
                   [t["seconds"] for t in timing], peak_mb(), commit)
    df, bins = evaluate_run(run_id, cfg["name"], seed, [pred_dir], tests, prompts, cfg, excluded)
    write_run_metrics(df, bins, run_id, out_dir=out / "metrics")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=["infer", "train"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path, help="resumable training state")
    ap.add_argument("--trainable", type=Path, help="best_trainable.pt for infer of A1/A2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-images", type=int, default=0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--prompts", type=Path, help="default: data.prompts_file of the config")
    ap.add_argument("--split", type=Path, help="default: data.val_split_file")
    ap.add_argument("--excluded", type=Path, help="default: data.excluded_file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    d = cfg["data"]
    prompts = json.loads((args.prompts or Path(d["prompts_file"])).read_text(encoding="utf-8"))
    split = json.loads((args.split or Path(d["val_split_file"])).read_text(encoding="utf-8"))
    excluded = json.loads((args.excluded or Path(d["excluded_file"])).read_text(encoding="utf-8"))["excluded"]
    records = discover(args.data_root, d)
    rec_by_id = {r.image_id: r for r in records}
    # "random" builds untrained weights; only the local test uses it.
    ckpt = None if str(args.checkpoint) == "random" else args.checkpoint
    model = build_model(cfg, ckpt, seed=args.seed)
    rows = []

    if args.task == "train":
        if not cfg["train"]["enabled"]:
            raise SystemExit(f"config {cfg['name']} has no training")
        train_recs = {i: rec_by_id[i] for ids in (*split["train"].values(), *split["val"].values()) for i in ids}
        cache = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if cfg["train"].get("cache_embeddings"):
            loader = SampleLoader(model, train_recs, prompts, cfg)
            train_ids = [i for ids in split["train"].values() for i in ids]
            cache = cache_embeddings(model, train_ids, loader, device=args.device)
            rows.append(cost_row(args.run_id, cfg, args.seed, "embed_cache", len(train_ids),
                                 time.perf_counter() - t0, [], peak_mb(), args.commit))
            t0 = time.perf_counter()
        best = train(model, cfg, args.seed, train_recs, prompts, split, args.work, device=args.device,
                     embed_cache=cache, max_steps=args.max_steps)
        rows.append(cost_row(args.run_id, cfg, args.seed, "train", len(train_recs),
                             time.perf_counter() - t0, [], peak_mb(), args.commit))
        (args.out / "logs" / "train").mkdir(parents=True, exist_ok=True)
        shutil.copy(args.work / "train_log.csv", args.out / "logs" / "train" / f"{args.run_id}.csv")
        ck_dir = args.out / "checkpoints" / args.run_id
        ck_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.work / "best_trainable.pt", ck_dir / "best_trainable.pt")
        (ck_dir / "best.json").write_text(json.dumps(best), encoding="utf-8")
        load_trainable(model, torch.load(ck_dir / "best_trainable.pt", map_location="cpu", weights_only=True))
    elif args.trainable:
        load_trainable(model, torch.load(args.trainable, map_location="cpu", weights_only=True))
    elif cfg["train"]["enabled"]:
        raise SystemExit("A1/A2 inference needs --trainable")

    rows.append(infer_and_evaluate(model, cfg, args.seed, args.run_id, records, prompts, excluded,
                                   args.out, args.device, args.commit, args.limit_images))
    (args.out / "metrics" / "cost").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COST_COLUMNS)).to_csv(
        args.out / "metrics" / "cost" / f"{args.run_id}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
