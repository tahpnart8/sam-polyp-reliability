"""Tasks that need finished runs: A3 ensemble metrics, and the shared inference-cost benchmark.

  ensemble   Average the saved probabilities of the three A2 seeds (PROTOCOL §5.4), then the
             usual per-image metrics. CPU only; reads predictions/<run_id>/ of each member.
  benchmark  Time A0, A1, A2 and A3 on the same GPU and the same first N images of the
             benchmark dataset (PROTOCOL §10). A3 = three full A2 passes plus the average.

Earlier outputs are found by run_id under --search-root (Kaggle mounts kernel outputs there).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samrel.config import load_config
from samrel.constants import COST_COLUMNS, TEST_DATASETS
from samrel.data import discover, read_image
from samrel.ensemble import combine_probs
from samrel.evaluate import evaluate_run, write_run_metrics
from samrel.infer import make_predictor, predict_image
from samrel.models import build_model
from samrel.train import load_trainable


def find_dir(root: Path, *parts: str) -> Path:
    hits = sorted(p for p in root.rglob(parts[-1]) if p.is_dir() and p.parts[-len(parts):] == parts)
    if len(hits) != 1:
        raise SystemExit(f"STOP: expected one {'/'.join(parts)} under {root}, found {hits}")
    return hits[0]


def find_trainable(root: Path, run_id: str) -> Path:
    return find_dir(root, "checkpoints", run_id) / "best_trainable.pt"


def timed_model(cfg_name: str, ckpt: Path, trainable: Path | None, device: str):
    cfg = load_config(cfg_name)
    m = build_model(cfg, ckpt)
    if trainable is not None:
        load_trainable(m, torch.load(trainable, map_location="cpu", weights_only=True))
    return make_predictor(m.to(device).eval())


def bench(predictors, images, boxes, warmup: int, device: str) -> tuple[list[float], float]:
    """Seconds per image (after ``warmup``) and peak memory; several predictors = A3."""
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for k, (img, bx) in enumerate(zip(images, boxes)):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = [predict_image(p, img, bx)[0] for p in predictors]
        if len(probs) > 1:
            combine_probs(probs)
        if cuda:
            torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
    return times, peak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=["ensemble", "benchmark"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--search-root", required=True, type=Path)
    ap.add_argument("--a2-runs", required=True, help="three A2 run_ids, comma separated")
    ap.add_argument("--a1-run", help="benchmark: one A1 run_id")
    ap.add_argument("--checkpoint", type=Path, help="benchmark: SAM checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--excluded", type=Path)
    args = ap.parse_args(argv)

    base = load_config("base")
    d = base["data"]
    prompts = json.loads((args.prompts or Path(d["prompts_file"])).read_text(encoding="utf-8"))
    excluded = json.loads((args.excluded or Path(d["excluded_file"])).read_text(encoding="utf-8"))["excluded"]
    records = discover(args.data_root, d)
    a2_runs = [r for r in args.a2_runs.split(",") if r]
    if len(a2_runs) != 3:
        raise SystemExit("A3 needs exactly three A2 runs")

    if args.task == "ensemble":
        cfg = load_config("a2")
        cfg["name"] = "a3"
        tests = [r for r in records if r.dataset in TEST_DATASETS]
        members = [find_dir(args.search_root, "predictions", r) for r in a2_runs]
        df, bins = evaluate_run(args.run_id, "a3", "012", members, tests, prompts, cfg, excluded)
        write_run_metrics(df, bins, args.run_id, out_dir=args.out / "metrics")
        (args.out / "logs" / "manifests").mkdir(parents=True, exist_ok=True)
        (args.out / "logs" / "manifests" / f"{args.run_id}_members.json").write_text(
            json.dumps({"members": a2_runs}), encoding="utf-8")
        return 0

    c = base["cost"]
    pool = sorted((r for r in records if r.dataset == c["benchmark_dataset"]), key=lambda r: r.image_id)
    pool = pool[: c["benchmark_warmup"] + c["benchmark_n_images"]]
    images = [read_image(r.image_path) for r in pool]
    boxes = [prompts["images"][r.image_id]["boxes"] for r in pool]
    plans = {
        "a0": [("a0", None)],
        "a1": [("a1", find_trainable(args.search_root, args.a1_run))],
        "a2": [("a2", find_trainable(args.search_root, a2_runs[0]))],
        "a3": [("a2", find_trainable(args.search_root, r)) for r in a2_runs],
    }
    name, count = (torch.cuda.get_device_name(0), torch.cuda.device_count()) if torch.cuda.is_available() else (None, 0)
    rows = []
    for config, members in plans.items():
        preds = [timed_model(cfg_name, args.checkpoint, tr, args.device) for cfg_name, tr in members]
        times, peak = bench(preds, images, boxes, c["benchmark_warmup"], args.device)
        rows.append({"run_id": args.run_id, "config": config, "seed": "012" if config == "a3" else "0",
                     "phase": "infer_benchmark", "gpu_name": name, "gpu_count": count, "amp": "none",
                     "batch_size": 1, "n_images": len(times), "wall_seconds": round(sum(times), 3),
                     "gpu_hours": round(sum(times) / 3600, 5), "sec_per_image_mean": round(statistics.mean(times), 4),
                     "sec_per_image_median": round(statistics.median(times), 4), "peak_gpu_mem_mb": round(peak, 1),
                     "commit": args.commit})
        del preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.out / "metrics" / "cost").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COST_COLUMNS)).to_csv(args.out / "metrics" / "cost" / f"{args.run_id}.csv", index=False)
    print(pd.DataFrame(rows)[["config", "sec_per_image_mean", "peak_gpu_mem_mb"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
