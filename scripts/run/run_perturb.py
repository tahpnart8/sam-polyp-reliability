"""Prompt-perturbation inference (PROTOCOL §14): existing models, several box sets per image.

Each model encodes every image once and decodes the oracle boxes (pp00) and every perturbed level,
then per-image metrics are computed per (model, level). A3 is the average of the three A2 members
at the same level (§5.4). Nothing is trained.

    python scripts/run/run_perturb.py --commit <c> --stamp 20261103-0900 --data-root <PraNet> \
        --checkpoint sam_vit_b_01ec64.pth --out <out> --search-root <dir with checkpoints/<run_id>/> \
        --models a0_s0_d60e848_20261029-1028,a1_s0_...  --levels data/prompts/prompts_pp10.json,...

--models takes the original run_ids (results/metrics/runs.csv); A1/A2 weights are found as
checkpoints/<run_id>/best_trainable.pt under --search-root. Run ids of the outputs follow
PROTOCOL §14.7: <config>-pp<NN>_s<seed>_<commit7>_<stamp>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samrel.config import load_config
from samrel.constants import PRIMARY_DATASETS
from samrel.data import discover, read_image
from samrel.evaluate import evaluate_run, write_run_metrics
from samrel.infer import decode_boxes, make_predictor, save_prediction
from samrel.models import build_model
from samrel.train import load_trainable

from run_post import find_trainable

# P0 re-run must reproduce the original per-image numbers (PROTOCOL §14.3)
REPRO_COLUMNS = ("dice", "iou", "conf_iou_pred")
REPRO_TOL = 1e-4


def out_run_id(config: str, level: str, seed: str, commit: str, stamp: str) -> str:
    return f"{config}-{level}_s{seed}_{commit[:7]}_{stamp}"


def parse_source(run_id: str) -> tuple[str, str]:
    config, seed = run_id.split("_")[:2]
    return config, seed.lstrip("s")


def load_levels(paths: list[Path], oracle: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Prompt payload and sha256 of the file for every level; pp00 (the oracle) comes first."""
    levels, shas = {}, {}
    for p in [oracle, *paths]:
        raw = p.read_bytes()
        d = json.loads(raw)
        name = "pp00" if p == oracle else f"pp{round(d['perturb']['sigma'] * 100):02d}"
        levels[name], shas[name] = d, hashlib.sha256(raw).hexdigest()
    return levels, shas


def predict_levels(model, records, levels: dict[str, dict], out: Path, run_ids: dict[str, str], device: str) -> float:
    """Encode each image once, decode every level; returns wall seconds."""
    model.to(device).eval()
    predictor = make_predictor(model)
    t0 = time.perf_counter()
    for r in records:
        image = read_image(r.image_path)
        predictor.set_image(image, image_format="RGB")
        stem = r.image_id.split("/", 1)[1]
        for level, prompts in levels.items():
            boxes = prompts["images"][r.image_id]["boxes"]
            prob, iou_pred = decode_boxes(predictor, image.shape[:2], boxes)
            save_prediction(out / "predictions" / run_ids[level] / r.dataset / f"{stem}.npz", prob, iou_pred, boxes)
    return time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--stamp", required=True, help="yyyymmdd-hhmm (UTC) shared by every output run_id")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--search-root", required=True, type=Path)
    ap.add_argument("--models", required=True, help="original run_ids, comma separated")
    ap.add_argument("--levels", required=True, help="perturbed prompt files, comma separated")
    ap.add_argument("--a3-members", default="", help="three A2 run_ids among --models to ensemble")
    ap.add_argument("--check-p0", action="store_true",
                    help="compare pp00 with per_image/<source run>.csv found under --search-root (§14.3)")
    ap.add_argument("--a3-source", default="", help="original A3 run_id, for the pp00 check of A3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oracle", type=Path, help="default: data.prompts_file of the config")
    ap.add_argument("--excluded", type=Path)
    ap.add_argument("--limit-images", type=int, default=0, help="local tests only")
    args = ap.parse_args(argv)

    base = load_config("base")
    d = base["data"]
    levels, shas = load_levels([Path(p) for p in args.levels.split(",") if p], args.oracle or Path(d["prompts_file"]))
    excluded = json.loads((args.excluded or Path(d["excluded_file"])).read_text(encoding="utf-8"))["excluded"]
    datasets = set(PRIMARY_DATASETS)
    for name, prompts in levels.items():
        if name != "pp00" and set(prompts["perturb"]["datasets"]) != datasets:
            raise SystemExit(f"STOP: {name} covers {prompts['perturb']['datasets']}, expected {sorted(datasets)}")
    records = sorted((r for r in discover(args.data_root, d) if r.dataset in datasets), key=lambda r: r.image_id)
    if args.limit_images:
        records = [r for ds in sorted(datasets) for r in [x for x in records if x.dataset == ds][:args.limit_images]]
    ckpt = None if str(args.checkpoint) == "random" else args.checkpoint  # "random": local tests only

    index, cost, made, checks = [], [], {}, []
    for source in [m for m in args.models.split(",") if m]:
        config, seed = parse_source(source)
        cfg = load_config(config)
        model = build_model(cfg, ckpt, seed=int(seed))
        if cfg["train"]["enabled"]:
            load_trainable(model, torch.load(find_trainable(args.search_root, source), map_location="cpu",
                                             weights_only=True))
        run_ids = {lv: out_run_id(config, lv, seed, args.commit, args.stamp) for lv in levels}
        made[source] = run_ids
        wall = predict_levels(model, records, levels, args.out, run_ids, args.device)
        cost.append({"source_run": source, "config": config, "seed": seed, "n_images": len(records),
                     "n_levels": len(levels), "wall_seconds": round(wall, 2)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for lv, rid in run_ids.items():
            df, bins = evaluate_run(rid, config, seed, [args.out / "predictions" / rid], records, levels[lv], cfg,
                                    excluded)
            write_run_metrics(df, bins, rid, out_dir=args.out / "metrics" / "perturb")
            sigma = 0.0 if lv == "pp00" else levels[lv]["perturb"]["sigma"]
            index.append({"run_id": rid, "config": config, "seed": seed, "level": lv, "sigma": sigma,
                          "source_run": source, "commit": args.commit, "prompts_sha256": shas[lv]})
            if lv == "pp00" and args.check_p0:
                checks.append(check_reproduction(df, find_csv(args.search_root, source), rid, source))

    members = [m for m in args.a3_members.split(",") if m]
    if members:
        if len(members) != 3 or any(m not in made for m in members):
            raise SystemExit("STOP: --a3-members must be three run_ids from --models")
        cfg = load_config("a2")
        cfg["name"] = "a3"
        for lv in levels:
            rid = out_run_id("a3", lv, "012", args.commit, args.stamp)
            dirs = [args.out / "predictions" / made[m][lv] for m in members]
            df, bins = evaluate_run(rid, "a3", "012", dirs, records, levels[lv], cfg, excluded)
            write_run_metrics(df, bins, rid, out_dir=args.out / "metrics" / "perturb")
            sigma = 0.0 if lv == "pp00" else levels[lv]["perturb"]["sigma"]
            index.append({"run_id": rid, "config": "a3", "seed": "012", "level": lv, "sigma": sigma,
                          "source_run": "+".join(members), "commit": args.commit, "prompts_sha256": shas[lv]})
            if lv == "pp00" and args.check_p0 and args.a3_source:
                checks.append(check_reproduction(df, find_csv(args.search_root, args.a3_source), rid, args.a3_source))

    pdir = args.out / "metrics" / "perturb"
    pd.DataFrame(index).to_csv(pdir / f"runs_{args.stamp}_{'-'.join(sorted({i['config'] for i in index}))}.csv",
                               index=False)
    pd.DataFrame(cost).to_csv(pdir / f"cost_{args.stamp}_{'-'.join(sorted({c['config'] for c in cost}))}.csv",
                              index=False)
    if checks:
        pd.DataFrame(checks).to_csv(pdir / f"p0_check_{args.stamp}_{'-'.join(sorted({c['config'] for c in cost}))}.csv",
                                    index=False)
    return 0


def find_csv(root: Path, run_id: str) -> Path:
    hits = sorted(p for p in root.rglob(f"{run_id}.csv") if p.parent.name == "per_image")
    if not hits:
        raise SystemExit(f"STOP: no per_image/{run_id}.csv under {root} for the pp00 check")
    return hits[0]


def check_reproduction(df: pd.DataFrame, ref_csv: Path, rid: str, source: str = "") -> dict:
    ref = pd.read_csv(ref_csv).set_index("image_id")
    new = df.set_index("image_id")
    worst = {c: float(np.max(np.abs(new[c].to_numpy() - ref.loc[new.index, c].to_numpy()))) for c in REPRO_COLUMNS}
    print(f"{rid}: P0 vs original, max |diff| {worst}", flush=True)
    if max(worst.values()) > REPRO_TOL:
        raise SystemExit(f"STOP: {rid} does not reproduce the original run (PROTOCOL §14.3): {worst}")
    return {"run_id": rid, "source_run": source, "n_images": len(new), **{f"max_abs_diff_{c}": v for c, v in worst.items()}}


if __name__ == "__main__":
    raise SystemExit(main())
