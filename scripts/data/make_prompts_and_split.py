"""Generate prompts.json and val_split.json once (PROTOCOL §3, §4).

Usage:
    python scripts/data/make_prompts_and_split.py --data-root <PraNet dir> \
        --prompts-out data/prompts/prompts.json --split-out data/splits/val_split.json

Refuses to overwrite an existing file unless --force: these files are shared by every
configuration and every Kaggle job, so regenerating them silently would break the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from samrel.config import load_config
from samrel.constants import DATASETS, TRAIN_DATASETS
from samrel.data import data_hash, discover
from samrel.prompts import build_prompts
from samrel.splits import make_val_split


def _write(obj, path: Path, force: bool) -> str:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force only on purpose")
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    path.write_text(blob, encoding="utf-8", newline="\n")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--prompts-out", required=True, type=Path)
    ap.add_argument("--split-out", required=True, type=Path)
    ap.add_argument("--config", default="base")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    records = discover(args.data_root, cfg["data"])
    counts = {ds: sum(r.dataset == ds and r.image_path is not None for r in records) for ds in DATASETS}
    wrong = {ds: (n, DATASETS[ds][1]) for ds, n in counts.items() if n != DATASETS[ds][1]}
    if wrong:
        print(f"WARNING: counts differ from PROTOCOL §1.1 (found, expected): {wrong}")
    dhash = data_hash(records)

    d = cfg["data"]
    prompts = build_prompts(records, cfg["root_seed"], cfg["prompt"]["jitter_max"],
                            d["mask_threshold"], d["connectivity"], dhash,
                            cfg["prompt"]["min_box_component_px"])
    ids = {ds: [r.image_id for r in records if r.dataset == ds and r.mask_path and r.image_path]
           for ds in TRAIN_DATASETS}
    split = make_val_split(ids, cfg["split"]["val_fraction"], cfg["root_seed"])
    split["data_hash"] = dhash

    p_sha = _write(prompts, args.prompts_out, args.force)
    s_sha = _write(split, args.split_out, args.force)
    n_boxes = sum(len(v["boxes"]) for v in prompts["images"].values())
    print(f"prompts: {len(prompts['images'])} images, {n_boxes} boxes, sha256 {p_sha}")
    print("val: " + ", ".join(f"{k} {len(v)}" for k, v in split["val"].items()) + f", sha256 {s_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
