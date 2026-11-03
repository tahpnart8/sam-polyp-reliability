"""Generate the perturbed box prompts once (PROTOCOL §14.2) and describe each level (§14.7).

Usage:
    python scripts/data/make_perturbed_prompts.py --base data/prompts/prompts.json \
        --out-dir data/prompts --levels-csv results/metrics/perturb/prompt_levels.csv

Reads only prompts.json (no image, no model output). Refuses to overwrite an existing file unless
--force: every configuration must read the same perturbed boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.constants import PRIMARY_DATASETS, ROOT_SEED
from samrel.prompts import box_iou, build_perturbed_prompts

SIGMAS = (0.10, 0.20, 0.30)  # PROTOCOL §14.1, fixed before any perturbed inference


def level_name(sigma: float) -> str:
    return f"pp{round(sigma * 100):02d}"


def describe(prompts: dict, level: str) -> list[dict]:
    rows = []
    for ds in prompts["perturb"]["datasets"]:
        ious, clipped, degen = [], 0, 0
        for image_id, e in prompts["images"].items():
            if image_id.split("/", 1)[0] != ds:
                continue
            ious += [box_iou(a, b) for a, b in zip(e["boxes"], e["oracle_boxes"])]
            clipped += sum(e["clipped"])
            degen += sum(e["degenerate"])
        q1, med, q3 = np.percentile(ious, [25, 50, 75])
        rows.append({"level": level, "sigma": prompts["perturb"]["sigma"], "dataset": ds, "n_boxes": len(ious),
                     "box_iou_median": round(med, 4), "box_iou_q1": round(q1, 4), "box_iou_q3": round(q3, 4),
                     "box_iou_min": round(min(ious), 4), "n_clipped": clipped, "n_degenerate": degen})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--levels-csv", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    raw = args.base.read_bytes()
    base = json.loads(raw)
    base_sha = hashlib.sha256(raw).hexdigest()
    rows = []
    for sigma in SIGMAS:
        level = level_name(sigma)
        prompts = build_perturbed_prompts(base, sigma, ROOT_SEED, PRIMARY_DATASETS, base_sha)
        path = args.out_dir / f"prompts_{level}.json"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force only on purpose")
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(prompts, indent=1, sort_keys=True) + "\n"
        path.write_text(blob, encoding="utf-8", newline="\n")
        rows += describe(prompts, level)
        print(f"{path}: {len(prompts['images'])} images, sha256 {hashlib.sha256(blob.encode()).hexdigest()}")
    args.levels_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.levels_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
