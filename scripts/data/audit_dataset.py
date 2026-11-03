"""Data audit of the PraNet split (PROTOCOL §1.5-§1.6; Table 2).

Usage (personal machine or a Kaggle CPU kernel):
    python scripts/data/audit_dataset.py --data-root <dir with TrainDataset/ TestDataset/> \
        --out-dir results/data_checks --splits-dir data/splits

Writes audit_per_image.csv, dataset_characteristics.csv, overlap_pairs.csv, audit_meta.json into
--out-dir and excluded_images.json into --splits-dir. Exits with code 2 when an image count does
not match PROTOCOL §1.1 (a stop point: do not continue).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from samrel.audit import audit_dataset
from samrel.config import config_sha256, load_config
from samrel.data import data_hash, discover
from samrel.manifest import environment, git_commit, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--config", default="base")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    records = discover(args.data_root, cfg["data"])
    res = audit_dataset(records, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    res["per_image"].to_csv(args.out_dir / "audit_per_image.csv", index=False)
    res["summary"].to_csv(args.out_dir / "dataset_characteristics.csv", index=False)
    res["pairs"].to_csv(args.out_dir / "overlap_pairs.csv", index=False)
    commit, dirty = git_commit()
    dhash = data_hash(records)
    write_json({"created_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
                "commit_dirty": dirty, "config_sha256": config_sha256(cfg), "data_hash": dhash,
                "phash_exclude_max": cfg["audit"]["phash_exclude_max"],
                "environment": environment()}, args.out_dir / "audit_meta.json")
    # Text kept identical to the original run so the file hash in the manifests still matches;
    # "CONTRACT" is the development name of docs/PROTOCOL.md.
    write_json({"version": 1, "rule": "CONTRACT §1.5: colondb/etis image with pHash Hamming "
                f"<= {cfg['audit']['phash_exclude_max']} to any train image",
                "data_hash": dhash, "excluded": res["excluded"]},
               args.splits_dir / "excluded_images.json")

    summary = res["summary"]
    print(summary.to_string(index=False))
    bad = summary[~summary["count_ok"]]
    if len(bad):
        print("\nSTOP: image/mask counts differ from PROTOCOL §1.1 for: "
              + ", ".join(bad["dataset"]), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
