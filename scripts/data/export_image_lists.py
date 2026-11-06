"""Write the list of every image with its dataset, role and split, and check the counts of the data table.

Sources: the audit table (results/data_checks/audit_per_image.csv) and data/splits/val_split.json. With
--data-root the sha256 of each image and mask file is added, so that anyone holding the PraNet archives
can confirm they have the same files. No image is copied.

    python scripts/data/export_image_lists.py --audit results/data_checks/audit_per_image.csv \
        --val-split data/splits/val_split.json --out-csv results/data_checks/image_lists.csv [--data-root data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from samrel.config import load_config
from samrel.data import discover


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--val-split", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--data-root", type=Path)
    args = ap.parse_args(argv)
    audit = pd.read_csv(args.audit)
    split = json.loads(args.val_split.read_text(encoding="utf-8"))
    val = {i for ids in split["val"].values() for i in ids}
    train = {i for ids in split["train"].values() for i in ids}
    if val & train:
        raise SystemExit("STOP: an image is in both train and validation")

    def which(r) -> str:
        if r.role != "train":
            return "test"
        return "validation" if r.image_id in val else "train"

    out = audit[["dataset", "role", "image_id", "image_file", "mask_file", "img_h", "img_w"]].copy()
    out["split"] = [which(r) for r in audit.itertuples()]
    if set(out.loc[out["split"] == "train", "image_id"]) != train:
        raise SystemExit("STOP: training images differ from val_split.json")
    if args.data_root:
        records = {r.image_id: r for r in discover(args.data_root, load_config("base")["data"])}
        out["image_sha256"] = [sha(records[i].image_path) for i in out["image_id"]]
        out["mask_sha256"] = [sha(records[i].mask_path) for i in out["image_id"]]
    out = out.sort_values(["dataset", "image_id"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.groupby(["dataset", "role", "split"]).size().to_string())
    print("total", len(out), "duplicate image_id", int(out["image_id"].duplicated().sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
