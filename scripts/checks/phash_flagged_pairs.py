"""List every external-test image flagged as a near-duplicate of a training image, with all its pairs.

Reads overlap_pairs.csv (written by the dataset audit), keeps pairs between ColonDB or ETIS and a training
image within the exclusion distance on the whole-image pHash, and states which of them would also
pass the centre-crop test (the exclusion rule of §1.5 needs both). With --data-root the pHash of both
images of each pair is recomputed from the image files as a check. With --montage it draws all pairs
side by side; the montage holds dataset images, so it is written to a path you choose and is not
meant for this repository.

    python scripts/checks/phash_flagged_pairs.py --data-checks results/data_checks \
        --out-csv results/data_checks/phash_flagged_pairs.csv [--data-root data/raw] [--montage <file.jpg>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageDraw

from samrel.config import load_config
from samrel.data import discover

EXTERNAL = ("colondb", "etis")
TRAIN = ("kvasir_train", "clinicdb_train")
THRESHOLD = 6


def phashes(path: Path) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        dx, dy = int(w * 0.2), int(h * 0.2)
        return imagehash.phash(rgb), imagehash.phash(rgb.crop((dx, dy, w - dx, h - dy)))


def flagged(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples():
        if r.hamming > THRESHOLD:
            continue
        if r.dataset_a in EXTERNAL and r.dataset_b in TRAIN:
            rows.append((r.dataset_a, r.image_a, r.dataset_b, r.image_b, r.hamming, r.hamming_center))
        elif r.dataset_b in EXTERNAL and r.dataset_a in TRAIN:
            rows.append((r.dataset_b, r.image_b, r.dataset_a, r.image_a, r.hamming, r.hamming_center))
    out = pd.DataFrame(rows, columns=["dataset", "image_id", "train_dataset", "train_image_id", "hamming_whole", "hamming_center"])
    out["both_within_threshold"] = (out["hamming_whole"] <= THRESHOLD) & (out["hamming_center"] <= THRESHOLD)
    return out.sort_values(["dataset", "image_id", "hamming_whole", "train_image_id"]).reset_index(drop=True)


def draw_montage(df: pd.DataFrame, records: dict, path: Path, per_row: int = 3, side: int = 260) -> None:
    n_rows = (len(df) + per_row - 1) // per_row
    canvas = Image.new("RGB", (per_row * (2 * side + 30), n_rows * (side + 28)), "white")
    pen = ImageDraw.Draw(canvas)
    for k, r in enumerate(df.itertuples()):
        x, y = (k % per_row) * (2 * side + 30), (k // per_row) * (side + 28)
        for j, image_id in enumerate((r.image_id, r.train_image_id)):
            with Image.open(records[image_id].image_path) as im:
                canvas.paste(im.convert("RGB").resize((side, side)), (x + j * side, y + 20))
        pen.text((x + 2, y + 4), f"{k + 1}: {r.image_id} | {r.train_image_id} | whole {r.hamming_whole} centre {r.hamming_center}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-checks", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--montage", type=Path)
    args = ap.parse_args(argv)
    df = flagged(pd.read_csv(args.data_checks / "overlap_pairs.csv"))
    chars = pd.read_csv(args.data_checks / "dataset_characteristics.csv").set_index("dataset")
    print(f"pairs {len(df)}, images {df['image_id'].nunique()}, by dataset {df.groupby('dataset')['image_id'].nunique().to_dict()}")
    print(f"pairs within {THRESHOLD} on both hashes: {int(df['both_within_threshold'].sum())}")
    for ds in EXTERNAL:
        print(ds, "audit n_near_dup_in_train", int(chars.loc[ds, "n_near_dup_in_train"]), "centre", int(chars.loc[ds, "n_near_dup_in_train_center"]),
              "excluded", int(chars.loc[ds, "n_excluded"]))
    if args.data_root:
        cfg = load_config("base")["data"]
        records = {r.image_id: r for r in discover(args.data_root, cfg)}
        bad = 0
        for r in df.itertuples():
            a, ac = phashes(records[r.image_id].image_path)
            b, bc = phashes(records[r.train_image_id].image_path)
            bad += int((a - b) != r.hamming_whole) + int((ac - bc) != r.hamming_center)
        print(f"recomputed from image files: {bad} differences in {2 * len(df)} distances")
        if args.montage:
            draw_montage(df, records, args.montage)
            print(args.montage)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
