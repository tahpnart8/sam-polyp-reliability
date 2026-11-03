"""Data audit for Table 2 and the near-duplicate rule (PROTOCOL §1.3-§1.6)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .constants import (AUDIT_PER_IMAGE_COLUMNS, DATASET_CHARACTERISTICS_COLUMNS, DATASETS,
                        EXCLUSION_DATASETS, OVERLAP_PAIR_COLUMNS, TRAIN_DATASETS)
from .data import Record, binarize, image_size, label_components, read_mask_raw


def phash_hex(path, center_frac: float | None = None) -> str:
    """64-bit pHash of the image, or of its central ``center_frac`` crop.

    The centre crop is the second check of PROTOCOL §1.5: colonoscopy frames share the same black
    octagonal border, which alone can bring unrelated low-texture frames within 6 bits.
    """
    import imagehash  # optional dependency, needed only by the audit

    with Image.open(path) as im:
        im = im.convert("RGB")
        if center_frac is not None:
            w, h = im.size
            dx, dy = int(w * (1 - center_frac) / 2), int(h * (1 - center_frac) / 2)
            im = im.crop((dx, dy, w - dx, h - dy))
        return str(imagehash.phash(im))


def _hex_to_bits(hexes: list[str]) -> np.ndarray:
    return np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(64)] for h in hexes], dtype=np.int32)


def hamming_matrix(hex_a: list[str], hex_b: list[str]) -> np.ndarray:
    """All pairwise Hamming distances between two lists of 64-bit pHashes."""
    a, b = _hex_to_bits(hex_a), _hex_to_bits(hex_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.int32)
    return a @ (1 - b).T + (1 - a) @ b.T


def audit_image(r: Record, threshold: int, connectivity: int, small_px: int,
                center_frac: float = 0.6) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in AUDIT_PER_IMAGE_COLUMNS}
    row.update(dataset=r.dataset, role=r.role, image_id=r.image_id,
               image_file=r.image_path.name if r.image_path else None,
               mask_file=r.mask_path.name if r.mask_path else None)
    if r.image_path:
        row["img_h"], row["img_w"] = image_size(r.image_path)
        row["phash"] = phash_hex(r.image_path)
        row["phash_center"] = phash_hex(r.image_path, center_frac)
    if r.mask_path:
        raw = read_mask_raw(r.mask_path)
        mask = binarize(raw, threshold)
        labels, n = label_components(mask, connectivity)
        sizes = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=np.int64)
        values = np.unique(raw)
        row.update(
            mask_h=int(raw.shape[0]), mask_w=int(raw.shape[1]),
            mask_unique_values=int(len(values)),
            mask_nonbinary=bool(not set(values.tolist()) <= {0, 255}),
            mask_empty=bool(not mask.any()),
            n_components=n, n_small_components=int((sizes < small_px).sum()),
            n_box_components=int((sizes >= small_px).sum()),
            fg_fraction=float(mask.mean()),
        )
    if r.image_path and r.mask_path:
        row["size_mismatch"] = (row["img_h"], row["img_w"]) != (row["mask_h"], row["mask_w"])
    return row


def near_duplicates(per_image: pd.DataFrame, exclude_max: int, report_max: int) -> tuple[pd.DataFrame, dict, dict, dict, list[str]]:
    """Pairs within ``report_max``, per-image flags and the exclusion list (PROTOCOL §1.5).

    An external-test image is excluded only when both the full-image and the centre-crop pHash
    are within ``exclude_max`` of some train image (Q-B10, PROTOCOL §12).
    """
    df = per_image.dropna(subset=["phash"]).reset_index(drop=True)
    D = hamming_matrix(df["phash"].tolist(), df["phash"].tolist())
    C = hamming_matrix(df["phash_center"].tolist(), df["phash_center"].tolist())
    np.fill_diagonal(D, 64 + 1)
    np.fill_diagonal(C, 64 + 1)
    ds = df["dataset"].to_numpy()
    is_train = np.isin(ds, TRAIN_DATASETS)

    pairs = []
    for i, j in zip(*np.where(np.triu(D <= report_max, k=1))):
        pairs.append({"dataset_a": ds[i], "image_a": df.at[i, "image_id"],
                      "dataset_b": ds[j], "image_b": df.at[j, "image_id"], "hamming": int(D[i, j]),
                      "hamming_center": int(C[i, j])})
    pairs_df = pd.DataFrame(pairs, columns=list(OVERLAP_PAIR_COLUMNS))

    within: dict[str, bool] = {}
    in_train: dict[str, bool] = {}
    in_train_center: dict[str, bool] = {}
    excluded: list[str] = []
    for i, image_id in enumerate(df["image_id"]):
        same = ds == ds[i]
        within[image_id] = bool((D[i, same] <= exclude_max).any())
        # For a train set, "in train" means the other train set (PROTOCOL §1.6).
        ref = is_train & ~same if is_train[i] else is_train
        in_train[image_id] = bool((D[i, ref] <= exclude_max).any())
        in_train_center[image_id] = bool((C[i, ref] <= exclude_max).any())
        both = bool(((D[i, ref] <= exclude_max) & (C[i, ref] <= exclude_max)).any())
        if ds[i] in EXCLUSION_DATASETS and both:
            excluded.append(image_id)
    return pairs_df, within, in_train, in_train_center, sorted(excluded)


def _hw(h, w) -> str:
    return f"{int(h)}x{int(w)}"


def summarise(per_image: pd.DataFrame, within: dict, in_train: dict, in_train_center: dict,
              excluded: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, (role, n_expected) in DATASETS.items():
        g = per_image[per_image["dataset"] == dataset]
        imgs = g[g["image_file"].notna()]
        masks = g[g["mask_file"].notna()]
        sizes = Counter(zip(imgs["img_h"], imgs["img_w"]))
        by_area = sorted(sizes, key=lambda s: (s[0] * s[1], s))
        rows.append({
            "dataset": dataset, "role": role, "n_expected": n_expected,
            "n_images": len(imgs), "n_masks": len(masks),
            "n_images_without_mask": int((g["image_file"].notna() & g["mask_file"].isna()).sum()),
            "n_masks_without_image": int((g["image_file"].isna() & g["mask_file"].notna()).sum()),
            "count_ok": bool(len(imgs) == n_expected and len(masks) == n_expected),
            "size_min_hw": _hw(*by_area[0]) if sizes else None,
            "size_max_hw": _hw(*by_area[-1]) if sizes else None,
            "size_mode_hw": _hw(*max(sorted(sizes), key=lambda s: sizes[s])) if sizes else None,
            "n_distinct_sizes": len(sizes),
            "n_mask_nonbinary": int(masks["mask_nonbinary"].fillna(False).astype(bool).sum()),
            "n_mask_empty": int(masks["mask_empty"].fillna(False).astype(bool).sum()),
            "n_size_mismatch": int(g["size_mismatch"].fillna(False).astype(bool).sum()),
            "n_multi_polyp": int((masks["n_box_components"] >= 2).sum()),
            "max_components": int(masks["n_components"].max()) if len(masks) else 0,
            "n_small_components": int(masks["n_small_components"].sum()) if len(masks) else 0,
            "fg_fraction_median": float(masks["fg_fraction"].median()) if len(masks) else None,
            "n_near_dup_within": int(sum(within.get(i, False) for i in imgs["image_id"])),
            "n_near_dup_in_train": int(sum(in_train.get(i, False) for i in imgs["image_id"])),
            "n_near_dup_in_train_center": int(sum(in_train_center.get(i, False) for i in imgs["image_id"])),
            "n_excluded": int(sum(i in set(excluded) for i in imgs["image_id"])),
        })
    return pd.DataFrame(rows, columns=list(DATASET_CHARACTERISTICS_COLUMNS))


def audit_dataset(records: list[Record], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the whole audit; returns per-image table, summary, pairs and exclusion list."""
    d, a = cfg["data"], cfg["audit"]
    per_image = pd.DataFrame(
        [audit_image(r, d["mask_threshold"], d["connectivity"], cfg["prompt"]["min_box_component_px"],
                     a["phash_center_frac"])
         for r in sorted(records, key=lambda r: r.image_id)],
        columns=list(AUDIT_PER_IMAGE_COLUMNS),
    )
    pairs, within, in_train, in_center, excluded = near_duplicates(
        per_image, a["phash_exclude_max"], a["phash_report_max"])
    summary = summarise(per_image, within, in_train, in_center, excluded)
    return {"per_image": per_image, "summary": summary, "pairs": pairs, "excluded": excluded}
