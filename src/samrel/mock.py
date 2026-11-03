"""Tiny fake PraNet tree for structural tests. Never used for real numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _disk(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _save_pair(img_dir: Path, mask_dir: Path, stem: str, rng: np.random.Generator,
               h: int = 48, w: int = 64, n_polyps: int = 1) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((h, w), dtype=bool)
    for k in range(n_polyps):
        mask |= _disk(h, w, int(rng.integers(10, h - 10)), 12 + 30 * k, 6)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    img[mask] = (200, 80, 80)
    Image.fromarray(img).save(img_dir / f"{stem}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{stem}.png")


def make_fake_pranet(root: Path, n_per_set: int = 6, seed: int = 0) -> Path:
    """Write a PraNet-shaped tree: numeric stems for ClinicDB, hash-like stems for Kvasir."""
    rng = np.random.default_rng(seed)
    train = root / "TrainDataset"
    for i in range(n_per_set):
        _save_pair(train / "image", train / "masks", str(i + 1), rng)
        _save_pair(train / "image", train / "masks", f"cju{i:04d}kvasir", rng, n_polyps=1 + i % 2)
    for folder in ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"):
        base = root / "TestDataset" / folder
        for i in range(n_per_set):
            _save_pair(base / "images", base / "masks", f"{i}", rng)
    return root
