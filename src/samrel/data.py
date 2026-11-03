"""Locating PraNet images and masks, reading masks (PROTOCOL §1)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from .constants import DATASETS

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_NUMERIC = re.compile(r"^\d+$")


@dataclass(frozen=True)
class Record:
    """One image of the PraNet split. ``mask_path`` is None when no mask file matches."""

    dataset: str
    image_id: str
    image_path: Path | None
    mask_path: Path | None

    @property
    def role(self) -> str:
        return DATASETS[self.dataset][0]


def classify_train_source(stem: str) -> str:
    """Kvasir vs ClinicDB inside TrainDataset (PROTOCOL §1.2).

    Assumption to be verified on the real data by the audit: ClinicDB files have purely numeric
    names. The audit stops if the counts are not 900 / 550.
    """
    return "clinicdb_train" if _NUMERIC.match(stem) else "kvasir_train"


def _by_stem(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"missing folder: {folder}")
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.stem in out:
                raise ValueError(f"two files share the stem {p.stem!r} in {folder}")
            out[p.stem] = p
    return out


def _pair(dataset_of, images: dict[str, Path], masks: dict[str, Path]) -> list[Record]:
    records = []
    for stem in sorted(set(images) | set(masks)):
        ds = dataset_of(stem)
        records.append(Record(ds, f"{ds}/{stem}", images.get(stem), masks.get(stem)))
    return records


def discover(root: str | Path, data_cfg: dict[str, Any]) -> list[Record]:
    """List every image (and orphan mask) of the PraNet train and test folders."""
    root = Path(root)
    train = root / data_cfg["train_dir"]
    records = _pair(
        classify_train_source,
        _by_stem(train / data_cfg["train_image_subdir"]),
        _by_stem(train / data_cfg["train_mask_subdir"]),
    )
    for key, folder in data_cfg["test_folders"].items():
        base = root / data_cfg["test_dir"] / folder
        records += _pair(
            lambda _stem, k=key: k,
            _by_stem(base / data_cfg["test_image_subdir"]),
            _by_stem(base / data_cfg["test_mask_subdir"]),
        )
    return records


def read_mask_raw(path: str | Path) -> np.ndarray:
    """Grey-level mask as uint8 (RGB masks are converted with PIL mode ``L``)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def binarize(raw: np.ndarray, threshold: int) -> np.ndarray:
    """``raw > threshold`` (PROTOCOL §1.3, threshold Q-B1)."""
    return raw > threshold


def read_mask(path: str | Path, threshold: int) -> np.ndarray:
    return binarize(read_mask_raw(path), threshold)


def read_image(path: str | Path) -> np.ndarray:
    """RGB uint8 image H×W×3."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def image_size(path: str | Path) -> tuple[int, int]:
    """(H, W) without decoding the pixels."""
    with Image.open(path) as im:
        w, h = im.size
    return h, w


def label_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Connected components (PROTOCOL §1.4)."""
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    labels, n = ndimage.label(mask, structure=structure)
    return labels, int(n)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def data_hash(records: list[Record]) -> str:
    """Hash of (image_id, image bytes, mask bytes) over every record, order independent."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r.image_id):
        h.update(r.image_id.encode())
        for p in (r.image_path, r.mask_path):
            h.update((file_sha256(p) if p else "-").encode())
    return h.hexdigest()
