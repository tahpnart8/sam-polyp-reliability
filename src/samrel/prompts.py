"""Box prompts, generated once for every image (PROTOCOL §3)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from .data import Record, label_components, read_mask
from .seeding import rng_for


def tight_boxes(mask: np.ndarray, connectivity: int, min_component_px: int = 0) -> list[list[int]]:
    """Inclusive ``[x0, y0, x1, y1]`` of every component with at least ``min_component_px`` pixels.

    The size filter (Q-B2, approved 2026-10-29) drops the 1-10 px annotation speckle found by the
    audit; it only affects prompts, never the ground-truth mask used for metrics.
    """
    labels, n = label_components(mask, connectivity)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    boxes = []
    for k, sl in enumerate(ndimage.find_objects(labels)[:n], start=1):
        if sizes[k] < min_component_px:
            continue
        ys, xs = sl
        boxes.append([int(xs.start), int(ys.start), int(xs.stop - 1), int(ys.stop - 1)])
    return boxes


def jitter_box(box: list[int], h: int, w: int, rng: np.random.Generator, jitter_max: float) -> list[int]:
    """Enlarge each side by ``U(0, jitter_max)`` times the box size, clip to the image."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    u = rng.uniform(0.0, jitter_max, size=4)
    nx0 = int(np.clip(round(x0 - u[0] * bw), 0, w - 1))
    ny0 = int(np.clip(round(y0 - u[1] * bh), 0, h - 1))
    nx1 = int(np.clip(round(x1 + u[2] * bw), 0, w - 1))
    ny1 = int(np.clip(round(y1 + u[3] * bh), 0, h - 1))
    return [nx0, ny0, nx1, ny1]


def boxes_from_mask(
    mask: np.ndarray, rng: np.random.Generator, jitter_max: float, connectivity: int = 8,
    min_component_px: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(tight, jittered)`` boxes for one binary mask."""
    h, w = mask.shape
    tight = tight_boxes(mask, connectivity, min_component_px)
    return tight, [jitter_box(b, h, w, rng, jitter_max) for b in tight]


def build_prompts(
    records: Iterable[Record],
    root_seed: int,
    jitter_max: float,
    mask_threshold: int,
    connectivity: int,
    data_hash: str,
    min_component_px: int = 0,
) -> dict[str, Any]:
    """Build the ``prompts.json`` payload.

    Each image draws from its own derived RNG, so its boxes do not depend on which other images
    exist or on iteration order. Excluded images still get prompts; exclusion is applied at
    analysis time (PROTOCOL §1.5) so the file never has to be regenerated.
    """
    images: dict[str, Any] = {}
    for r in sorted(records, key=lambda r: r.image_id):
        if r.mask_path is None or r.image_path is None:
            continue
        mask = read_mask(r.mask_path, mask_threshold)
        rng = rng_for(root_seed, "prompts", r.image_id)
        tight, boxes = boxes_from_mask(mask, rng, jitter_max, connectivity, min_component_px)
        if not boxes:
            raise ValueError(f"{r.image_id}: no component of at least {min_component_px} px")
        images[r.image_id] = {"h": int(mask.shape[0]), "w": int(mask.shape[1]),
                              "boxes": boxes, "tight_boxes": tight}
    return {
        "version": 1, "root_seed": root_seed, "jitter_max": jitter_max,
        "connectivity": connectivity, "mask_threshold": mask_threshold,
        "min_component_px": min_component_px,
        "data_hash": data_hash, "images": images,
    }


def perturb_box(box: list[int], h: int, w: int, u: np.ndarray, sigma: float) -> tuple[list[int], bool, bool]:
    """Shift the centre by ``sigma*u[0:2]`` and rescale by ``1 + sigma*u[2:4]`` of the box size
    (PROTOCOL §14.1). Returns ``(box, clipped, degenerate)``; ``sigma = 0`` returns ``box`` unchanged.
    """
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = (x0 + x1 + 1) / 2 + sigma * u[0] * bw, (y0 + y1 + 1) / 2 + sigma * u[1] * bh
    nw, nh = bw * (1 + sigma * u[2]), bh * (1 + sigma * u[3])
    half_up = lambda v: int(np.floor(v + 0.5))  # noqa: E731  (round() rounds .5 to even and loses a pixel)
    raw = [half_up(cx - nw / 2), half_up(cy - nh / 2), half_up(cx + nw / 2) - 1, half_up(cy + nh / 2) - 1]
    out = [int(np.clip(raw[0], 0, w - 1)), int(np.clip(raw[1], 0, h - 1)),
           int(np.clip(raw[2], 0, w - 1)), int(np.clip(raw[3], 0, h - 1))]
    clipped = out != raw
    degenerate = out[2] < out[0] or out[3] < out[1]
    if degenerate:
        px, py = int(np.clip(round(cx - 0.5), 0, w - 1)), int(np.clip(round(cy - 0.5), 0, h - 1))
        out = [px, py, px, py]
    return out, clipped, degenerate


def box_iou(a: list[int], b: list[int]) -> float:
    """IoU of two inclusive pixel boxes."""
    ix = min(a[2], b[2]) - max(a[0], b[0]) + 1
    iy = min(a[3], b[3]) - max(a[1], b[1]) + 1
    inter = max(ix, 0) * max(iy, 0)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def build_perturbed_prompts(base: dict[str, Any], sigma: float, root_seed: int, datasets: Iterable[str],
                            base_sha256: str) -> dict[str, Any]:
    """Perturbed copy of ``prompts.json`` for the test images of ``datasets`` (PROTOCOL §14.2).

    ``u`` of box ``j`` depends only on ``(root_seed, image_id, j)``, so every level reuses the
    same direction and only the magnitude ``sigma`` changes. ``tight_boxes`` stay the ground-truth
    boxes (ECE ROI, §7.2); each entry also records ``clipped`` and ``degenerate`` per box.
    """
    keep = set(datasets)
    images: dict[str, Any] = {}
    for image_id in sorted(base["images"]):
        if image_id.split("/", 1)[0] not in keep:
            continue
        e = base["images"][image_id]
        boxes, clipped, degenerate = [], [], []
        for j, b in enumerate(e["boxes"]):
            u = rng_for(root_seed, "prompt_perturb", image_id, j).uniform(-1.0, 1.0, size=4)
            nb, c, d = perturb_box(b, e["h"], e["w"], u, sigma)
            boxes.append(nb)
            clipped.append(c)
            degenerate.append(d)
        images[image_id] = {"h": e["h"], "w": e["w"], "boxes": boxes, "tight_boxes": e["tight_boxes"],
                            "oracle_boxes": e["boxes"], "clipped": clipped, "degenerate": degenerate}
    out = {k: v for k, v in base.items() if k != "images"}
    out["perturb"] = {"sigma": sigma, "base_file": "prompts.json", "base_sha256": base_sha256,
                      "datasets": sorted(keep)}
    out["images"] = images
    return out
