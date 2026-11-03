"""Box-prompted inference for A0/A1/A2 and saving probability maps (PROTOCOL §5.1, §8, §10)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import Record, read_image


def make_predictor(model):
    """Official ``SamPredictor``: ResizeLongestSide(1024), SAM pixel normalisation, padding, and
    ``postprocess_masks`` back to the original size (PROTOCOL §5.1)."""
    from segment_anything import SamPredictor

    return SamPredictor(model)


@torch.no_grad()
def predict_image(predictor, image: np.ndarray, boxes: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Probability map ``p`` (float32, H×W) and per-box predicted IoU for one image.

    All boxes of the image go through the decoder in one batch; each box is decoded independently,
    so this equals running SAM once per box. Boxes are passed as stored in ``prompts.json``
    (inclusive pixel coordinates). Masks of several boxes are merged by pixel-wise max of the
    probabilities.
    """
    if not boxes:
        raise ValueError("every image has at least one ground-truth box (PROTOCOL §3)")
    predictor.set_image(image, image_format="RGB")
    return decode_boxes(predictor, image.shape[:2], boxes)


@torch.no_grad()
def decode_boxes(predictor, image_hw: tuple[int, int], boxes: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Second half of :func:`predict_image` on the embedding already set in ``predictor``; lets the
    prompt-perturbation run (PROTOCOL §14.3) encode an image once and decode several box sets."""
    if not boxes:
        raise ValueError("every image has at least one box (PROTOCOL §3, §14.1)")
    b = torch.as_tensor(np.asarray(boxes, dtype=np.float32), device=predictor.device)
    b = predictor.transform.apply_boxes_torch(b, image_hw)
    logits, iou_pred, _ = predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=b, multimask_output=False, return_logits=True)
    prob = torch.sigmoid(logits.float())[:, 0].amax(dim=0)
    return prob.cpu().numpy().astype(np.float32), iou_pred[:, 0].float().cpu().numpy()


def save_prediction(path: Path, prob: np.ndarray, iou_pred: np.ndarray, boxes: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prob=prob.astype(np.float16), iou_pred=iou_pred.astype(np.float32),
                        boxes=np.asarray(boxes, dtype=np.int32))


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def run_inference(model, records: Iterable[Record], prompts: dict[str, Any], out_dir: Path,
                  device: str = "cuda") -> list[dict[str, Any]]:
    """Predict every record and write ``<out_dir>/<dataset>/<stem>.npz``; returns timing rows.

    Timing covers encoder, every box and postprocess for an image already in RAM (PROTOCOL §10);
    disk reads and writes are outside the timed window.
    """
    model.to(device).eval()
    predictor = make_predictor(model)
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    rows = []
    for r in sorted(records, key=lambda r: r.image_id):
        entry = prompts["images"][r.image_id]
        image = read_image(r.image_path)
        if image.shape[:2] != (entry["h"], entry["w"]):
            raise ValueError(f"{r.image_id}: image size differs from prompts.json")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob, iou_pred = predict_image(predictor, image, entry["boxes"])
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        stem = r.image_id.split("/", 1)[1]
        save_prediction(out_dir / r.dataset / f"{stem}.npz", prob, iou_pred, entry["boxes"])
        rows.append({"image_id": r.image_id, "dataset": r.dataset, "seconds": seconds,
                     "n_boxes": len(entry["boxes"])})
    return rows
