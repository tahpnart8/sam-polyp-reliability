"""Fine-tuning A1 (decoder) and A2 (LoRA + decoder) with resumable checkpoints (PROTOCOL §6).

One training sample is one prompt box with the mask of its own connected component (Q-B5).
Boxes come from ``prompts.json`` (fixed, PROTOCOL §3); the split from ``val_split.json`` (§4).
Seed redraws the minibatch order and, for A2, the LoRA A init (§2). Nothing here reads test data.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .constants import TRAIN_LOG_COLUMNS
from .data import Record, label_components, read_image, read_mask
from .infer import make_predictor, predict_image
from .losses import sam_loss
from .metrics import dice_iou
from .seeding import derive_seed

LOW_RES = 256


@dataclass(frozen=True)
class Sample:
    image_id: str
    box_index: int


def component_masks(mask: np.ndarray, tight_boxes: list[list[int]], connectivity: int) -> list[np.ndarray]:
    """Mask of the component behind each tight box, matched by its exact bounding box."""
    labels, n = label_components(mask, connectivity)
    from scipy import ndimage

    by_box = {}
    for k, (ys, xs) in enumerate(ndimage.find_objects(labels)[:n], start=1):
        by_box.setdefault((xs.start, ys.start, xs.stop - 1, ys.stop - 1), k)
    out = []
    for b in tight_boxes:
        k = by_box.get(tuple(b))
        if k is None:
            raise ValueError(f"box {b} has no matching component")
        out.append(labels == k)
    return out


def build_samples(ids: Iterable[str], prompts: dict) -> list[Sample]:
    return [Sample(i, k) for i in sorted(ids) for k in range(len(prompts["images"][i]["boxes"]))]


def lr_lambda(total_steps: int, warmup_steps: int = 0):
    """Cosine from 1 to 0 over optimizer steps, optional linear warmup (Q-B4)."""
    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return f


class SampleLoader:
    """Prepares the 1024 input and the 256×256 low-res target of one sample (cached per image)."""

    def __init__(self, model, records: dict[str, Record], prompts: dict, cfg: dict):
        from segment_anything.utils.transforms import ResizeLongestSide

        self.model, self.records, self.prompts, self.cfg = model, records, prompts, cfg
        self.transform = ResizeLongestSide(cfg["model"]["image_size"])
        self._targets: dict[str, list[torch.Tensor]] = {}

    def image_tensor(self, image_id: str) -> torch.Tensor:
        """SAM-normalised, padded ``[3, 1024, 1024]`` float tensor on the model's device."""
        img = read_image(self.records[image_id].image_path)
        x = torch.as_tensor(self.transform.apply_image(img)).permute(2, 0, 1).float()
        # preprocess() uses the model's pixel_mean/std buffers, so the image must be on their device.
        x = x.to(self.model.pixel_mean.device)
        return self.model.preprocess(x[None])[0]

    def targets(self, image_id: str) -> list[torch.Tensor]:
        """Low-res target per box: the component mask resized like the image, padded, then
        bilinearly reduced to 256×256 and thresholded at 0.5."""
        if image_id not in self._targets:
            d = self.cfg["data"]
            entry = self.prompts["images"][image_id]
            gt = read_mask(self.records[image_id].mask_path, d["mask_threshold"])
            size = self.cfg["model"]["image_size"]
            h, w = self.transform.get_preprocess_shape(gt.shape[0], gt.shape[1], size)
            out = []
            for comp in component_masks(gt, entry["tight_boxes"], d["connectivity"]):
                m = torch.as_tensor(comp, dtype=torch.float32)[None, None]
                m = F.interpolate(m, (h, w), mode="bilinear", align_corners=False)
                m = F.pad(m, (0, size - w, 0, size - h))
                m = F.interpolate(m, (LOW_RES, LOW_RES), mode="bilinear", align_corners=False)
                out.append((m[0] >= 0.5).float())
            self._targets[image_id] = out
        return self._targets[image_id]

    def box_tensor(self, image_id: str, k: int) -> torch.Tensor:
        entry = self.prompts["images"][image_id]
        b = np.asarray([entry["boxes"][k]], dtype=np.float32)
        return torch.as_tensor(self.transform.apply_boxes(b, (entry["h"], entry["w"])))


def forward_sample(model, embedding: torch.Tensor, box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder pass for one image embedding ``[1,256,64,64]`` and one box ``[1,4]``."""
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=box, masks=None)
    logits, iou = model.mask_decoder(
        image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    return logits, iou


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def load_trainable(model, state: dict[str, torch.Tensor]) -> None:
    names = {k for k, v in model.named_parameters() if v.requires_grad}
    if set(state) != names:
        raise KeyError(f"trainable keys differ: missing {sorted(names - set(state))[:3]}, "
                       f"unexpected {sorted(set(state) - names)[:3]}")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def validation_dice(model, val_ids: list[str], loader: SampleLoader, threshold: float) -> float:
    """Mean per-image Dice on validation, same inference path as the test runs (§5.1)."""
    model.eval()
    pred = make_predictor(model)
    dices = []
    for i in val_ids:
        entry = loader.prompts["images"][i]
        prob, _ = predict_image(pred, read_image(loader.records[i].image_path), entry["boxes"])
        gt = read_mask(loader.records[i].mask_path, loader.cfg["data"]["mask_threshold"])
        dices.append(dice_iou(prob >= threshold, gt)[0])
    return float(np.mean(dices))


def train(model, cfg: dict, seed: int, records: dict[str, Record], prompts: dict, split: dict,
          work_dir: Path, device: str = "cuda", embed_cache: dict[str, torch.Tensor] | None = None,
          max_steps: int | None = None, log=print) -> dict[str, Any]:
    """Train, checkpoint every epoch into ``work_dir`` and resume from it if present.

    ``embed_cache`` (A1) maps image_id to a frozen encoder embedding; otherwise the encoder runs
    every step (A2, gradients flow into LoRA only). ``max_steps`` caps optimizer steps for the
    smoke run. Returns the best epoch and its validation Dice.
    """
    t = cfg["train"]
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model.to(device)
    loader = SampleLoader(model, records, prompts, cfg)
    train_ids = [i for ids in split["train"].values() for i in ids]
    val_ids = sorted(i for ids in split["val"].values() for i in ids)
    samples = build_samples(train_ids, prompts)
    per_step = t["batch_size"] * t["grad_accum"]
    steps_per_epoch = math.ceil(len(samples) / per_step)
    total = steps_per_epoch * t["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, t.get("warmup_steps", 0)))
    amp = t["amp"] == "fp16" and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    work_dir.mkdir(parents=True, exist_ok=True)
    last, best_path, log_path = work_dir / "last.pt", work_dir / "best_trainable.pt", work_dir / "train_log.csv"
    start_epoch, step, best = 0, 0, {"epoch": -1, "val_dice": -1.0}
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        load_trainable(model, ck["trainable"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        log(f"resumed after epoch {ck['epoch']} (step {step})")
    if not log_path.exists():
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(TRAIN_LOG_COLUMNS)

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        t0 = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator().manual_seed(derive_seed(seed, "train", "shuffle", epoch))
        order = torch.randperm(len(samples), generator=gen).tolist()
        sums = np.zeros(4)
        n_micro = 0
        opt.zero_grad(set_to_none=True)
        for start in range(0, len(order), t["batch_size"]):
            batch = [samples[j] for j in order[start:start + t["batch_size"]]]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, ious, tgts = [], [], []
                for s in batch:
                    if embed_cache is not None:
                        emb = embed_cache[s.image_id].to(device).float()[None]
                    else:
                        emb = model.image_encoder(loader.image_tensor(s.image_id).to(device)[None])
                    lg, io = forward_sample(model, emb, loader.box_tensor(s.image_id, s.box_index).to(device))
                    logits.append(lg); ious.append(io)
                    tgts.append(loader.targets(s.image_id)[s.box_index].to(device))
                out = sam_loss(torch.cat(logits).float(), torch.stack(tgts), torch.cat(ious), t["loss"])
            scaler.scale(out["loss"] / t["grad_accum"]).backward()
            sums += [out["loss"].item(), out["focal"].item(), out["dice"].item(), out["iou_mse"].item()]
            n_micro += 1
            last_micro = start + t["batch_size"] >= len(order)
            if n_micro % t["grad_accum"] == 0 or last_micro:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
        val = validation_dice(model, val_ids, loader, cfg["model"]["threshold"])
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0
        m = sums / max(1, n_micro)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, step, *m.round(6), round(val, 6), sched.get_last_lr()[0],
                                     round(time.perf_counter() - t0, 1), round(peak, 1)])
        log(f"epoch {epoch} step {step} loss {m[0]:.4f} val_dice {val:.4f}")
        if val > best["val_dice"]:  # strict: ties keep the earlier epoch (PROTOCOL §6)
            best = {"epoch": epoch, "val_dice": val}
            torch.save(trainable_state(model), best_path)
        torch.save({"trainable": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best}, last)
        if max_steps is not None and step >= max_steps:
            break
    return best


@torch.no_grad()
def cache_embeddings(model, image_ids: Iterable[str], loader: SampleLoader, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Frozen encoder embeddings (float16, CPU) for A1. Valid because there is no augmentation."""
    model.to(device).eval()
    return {i: model.image_encoder(loader.image_tensor(i).to(device)[None])[0].half().cpu()
            for i in sorted(image_ids)}
