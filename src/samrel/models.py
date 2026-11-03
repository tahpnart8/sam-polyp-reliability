"""Build A0/A1/A2 from SAM ViT-B (PROTOCOL §5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .lora import apply_lora
from .seeding import derive_seed


def build_model(cfg: dict[str, Any], checkpoint: str | Path | None, seed: int = 0,
                sam: nn.Module | None = None) -> nn.Module:
    """SAM ViT-B with the trainable parts of the configuration unfrozen.

    ``checkpoint=None`` builds random weights; only tests do that (architecture and counts).
    ``sam`` lets tests pass a tiny SAM so the training loop runs on CPU in seconds.
    """
    m = cfg["model"]
    if sam is None:
        from segment_anything import sam_model_registry  # imported lazily: heavy, optional for tests

        sam = sam_model_registry[m["type"]](checkpoint=str(checkpoint) if checkpoint else None)
    for p in sam.parameters():
        p.requires_grad_(False)
    if m["lora"]["enabled"]:
        gen = torch.Generator().manual_seed(derive_seed(seed, "train", "lora_init"))
        apply_lora(sam.image_encoder, m["lora"]["rank"], m["lora"]["targets"], gen)
    if m["train_decoder"]:
        for p in sam.mask_decoder.parameters():
            p.requires_grad_(True)
    return sam


def param_row(cfg: dict[str, Any], counts: dict[str, int], commit: str) -> dict[str, Any]:
    """One row of ``trainable_params.csv`` (PROTOCOL §5.5)."""
    lora = cfg["model"]["lora"]
    return {
        "config": cfg["name"], **counts,
        "lora_targets": "+".join(lora["targets"]) if lora["enabled"] else "",
        "lora_rank": lora["rank"] if lora["enabled"] else 0,
        "commit": commit,
    }
