"""LoRA on the q and v branches of SAM's fused ``qkv`` projection (PROTOCOL §5.3, §5.5)."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LoRAQKV(nn.Module):
    """Wrap a frozen ``nn.Linear(dim, 3*dim)`` and add ``B A x`` to the selected branches.

    Scale is 1 (no alpha), A is Kaiming-uniform, B is zero, following SAMed, so the wrapped layer
    starts exactly equal to the pretrained one.
    """

    def __init__(self, qkv: nn.Linear, rank: int, targets: Iterable[str] = ("q", "v"),
                 generator: torch.Generator | None = None):
        super().__init__()
        if qkv.out_features != 3 * qkv.in_features:
            raise ValueError("expected a fused qkv projection")
        self.qkv = qkv
        self.dim = qkv.in_features
        self.targets = tuple(targets)
        bad = set(self.targets) - {"q", "k", "v"}
        if bad:
            raise ValueError(f"unknown LoRA targets {bad}")
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        for t in self.targets:
            a = torch.empty(rank, self.dim)
            _kaiming_uniform(a, generator)
            self.lora_a[t] = nn.Parameter(a)
            self.lora_b[t] = nn.Parameter(torch.zeros(self.dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x)
        deltas = []
        for i, t in enumerate(("q", "k", "v")):
            if t in self.targets:
                deltas.append(x @ self.lora_a[t].T @ self.lora_b[t].T)
            else:
                deltas.append(torch.zeros_like(out[..., i * self.dim:(i + 1) * self.dim]))
        return out + torch.cat(deltas, dim=-1)


def _kaiming_uniform(t: torch.Tensor, generator: torch.Generator | None) -> None:
    # Same bound as nn.init.kaiming_uniform_(a=sqrt(5)) but with an explicit generator so the
    # seed controls LoRA init independently of any other RNG use (PROTOCOL §2).
    fan_in = t.shape[1]
    gain = math.sqrt(2.0 / (1 + 5.0))
    bound = gain * math.sqrt(3.0 / fan_in)
    with torch.no_grad():
        t.uniform_(-bound, bound, generator=generator)


def apply_lora(image_encoder: nn.Module, rank: int, targets: Iterable[str] = ("q", "v"),
               generator: torch.Generator | None = None) -> int:
    """Replace ``blocks[i].attn.qkv`` in every encoder block; returns the number of blocks."""
    n = 0
    for block in image_encoder.blocks:
        block.attn.qkv = LoRAQKV(block.attn.qkv, rank, targets, generator)
        n += 1
    return n


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts by group (PROTOCOL §5.5). ``n_trainable_other`` must be 0."""
    out = {"n_total": 0, "n_trainable": 0, "n_trainable_lora": 0,
           "n_trainable_decoder": 0, "n_trainable_other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        out["n_total"] += n
        if not p.requires_grad:
            continue
        out["n_trainable"] += n
        if ".lora_a." in name or ".lora_b." in name:
            out["n_trainable_lora"] += n
        elif name.startswith("mask_decoder."):
            out["n_trainable_decoder"] += n
        else:
            out["n_trainable_other"] += n
    return out
