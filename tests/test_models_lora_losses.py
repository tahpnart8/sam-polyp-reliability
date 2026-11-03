"""PROTOCOL §5.3, §5.5 (LoRA, parameter counts) and §6 (loss)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from samrel.config import load_config
from samrel.lora import LoRAQKV, apply_lora, count_parameters
from samrel.losses import focal_loss, sam_loss


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attn(dim)


class _Encoder(nn.Module):
    def __init__(self, dim=768, depth=12):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(depth))


def test_lora_qv_param_count_vit_b_geometry():
    enc = _Encoder()
    for p in enc.parameters():
        p.requires_grad_(False)
    apply_lora(enc, rank=4, targets=("q", "v"))
    n = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert n == 12 * 2 * (768 * 4 + 4 * 768) == 147_456


def test_lora_starts_identical_and_only_touches_q_v():
    torch.manual_seed(0)
    lin = nn.Linear(16, 48)
    ref = nn.Linear(16, 48); ref.load_state_dict(lin.state_dict())
    lora = LoRAQKV(lin, 2, ("q", "v"), torch.Generator().manual_seed(1))
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), ref(x))
    with torch.no_grad():
        for t in ("q", "v"):
            lora.lora_b[t].fill_(0.1)
    diff = (lora(x) - ref(x)).abs().sum(0)
    assert diff[16:32].max() == 0 and diff[:16].min() > 0 and diff[32:].min() > 0


def test_lora_init_depends_on_generator_seed():
    a = LoRAQKV(nn.Linear(8, 24), 2, generator=torch.Generator().manual_seed(1))
    b = LoRAQKV(nn.Linear(8, 24), 2, generator=torch.Generator().manual_seed(1))
    c = LoRAQKV(nn.Linear(8, 24), 2, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.lora_a["q"], b.lora_a["q"]) and not torch.equal(a.lora_a["q"], c.lora_a["q"])


def test_focal_reduces_to_bce_with_gamma_zero():
    logits, target = torch.randn(2, 1, 4, 4), (torch.rand(2, 1, 4, 4) > 0.5).float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(focal_loss(logits, target, gamma=0.0), bce)


def test_sam_loss_weights_and_grad():
    w = load_config("a2")["train"]["loss"]
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    iou = torch.rand(2, 1, requires_grad=True)
    out = sam_loss(logits, (torch.rand(2, 1, 8, 8) > 0.5).float(), iou, w)
    expected = 20 * out["focal"] + 1 * out["dice"] + 1 * out["iou_mse"]
    assert torch.allclose(out["loss"], expected)
    out["loss"].backward()
    assert logits.grad is not None and iou.grad is not None


@pytest.mark.slow
@pytest.mark.parametrize("name", ["a0", "a1", "a2"])
def test_real_sam_vit_b_trainable_parts(name):
    """Counts on the real segment-anything ViT-B architecture (random weights, no checkpoint)."""
    pytest.importorskip("segment_anything")
    from samrel.models import build_model

    counts = count_parameters(build_model(load_config(name), checkpoint=None, seed=0))
    assert counts["n_trainable_other"] == 0
    if name == "a0":
        assert counts["n_trainable"] == 0
    if name == "a1":
        assert counts["n_trainable_lora"] == 0 and counts["n_trainable_decoder"] > 0
    if name == "a2":
        assert counts["n_trainable_lora"] == 147_456 and counts["n_trainable_decoder"] > 0
    print(name, counts)
