"""SAM pre-training loss: 20 focal + 1 Dice on masks, plus MSE on the IoU head (PROTOCOL §6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               alpha: float | None = None) -> torch.Tensor:
    """Mean binary focal loss over pixels."""
    target = target.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - p_t) ** gamma
    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice per mask, averaged over the batch."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.float().flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def mask_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Hard IoU of ``logits > 0`` against the target, the regression target of the IoU head."""
    pred = (logits > 0).flatten(1)
    t = target.bool().flatten(1)
    inter = (pred & t).sum(1).float()
    union = (pred | t).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def sam_loss(mask_logits: torch.Tensor, gt: torch.Tensor, iou_pred: torch.Tensor,
             weights: dict) -> dict[str, torch.Tensor]:
    """Total loss and its parts. ``mask_logits``, ``gt``: ``[B,1,H,W]``; ``iou_pred``: ``[B,1]``."""
    focal = focal_loss(mask_logits, gt, weights.get("focal_gamma", 2.0), weights.get("focal_alpha"))
    dice = soft_dice_loss(mask_logits, gt)
    iou_target = mask_iou(mask_logits.detach(), gt).unsqueeze(1)
    iou_mse = F.mse_loss(iou_pred.float(), iou_target)
    total = (weights["focal_weight"] * focal + weights["dice_weight"] * dice
             + weights["iou_mse_weight"] * iou_mse)
    return {"loss": total, "focal": focal.detach(), "dice": dice.detach(), "iou_mse": iou_mse.detach()}
