"""A3: mean of member probabilities and pairwise agreement (PROTOCOL §5.4, §7.1)."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from .metrics import dice_iou


def combine_probs(probs: Sequence[np.ndarray]) -> np.ndarray:
    """Pixel-wise mean of post-sigmoid probabilities (float32 accumulation)."""
    if len(probs) < 2:
        raise ValueError("an ensemble needs at least two members")
    shapes = {p.shape for p in probs}
    if len(shapes) != 1:
        raise ValueError(f"member shapes differ: {shapes}")
    return np.mean(np.stack([p.astype(np.float32) for p in probs]), axis=0)


def pairwise_dice(masks: Sequence[np.ndarray], empty_pair_value: float = 1.0) -> float:
    """Mean Dice over all member pairs; a pair of empty masks counts as ``empty_pair_value``."""
    vals = []
    for a, b in combinations(masks, 2):
        if not a.any() and not b.any():
            vals.append(float(empty_pair_value))
        else:
            vals.append(dice_iou(a, b)[0])
    return float(np.mean(vals))
