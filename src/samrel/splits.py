"""Validation split, stratified by source (PROTOCOL §4)."""

from __future__ import annotations

from typing import Any

from .seeding import rng_for


def make_val_split(ids_by_source: dict[str, list[str]], fraction: float, root_seed: int) -> dict[str, Any]:
    """Hold out ``round(fraction * n)`` images of each source as validation."""
    train, val = {}, {}
    for source in sorted(ids_by_source):
        ids = sorted(ids_by_source[source])
        n_val = int(round(fraction * len(ids)))
        order = rng_for(root_seed, "val_split", source).permutation(len(ids))
        chosen = {ids[i] for i in order[:n_val]}
        val[source] = [i for i in ids if i in chosen]
        train[source] = [i for i in ids if i not in chosen]
    return {"version": 1, "root_seed": root_seed, "fraction": fraction, "train": train, "val": val}
