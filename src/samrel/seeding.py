"""Pure seed derivation (PROTOCOL §2)."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(root: int, *labels: object) -> int:
    """Derive a seed in ``[0, 2**31)`` from a root seed and role labels.

    The same call gives the same seed on every machine and in any call order, because it only
    hashes ``repr((root, *labels))``.
    """
    digest = hashlib.sha256(repr((int(root), *labels)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def rng_for(root: int, *labels: object) -> np.random.Generator:
    """Numpy generator seeded with :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(root, *labels))
