"""PROTOCOL §5.4 (A3) and §7.4 (bootstrap)."""

from __future__ import annotations

import numpy as np
import pytest

from samrel.bootstrap import ci_percentile, indices_for_dataset, paired_difference
from samrel.ensemble import combine_probs, pairwise_dice


def test_indices_deterministic_and_dataset_specific():
    a = indices_for_dataset("etis", 196, 1000, 20261029)
    assert a.shape == (1000, 196)
    assert np.array_equal(a, indices_for_dataset("etis", 196, 1000, 20261029))
    assert not np.array_equal(a, indices_for_dataset("colondb", 196, 1000, 20261029))


def test_paired_difference_identical_configs_is_zero():
    x = np.random.default_rng(0).uniform(size=100)
    idx = indices_for_dataset("kvasir", 100, 200, 1)
    res = paired_difference(lambda r: x[r].mean(), lambda r: x[r].mean(), idx)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0 and not res["excludes_zero"]


def test_paired_detects_constant_shift():
    x = np.random.default_rng(0).uniform(size=100)
    idx = indices_for_dataset("kvasir", 100, 500, 1)
    res = paired_difference(lambda r: (x[r] + 0.05).mean(), lambda r: x[r].mean(), idx)
    assert res["diff"] == pytest.approx(0.05) and res["excludes_zero"]


def test_ci_ignores_nan():
    lo, hi = ci_percentile(np.array([np.nan, 1.0, 2.0, 3.0]), 0.5)
    assert 1.0 <= lo <= hi <= 3.0


def test_combine_probs_mean_and_shape_check():
    a, b, c = (np.full((2, 3), v, np.float16) for v in (0.2, 0.4, 0.9))
    assert np.allclose(combine_probs([a, b, c]), 0.5, atol=1e-3)
    with pytest.raises(ValueError):
        combine_probs([a, np.zeros((3, 2))])


def test_pairwise_dice():
    m1 = np.zeros((4, 4), bool); m1[:2] = True
    m2 = m1.copy()
    m3 = np.zeros((4, 4), bool)
    assert pairwise_dice([m1, m2, m3]) == pytest.approx((1 + 0 + 0) / 3)
    assert pairwise_dice([m3, m3, m3], empty_pair_value=1.0) == 1.0
