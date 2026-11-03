"""PROTOCOL §7.1-§7.3, checked against hand values and a second implementation."""

from __future__ import annotations

import numpy as np
import pytest

from samrel import metrics as M

CFG = {"ece_bins": 15, "ece_range": [0.0, 1.0], "roi_expand_per_side": 0.10, "boundary_frac_diag": 0.01}


def test_dice_iou_hand_values():
    gt = np.zeros((4, 4), bool); gt[:2, :2] = True           # 4 px
    pred = np.zeros((4, 4), bool); pred[:2, :3] = True       # 6 px, 4 overlap
    d, j = M.dice_iou(pred, gt)
    assert d == pytest.approx(8 / 10) and j == pytest.approx(4 / 6)
    assert M.dice_iou(np.zeros_like(gt), gt) == (0.0, 0.0)   # empty prediction on a polyp image


def test_mean_inside_confidence_and_empty():
    p = np.array([[0.9, 0.6], [0.2, 0.1]])
    assert M.mean_inside_confidence(p) == pytest.approx((0.9 + 0.6) / 2)
    assert M.mean_inside_confidence(np.full((2, 2), 0.1), empty_value=0.0) == 0.0


def test_ece_bins_match_direct_implementation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(size=(30, 40)) ** rng.uniform(0.3, 3)
        y = rng.uniform(size=(30, 40)) < p
        full = np.ones_like(y)
        stats = M.ece_bin_stats(p, y, full)
        assert M.ece_from_bins(stats) == pytest.approx(M.ece_direct(p, y), abs=1e-12)
        # half-range binning must agree with the direct version too (Q-B7 alternative)
        stats2 = M.ece_bin_stats(p, y, full, 15, 0.5, 1.0)
        assert M.ece_from_bins(stats2) == pytest.approx(M.ece_direct(p, y, 15, 0.5, 1.0), abs=1e-12)


def test_ece_perfectly_calibrated_hard_predictions_is_zero():
    y = np.zeros((10, 10), bool); y[2:5, 2:5] = True
    p = y.astype(float)  # confident and correct
    assert M.ece_from_bins(M.ece_bin_stats(p, y, np.ones_like(y))) == pytest.approx(0.0)


def test_pooled_ece_equals_ece_of_concatenated_pixels():
    rng = np.random.default_rng(1)
    ps = [rng.uniform(size=(8, 9)) for _ in range(3)]
    ys = [rng.uniform(size=(8, 9)) < 0.3 for _ in range(3)]
    pooled = np.stack([M.ece_bin_stats(p, y, np.ones_like(y)) for p, y in zip(ps, ys)])
    concat = M.ece_direct(np.concatenate([p.ravel() for p in ps]), np.concatenate([y.ravel() for y in ys]))
    assert M.ece_from_bins(pooled) == pytest.approx(concat, abs=1e-12)


def test_roi_expands_ten_percent_per_side_and_clips():
    roi = M.roi_mask((100, 100), [[40, 40, 59, 59]], 0.10)   # 20x20 box -> +2 px each side
    ys, xs = np.where(roi)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (38, 61, 38, 61)
    roi = M.roi_mask((50, 50), [[0, 0, 9, 9]], 0.10)
    assert roi[0, 0] and roi.sum() == 11 * 11


def test_boundary_band_symmetric_and_thin():
    gt = np.zeros((200, 200), bool); gt[50:150, 50:150] = True
    band = M.boundary_mask(gt, 0.01)                          # r = 2.83 px
    assert band[50, 100] and band[48, 100] and not band[100, 100] and not band[44, 100]
    # inside: column 51 is 2 px from the background column 49, column 52 is 3 px
    assert band[100, 51] and not band[100, 52]


def test_image_bin_stats_shape():
    gt = np.zeros((20, 30), bool); gt[5:10, 5:12] = True
    s = M.image_bin_stats(np.full((20, 30), 0.3), gt, [[5, 5, 11, 9]], CFG)
    assert s.shape == (3, 15, 3)
    assert s[0, :, 0].sum() == 600 and s[1, :, 0].sum() < 600


def test_auroc_failure_direction_and_nan():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    failed = np.array([False, False, True, True])
    assert M.auroc_failure(conf, failed) == 1.0              # low confidence flags failures
    assert np.isnan(M.auroc_failure(conf, np.zeros(4, bool)))


def test_aurc_hand_value_and_tie_break():
    dice = np.array([1.0, 0.0, 0.5])
    conf = np.array([0.9, 0.1, 0.5])
    # order by conf: 1.0, 0.5, 0.0 -> losses 0, .5, 1 -> risks 0, .25, .5 -> mean .25
    assert M.aurc(conf, dice, ["a", "b", "c"]) == pytest.approx(0.25)
    tied = np.array([0.5, 0.5])
    assert M.aurc(tied, np.array([0.0, 1.0]), ["a", "b"]) == pytest.approx((1 + 0.5) / 2)
    assert M.aurc(tied, np.array([0.0, 1.0]), ["b", "a"]) == pytest.approx((0 + 0.5) / 2)


def test_aurc_oracle_is_lower_bound():
    rng = np.random.default_rng(2)
    dice = rng.uniform(size=50)
    ids = [f"{i:03d}" for i in range(50)]
    oracle = M.aurc(dice, dice, ids)
    assert oracle <= M.aurc(rng.uniform(size=50), dice, ids) + 1e-12
