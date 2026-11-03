"""PROTOCOL §7.4 aggregation on synthetic per-image tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from samrel.aggregate import paired, seed_stability, summarise_run
from samrel.constants import PER_IMAGE_COLUMNS


def fake_run(n=40, shift=0.0, seed=0, excluded=()):
    rng = np.random.default_rng(seed)
    ids = [f"etis/{i:03d}" for i in range(n)]
    dice = np.clip(rng.uniform(0.3, 1.0, n) + shift, 0, 1)
    df = pd.DataFrame({c: np.nan for c in PER_IMAGE_COLUMNS}, index=range(n))
    df["image_id"], df["dataset"], df["dice"], df["iou"] = ids, "etis", dice, dice / (2 - dice)
    df["failed"] = dice < 0.5
    df["conf_iou_pred"] = dice + rng.normal(0, 0.05, n)
    df["conf_mean_inside"] = rng.uniform(0.6, 1.0, n)
    df["excluded"] = [i in excluded for i in ids]
    stats = np.zeros((n, 3, 15, 3))
    stats[:, :, 14, 0] = 100
    stats[:, :, 14, 1] = 97
    stats[:, :, 14, 2] = 90
    return df, {"stats": stats, "image_ids": np.array(ids)}


def test_summary_values_and_ci():
    df, bins = fake_run()
    s = summarise_run(df, bins, 200, 0.95, 1).set_index("metric")
    assert s.loc["dice", "value"] == pytest.approx(df["dice"].mean())
    assert s.loc["dice", "ci_low"] <= s.loc["dice", "value"] <= s.loc["dice", "ci_high"]
    assert s.loc["ece_roi", "value"] == pytest.approx(0.07)
    assert "auroc_pairwise_dice" not in s.index   # only A3 has this score
    assert s.loc["n_failures", "value"] == df["failed"].sum()


def test_excluded_rows_dropped():
    df, bins = fake_run(excluded=("etis/000", "etis/001"))
    s = summarise_run(df, bins, 50, 0.95, 1)
    assert (s["n_images"] == 38).all()


def test_paired_shift_detected_and_identity_zero():
    a, ba = fake_run(shift=0.0)
    b, bb = fake_run(shift=0.0)
    p = paired(a, ba, b, bb, 200, 0.95, 1).set_index("metric")
    assert p.loc["dice", "diff"] == 0 and not p.loc["dice", "excludes_zero"]
    c, bc = fake_run(shift=0.1)
    p = paired(c, bc, a, ba, 200, 0.95, 1).set_index("metric")
    assert p.loc["dice", "diff"] > 0 and p.loc["dice", "excludes_zero"]


def test_paired_requires_same_images():
    a, ba = fake_run(n=40)
    b, bb = fake_run(n=39)
    with pytest.raises(ValueError):
        paired(a, ba, b, bb, 10, 0.95, 1)


def test_seed_stability():
    s = pd.DataFrame({"config": ["a2"] * 3 + ["a0"], "dataset": "etis", "metric": "dice",
                      "value": [0.8, 0.9, 1.0, 0.5], "run_ids": ["r0", "r1", "r2", "x"]})
    out = seed_stability(s)
    assert len(out) == 1 and out.loc[0, "mean"] == pytest.approx(0.9) and out.loc[0, "sd"] == pytest.approx(0.1)


def test_secondary_metrics_present_and_flagged():
    from samrel.aggregate import is_secondary

    df, bins = fake_run()
    s = summarise_run(df, bins, 100, 0.95, 1).set_index("metric")
    assert s.loc["n_failures_07", "value"] == (df["dice"] < 0.7).sum()
    assert -1 <= s.loc["spearman_iou_pred", "value"] <= 1 and s.loc["spearman_iou_pred", "value"] > 0.5
    assert "auroc07_iou_pred" in s.index
    assert is_secondary("auroc07_iou_pred") and is_secondary("n_failures_07")
    assert not is_secondary("spearman_mean_inside")  # main metric since PROTOCOL §7.7
    assert not is_secondary("auroc_iou_pred") and not is_secondary("n_failures") and not is_secondary("aurc_iou_pred")
