"""PROTOCOL §5.1 (inference), §7-§8 (per-image metrics, ECE bins) on fake data."""

from __future__ import annotations

import numpy as np
import pytest

from samrel.config import load_config
from samrel.data import discover, read_mask
from samrel.evaluate import evaluate_run, write_run_metrics
from samrel.infer import load_prediction, save_prediction
from samrel.metrics import ece_from_bins
from samrel.prompts import build_prompts

CFG = load_config("base")


def _setup(fake_pranet, tmp_path, members):
    recs = [r for r in discover(fake_pranet, CFG["data"]) if r.dataset == "etis"]
    prompts = build_prompts(recs, CFG["root_seed"], 0.1, 127, 8, "x")
    dirs = []
    for k, quality in enumerate(members):
        d = tmp_path / f"m{k}"
        for r in recs:
            gt = read_mask(r.mask_path, 127)
            noise = np.random.default_rng(k).uniform(0, 0.3, gt.shape)
            prob = np.clip(np.where(gt, quality, 1 - quality) + noise - 0.15, 0, 1)
            entry = prompts["images"][r.image_id]
            save_prediction(d / r.dataset / f"{r.image_id.split('/')[1]}.npz", prob,
                            np.full(len(entry["boxes"]), 0.9), entry["boxes"])
        dirs.append(d)
    return recs, prompts, dirs


def test_prediction_roundtrip(tmp_path):
    p = np.random.default_rng(0).uniform(size=(5, 7)).astype(np.float32)
    save_prediction(tmp_path / "x.npz", p, np.array([0.5]), [[0, 0, 3, 3]])
    z = load_prediction(tmp_path / "x.npz")
    assert z["prob"].dtype == np.float16 and np.allclose(z["prob"], p, atol=1e-3)
    assert z["boxes"].tolist() == [[0, 0, 3, 3]]


def test_evaluate_single_member(fake_pranet, tmp_path):
    recs, prompts, dirs = _setup(fake_pranet, tmp_path, [0.95])
    df, bins = evaluate_run("a0_s0_x", "a0", 0, dirs, recs, prompts, CFG, excluded=["etis/0"])
    assert len(df) == len(recs) and df["dice"].min() > 0.9 and not df["failed"].any()
    assert df["conf_pairwise_dice"].isna().all()
    assert df.set_index("image_id").loc["etis/0", "excluded"]
    assert bins["stats"].shape == (len(recs), 3, 15, 3)
    assert ece_from_bins(bins["stats"][:, 0]) >= 0
    out = tmp_path / "metrics"
    write_run_metrics(df, bins, "a0_s0_x", out_dir=out)
    assert (out / "per_image" / "a0_s0_x.csv").exists() and (out / "ece_bins" / "a0_s0_x.npz").exists()


def test_evaluate_ensemble_uses_mean_prob_and_pairwise(fake_pranet, tmp_path):
    recs, prompts, dirs = _setup(fake_pranet, tmp_path, [0.95, 0.9, 0.85])
    df, _ = evaluate_run("a3_s012_x", "a3", "012", dirs, recs, prompts, CFG)
    assert df["conf_pairwise_dice"].between(0, 1).all()
    assert (df["seed"] == "012").all()


@pytest.mark.slow
def test_predict_image_on_random_sam():
    """Shapes and ranges on the real architecture (random weights, CPU, one small image)."""
    pytest.importorskip("segment_anything")
    from samrel.infer import make_predictor, predict_image
    from samrel.models import build_model

    model = build_model(load_config("a0"), checkpoint=None).eval()
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    prob, iou = predict_image(make_predictor(model), img, [[5, 5, 30, 30], [40, 10, 70, 50]])
    assert prob.shape == (60, 80) and prob.dtype == np.float32
    assert 0 <= prob.min() and prob.max() <= 1 and iou.shape == (2,)
