"""Prompt-perturbation run (scripts/run/run_perturb.py, PROTOCOL §14) on fake data, random ViT-B, CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samrel.config import load_config
from samrel.data import discover, read_image
from samrel.prompts import build_perturbed_prompts, build_prompts

pytest.importorskip("segment_anything")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "run"))


@pytest.mark.slow
def test_decode_boxes_equals_predict_image(fake_pranet):
    import torch

    from samrel.infer import decode_boxes, make_predictor, predict_image
    from samrel.models import build_model

    torch.manual_seed(0)
    model = build_model(load_config("a0"), None).eval()
    pred = make_predictor(model)
    rec = next(r for r in discover(fake_pranet, load_config("base")["data"]) if r.dataset == "etis")
    img = read_image(rec.image_path)
    boxes = [[5, 5, 30, 30]]
    p1, i1 = predict_image(pred, img, boxes)
    p2, i2 = decode_boxes(pred, img.shape[:2], [[0, 0, 40, 40]])  # another box on the same embedding
    p3, i3 = decode_boxes(pred, img.shape[:2], boxes)
    assert np.array_equal(p1, p3) and np.array_equal(i1, i3)
    assert not np.array_equal(p1, p2)


@pytest.mark.slow
def test_run_perturb_a0_and_a3(fake_pranet, tmp_path):
    from run_perturb import main

    cfg = load_config("base")
    recs = discover(fake_pranet, cfg["data"])
    base = build_prompts(recs, 1, 0.1, 127, 8, "x", 50)
    (tmp_path / "prompts.json").write_text(json.dumps(base))
    levels = []
    for s in (0.1, 0.3):
        p = tmp_path / f"pp{round(s * 100):02d}.json"
        p.write_text(json.dumps(build_perturbed_prompts(base, s, 1, ["colondb", "etis"], "x")))
        levels.append(str(p))
    (tmp_path / "excl.json").write_text(json.dumps({"excluded": []}))
    out = tmp_path / "out"
    models = ["a0_s0_x", "a0_s1_x", "a0_s2_x"]  # stand-ins for the three A2 members
    rc = main(["--commit", "abcdef1234", "--stamp", "20261103-0000", "--data-root", str(fake_pranet),
               "--checkpoint", "random", "--out", str(out), "--search-root", str(tmp_path),
               "--models", ",".join(models), "--levels", ",".join(levels), "--a3-members", ",".join(models),
               "--device", "cpu", "--oracle", str(tmp_path / "prompts.json"),
               "--excluded", str(tmp_path / "excl.json"), "--limit-images", "1"])
    assert rc == 0
    pdir = out / "metrics" / "perturb"
    index = pd.read_csv(next(pdir.glob("runs_*.csv")))
    assert len(index) == 4 * 3 and set(index["level"]) == {"pp00", "pp10", "pp30"}
    assert "a3-pp30_s012_abcdef1_20261103-0000" in set(index["run_id"])
    df = pd.read_csv(pdir / "per_image" / "a0-pp10_s0_abcdef1_20261103-0000.csv")
    assert len(df) == 2 and set(df["dataset"]) == {"colondb", "etis"}
    # boxes saved with the prediction are the perturbed ones, not the oracle ones
    rid = "a0-pp30_s0_abcdef1_20261103-0000"
    saved = np.load(next((out / "predictions" / rid / "etis").glob("*.npz")))["boxes"].tolist()
    stem = next((out / "predictions" / rid / "etis").glob("*.npz")).stem
    assert saved == json.loads(Path(levels[1]).read_text())["images"][f"etis/{stem}"]["boxes"]


def test_check_reproduction_stops_on_mismatch(tmp_path):
    from run_perturb import check_reproduction

    ref = pd.DataFrame({"image_id": ["etis/1", "etis/2"], "dice": [0.9, 0.8], "iou": [0.8, 0.7],
                        "conf_iou_pred": [0.95, 0.9]})
    ref.to_csv(tmp_path / "ref.csv", index=False)
    assert check_reproduction(ref.iloc[::-1].copy(), tmp_path / "ref.csv", "ok")["max_abs_diff_dice"] == 0
    bad = ref.copy()
    bad.loc[0, "dice"] += 1e-3
    with pytest.raises(SystemExit):
        check_reproduction(bad, tmp_path / "ref.csv", "bad")
