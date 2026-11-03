"""A3 ensemble task of scripts/run/run_post.py on fake saved predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.config import load_config
from samrel.data import discover, read_mask
from samrel.infer import save_prediction
from samrel.prompts import build_prompts

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "run"))


def test_ensemble_task(fake_pranet, tmp_path):
    from run_post import main

    cfg = load_config("base")
    recs = discover(fake_pranet, cfg["data"])
    prompts = build_prompts(recs, 1, 0.1, 127, 8, "x", 50)
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    (tmp_path / "excl.json").write_text(json.dumps({"excluded": []}))
    runs = ["a2_s0_x", "a2_s1_x", "a2_s2_x"]
    for k, rid in enumerate(runs):
        for r in recs:
            if r.dataset.endswith("_train"):
                continue
            gt = read_mask(r.mask_path, 127)
            p = np.where(gt, 0.9 - 0.1 * k, 0.1)
            e = prompts["images"][r.image_id]
            save_prediction(tmp_path / "in" / f"kernel{k}" / "out" / "predictions" / rid / r.dataset /
                            f"{r.image_id.split('/')[1]}.npz", p, np.full(len(e["boxes"]), 0.8), e["boxes"])
    out = tmp_path / "out"
    assert main(["--task", "ensemble", "--run-id", "a3_s012_x", "--commit", "x", "--data-root", str(fake_pranet),
                 "--out", str(out), "--search-root", str(tmp_path / "in"), "--a2-runs", ",".join(runs),
                 "--prompts", str(tmp_path / "prompts.json"), "--excluded", str(tmp_path / "excl.json")]) == 0
    df = pd.read_csv(out / "metrics" / "per_image" / "a3_s012_x.csv")
    assert len(df) == 30 and (df["config"] == "a3").all() and df["dice"].min() == 1.0
    assert (df["conf_pairwise_dice"] == 1.0).all()
    assert (df["conf_iou_pred"].round(6) == 0.8).all()
