"""Perturbation tables (scripts/analysis/aggregate_perturb.py, PROTOCOL §14.5, §14.8) on synthetic runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from samrel.constants import PER_IMAGE_COLUMNS
from samrel.perturb import consistency, direction, reading

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "analysis"))


def test_reading_rules():
    assert direction("aurc_iou_pred") == -1 and direction("auroc07_iou_pred") == 1 and direction("dice") == 1
    assert consistency([1, 1, 1], [True] * 3, 3) == "consistent_+"
    assert consistency([1, 1, 0], [True] * 3, 3) == "not_consistent"
    assert consistency([1, 1, 1], [True, False, True], 3) == "insufficient_cases"
    assert consistency([-1], [True], 1) == "consistent_-"
    assert reading("consistent_+", "dice") == "less_sensitive"
    assert reading("consistent_+", "aurc_iou_pred") == "more_sensitive"
    assert reading("not_consistent", "dice") == "not_distinguishable"


def _write_run(pdir, run_id, config, seed, dice, conf):
    n = len(dice)
    ids = [f"{'colondb' if i < n // 2 else 'etis'}/{i}" for i in range(n)]
    df = pd.DataFrame({"run_id": run_id, "config": config, "seed": seed, "dataset": [i.split("/")[0] for i in ids],
                       "image_id": ids, "img_h": 10, "img_w": 10, "n_boxes": 1, "gt_area": 10, "pred_area": 10,
                       "dice": dice, "iou": dice * 0.9, "failed": dice < 0.5, "conf_iou_pred": conf,
                       "conf_mean_inside": conf, "conf_pairwise_dice": np.nan if config != "a3" else conf,
                       "excluded": False}, columns=list(PER_IMAGE_COLUMNS))
    (pdir / "per_image").mkdir(parents=True, exist_ok=True)
    (pdir / "ece_bins").mkdir(parents=True, exist_ok=True)
    df.to_csv(pdir / "per_image" / f"{run_id}.csv", index=False)
    stats = np.zeros((n, 3, 15, 3))
    stats[:, :, 14] = [10, 9.5, 9]
    np.savez(pdir / "ece_bins" / f"{run_id}.npz", stats=stats, image_ids=np.array(ids),
             regions=np.array(["img", "roi", "bnd"]), columns=np.array(["count", "sum_conf", "sum_correct"]))


def test_aggregate_perturb_tables(tmp_path):
    from aggregate_perturb import main

    rng = np.random.default_rng(0)
    pdir = tmp_path / "perturb"
    n = 120
    base_dice = rng.uniform(0.2, 1.0, n)
    index = []
    plan = [("a0", "0")] + [("a1", s) for s in "012"] + [("a2", s) for s in "012"] + [("a3", "012")]
    for k, lv in enumerate(("pp00", "pp10", "pp20", "pp30")):
        for c, s in plan:
            # A0 loses 0.1 Dice per level, the fine-tuned ones 0.02: fine-tuned are less sensitive by design
            drop = 0.1 * k if c == "a0" else 0.02 * k
            dice = np.clip(base_dice - drop + rng.normal(0, 0.005, n), 0, 1)
            rid = f"{c}-{lv}_s{s}_x_t"
            _write_run(pdir, rid, c, s, dice, dice + rng.normal(0, 0.05, n))
            index.append({"run_id": rid, "config": c, "seed": s, "level": lv, "sigma": 0.1 * k, "source_run": "-",
                          "commit": "x", "prompts_sha256": "-"})
    pd.DataFrame(index).to_csv(pdir / "runs_t_all.csv", index=False)
    assert main(["--perturb-dir", str(pdir), "--n-resamples", "200"]) == 0

    summ = pd.read_csv(pdir / "summary.csv", dtype={"seed": str})
    a = summ[(summ.metric == "auroc_iou_pred") & (summ.config == "a0")]
    assert set(a.columns) >= {"auroc_eligible", "n_fail_05", "n_success_05", "n_boot_undefined"}
    for r in a.itertuples():
        assert r.auroc_eligible == (min(r.n_fail_05, r.n_success_05) >= 20)
    dd = pd.read_csv(pdir / "did_consistency.csv", dtype={"seed": str})
    dice = dd[(dd.metric == "dice") & (dd.level == "pp30")]
    assert set(dice.reading) == {"less_sensitive"} and set(dice.config) == {"a1", "a2", "a3"}
    rc = pd.read_csv(pdir / "rq2_consistency.csv")
    assert set(rc.family) == {"a1_vs_a0", "a2_vs_a0", "a2_vs_a1", "a3_vs_a2"}
    assert rc[rc.level == "pp00"]["same_as_pp00"].all()
    # AUROC cells that are not eligible are never called consistent
    au = pd.read_csv(pdir / "rq2_pairs.csv")
    bad = au[au.metric.str.startswith("auroc") & ~au.eligible]
    keys = set(zip(bad.family, bad.level, bad.dataset, bad.metric))
    flagged = rc.set_index(["family", "level", "dataset", "metric"])["class"]
    assert all(flagged[k] == "insufficient_cases" for k in keys)
