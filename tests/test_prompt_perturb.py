"""Perturbed box prompts (PROTOCOL §14.1, §14.2)."""

from __future__ import annotations

import json

import numpy as np

from samrel.config import load_config
from samrel.data import data_hash, discover
from samrel.prompts import box_iou, build_perturbed_prompts, build_prompts, perturb_box


def test_sigma_zero_returns_oracle_box():
    for u in (np.zeros(4), np.array([1.0, -1.0, 0.5, -0.5])):
        assert perturb_box([10, 20, 39, 59], 100, 100, u, 0.0) == ([10, 20, 39, 59], False, False)


def test_shift_and_scale_follow_box_size():
    # box 30 wide, 40 high; centre shift 0.1*30 = 3 px right, width x1.1 = 33, height unchanged
    box, clipped, degenerate = perturb_box([10, 20, 39, 59], 200, 200, np.array([1.0, 0.0, 1.0, 0.0]), 0.1)
    assert box == [12, 20, 44, 59] and not clipped and not degenerate
    # shrink by 30% on both axes keeps the centre: 30 -> 21 wide, 40 -> 28 high
    box, _, _ = perturb_box([10, 20, 39, 59], 200, 200, np.array([0.0, 0.0, -1.0, -1.0]), 0.3)
    assert box == [15, 26, 35, 53]


def test_clip_to_image_is_flagged():
    box, clipped, degenerate = perturb_box([0, 0, 9, 9], 10, 10, np.array([-1.0, -1.0, 1.0, 1.0]), 0.3)
    assert box[0] == 0 and box[1] == 0 and box[2] <= 9 and box[3] <= 9
    assert clipped and not degenerate


def test_box_iou():
    assert box_iou([0, 0, 9, 9], [0, 0, 9, 9]) == 1.0
    assert box_iou([0, 0, 9, 9], [10, 10, 19, 19]) == 0.0
    assert abs(box_iou([0, 0, 9, 9], [5, 0, 14, 9]) - 50 / 150) < 1e-12


def _base(fake_pranet):
    cfg = load_config("base")
    d = cfg["data"]
    recs = discover(fake_pranet, d)
    return build_prompts(recs, cfg["root_seed"], 0.10, d["mask_threshold"], d["connectivity"], data_hash(recs), 0)


def test_perturbed_prompts_shared_direction_and_fixed(fake_pranet):
    base = _base(fake_pranet)
    lv = {s: build_perturbed_prompts(base, s, 20261029, ["colondb", "etis"], "x") for s in (0.1, 0.2, 0.3)}
    ids = list(lv[0.1]["images"])
    assert ids and all(i.split("/")[0] in ("colondb", "etis") for i in ids)
    for i in ids:
        e = [lv[s]["images"][i] for s in (0.1, 0.2, 0.3)]
        assert all(x["tight_boxes"] == base["images"][i]["tight_boxes"] for x in e)
        assert all(x["oracle_boxes"] == base["images"][i]["boxes"] for x in e)
        # IoU with the oracle box shrinks (weakly) as sigma grows, since u is shared
        iou = [box_iou(x["boxes"][0], x["oracle_boxes"][0]) for x in e]
        assert iou[0] >= iou[2] - 1e-9
    again = build_perturbed_prompts(base, 0.2, 20261029, ["colondb", "etis"], "x")
    assert json.dumps(again, sort_keys=True) == json.dumps(lv[0.2], sort_keys=True)
    assert lv[0.2]["perturb"]["sigma"] == 0.2


def test_perturbed_prompts_do_not_depend_on_other_images(fake_pranet):
    base = _base(fake_pranet)
    full = build_perturbed_prompts(base, 0.3, 20261029, ["colondb", "etis"], "x")
    only_etis = build_perturbed_prompts(base, 0.3, 20261029, ["etis"], "x")
    for i, e in only_etis["images"].items():
        assert e["boxes"] == full["images"][i]["boxes"]
