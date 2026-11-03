"""PROTOCOL §1 (data, audit), §3 (prompts), §4 (split) on the fake PraNet tree."""

from __future__ import annotations

import json

import numpy as np
import pytest

from samrel.audit import audit_dataset, hamming_matrix
from samrel.config import load_config
from samrel.data import classify_train_source, data_hash, discover
from samrel.paths import data_root
from samrel.prompts import boxes_from_mask, build_prompts, jitter_box
from samrel.seeding import rng_for
from samrel.splits import make_val_split

CFG = load_config("base")


def test_classify_train_source():
    assert classify_train_source("123") == "clinicdb_train"
    assert classify_train_source("cju0qkwl35piu0993l0dewei2") == "kvasir_train"


def test_discover_fake_tree(fake_pranet):
    recs = discover(fake_pranet, CFG["data"])
    by = {}
    for r in recs:
        by.setdefault(r.dataset, []).append(r)
    assert set(by) == {"kvasir_train", "clinicdb_train", "kvasir", "clinicdb", "colondb", "etis", "cvc300"}
    assert all(len(v) == 6 for v in by.values())
    assert len({r.image_id for r in recs}) == len(recs)


def test_boxes_one_per_component_and_jitter_bounds():
    mask = np.zeros((50, 80), bool)
    mask[10:20, 5:15] = True
    mask[30:40, 50:70] = True
    tight, boxes = boxes_from_mask(mask, np.random.default_rng(0), 0.10)
    assert tight == [[5, 10, 14, 19], [50, 30, 69, 39]]
    for (x0, y0, x1, y1), (a0, b0, a1, b1) in zip(tight, boxes):
        w, h = x1 - x0 + 1, y1 - y0 + 1
        assert x0 - 0.1 * w - 0.5 <= a0 <= x0 and y0 - 0.1 * h - 0.5 <= b0 <= y0
        assert x1 <= a1 <= x1 + 0.1 * w + 0.5 and y1 <= b1 <= y1 + 0.1 * h + 0.5


def test_speckle_components_get_no_box():
    """Q-B2: components under 50 px (annotation speckle) give no prompt box."""
    mask = np.zeros((60, 60), bool)
    mask[10:30, 10:30] = True          # 400 px polyp
    mask[50, 50] = mask[55, 2] = True  # 1 px specks
    assert len(boxes_from_mask(mask, np.random.default_rng(0), 0.1, 8, 0)[0]) == 3
    assert boxes_from_mask(mask, np.random.default_rng(0), 0.1, 8, 50)[0] == [[10, 10, 29, 29]]


def test_jitter_clips_to_image():
    assert jitter_box([0, 0, 9, 9], 10, 10, np.random.default_rng(0), 0.10) == [0, 0, 9, 9]


def test_eight_connectivity_merges_diagonal_touch():
    mask = np.zeros((6, 6), bool); mask[1, 1] = mask[2, 2] = True
    assert len(boxes_from_mask(mask, np.random.default_rng(0), 0.1, 8)[0]) == 1
    assert len(boxes_from_mask(mask, np.random.default_rng(0), 0.1, 4)[0]) == 2


def test_prompts_reproducible_and_per_image_independent(fake_pranet):
    recs = discover(fake_pranet, CFG["data"])
    kw = dict(root_seed=CFG["root_seed"], jitter_max=0.1, mask_threshold=127, connectivity=8, data_hash="x")
    a = build_prompts(recs, **kw)
    b = build_prompts(list(reversed(recs)), **kw)
    c = build_prompts(recs[::2], **kw)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for k, v in c["images"].items():
        assert a["images"][k] == v
    assert len(a["images"]) == len(recs)
    # the seed of an image depends only on the root seed and its id
    assert rng_for(CFG["root_seed"], "prompts", "etis/1").uniform() == rng_for(CFG["root_seed"], "prompts", "etis/1").uniform()


def test_val_split_stratified_and_disjoint():
    ids = {"kvasir_train": [f"kvasir_train/{i}" for i in range(900)],
           "clinicdb_train": [f"clinicdb_train/{i}" for i in range(550)]}
    s = make_val_split(ids, 0.10, CFG["root_seed"])
    assert len(s["val"]["kvasir_train"]) == 90 and len(s["val"]["clinicdb_train"]) == 55
    for src in ids:
        assert not set(s["val"][src]) & set(s["train"][src])
        assert sorted(s["val"][src] + s["train"][src]) == sorted(ids[src])
    assert s == make_val_split(ids, 0.10, CFG["root_seed"])


def test_hamming_matrix():
    assert hamming_matrix(["0" * 16], ["f" * 16, "0" * 15 + "1"]).tolist() == [[64, 1]]


def test_audit_on_fake_tree(fake_pranet):
    recs = discover(fake_pranet, CFG["data"])
    res = audit_dataset(recs, CFG)
    s = res["summary"].set_index("dataset")
    assert (s["n_images"] == 6).all() and not s["count_ok"].any()     # mock counts are not PraNet's
    assert (s["n_size_mismatch"] == 0).all() and (s["n_mask_nonbinary"] == 0).all()
    assert s.loc["kvasir_train", "n_multi_polyp"] == 3
    assert s.loc["etis", "size_mode_hw"] == "48x64"
    assert set(res["excluded"]) <= {i for i in res["per_image"]["image_id"] if i.split("/")[0] in ("colondb", "etis")}
    assert len(data_hash(recs)) == 64


def test_audit_flags_duplicate_of_train_image(tmp_path):
    from samrel.mock import make_fake_pranet
    import shutil

    root = make_fake_pranet(tmp_path / "p", n_per_set=3)
    src = root / "TrainDataset"
    dst = root / "TestDataset" / "ETIS-LaribPolypDB"
    shutil.copy(src / "image" / "1.png", dst / "images" / "dup.png")
    shutil.copy(src / "masks" / "1.png", dst / "masks" / "dup.png")
    res = audit_dataset(discover(root, CFG["data"]), CFG)
    assert "etis/dup" in res["excluded"]
    assert res["summary"].set_index("dataset").loc["etis", "n_excluded"] == 1


@pytest.mark.realdata
def test_real_counts_match_contract():
    root = data_root()
    if root is None:
        pytest.skip("SAMREL_DATA_ROOT not set")
    from samrel.constants import DATASETS

    recs = discover(root, CFG["data"])
    for ds, (_, n) in DATASETS.items():
        assert sum(r.dataset == ds and r.image_path is not None for r in recs) == n, ds
