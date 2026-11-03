"""PROTOCOL §6 training loop on a tiny SAM and the fake PraNet tree (CPU, seconds)."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
import torch

from samrel.config import load_config
from samrel.data import discover
from samrel.lora import count_parameters
from samrel.models import build_model
from samrel.prompts import build_prompts
from samrel.splits import make_val_split
from samrel.train import (SampleLoader, build_samples, cache_embeddings, component_masks, load_trainable,
                          lr_lambda, train, trainable_state)

pytest.importorskip("segment_anything")


def tiny_sam():
    from segment_anything.build_sam import _build_sam

    torch.manual_seed(0)
    return _build_sam(encoder_embed_dim=32, encoder_depth=2, encoder_num_heads=2,
                      encoder_global_attn_indexes=[1], checkpoint=None)


@pytest.fixture(scope="module")
def setup(fake_pranet):
    cfg = load_config("a2")
    cfg["train"].update(epochs=2, batch_size=2, grad_accum=2)
    recs = [r for r in discover(fake_pranet, cfg["data"]) if r.dataset.endswith("_train")]
    prompts = build_prompts(recs, cfg["root_seed"], 0.1, 127, 8, "x", 50)
    ids = {ds: [r.image_id for r in recs if r.dataset == ds] for ds in ("kvasir_train", "clinicdb_train")}
    split = make_val_split(ids, 0.34, cfg["root_seed"])
    return cfg, {r.image_id: r for r in recs}, prompts, split


def test_component_masks_match_boxes():
    m = np.zeros((20, 20), bool); m[2:5, 2:5] = True; m[10:15, 12:18] = True
    comps = component_masks(m, [[12, 10, 17, 14], [2, 2, 4, 4]], 8)
    assert comps[0].sum() == 30 and comps[1].sum() == 9


def test_lr_cosine_ends_at_zero():
    f = lr_lambda(100)
    assert f(0) == 1.0 and f(50) == pytest.approx(0.5) and f(100) == pytest.approx(0.0)


def test_samples_one_per_box(setup):
    cfg, recs, prompts, split = setup
    ids = list(recs)
    assert len(build_samples(ids, prompts)) == sum(len(prompts["images"][i]["boxes"]) for i in ids)


def test_targets_are_binary_low_res(setup):
    cfg, recs, prompts, _ = setup
    loader = SampleLoader(tiny_sam(), recs, prompts, cfg)
    i = sorted(recs)[0]
    t = loader.targets(i)[0]
    assert t.shape == (1, 256, 256) and set(t.unique().tolist()) <= {0.0, 1.0} and t.sum() > 0
    assert loader.image_tensor(i).shape == (3, 1024, 1024)


def test_train_a2_updates_only_lora_and_decoder_and_resumes(setup, tmp_path):
    cfg, recs, prompts, split = setup
    model = build_model(cfg, None, seed=0, sam=tiny_sam())
    frozen_before = {k: v.clone() for k, v in model.named_parameters() if not v.requires_grad}
    counts = count_parameters(model)
    assert counts["n_trainable_other"] == 0 and counts["n_trainable_lora"] > 0
    best = train(model, cfg, 0, recs, prompts, split, tmp_path, device="cpu", log=lambda *_: None)
    assert best["epoch"] in (0, 1)
    for k, v in model.named_parameters():
        if not v.requires_grad:
            assert torch.equal(v, frozen_before[k]), k
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    # Resume: a finished run does not train again and keeps its log.
    train(model, cfg, 0, recs, prompts, split, tmp_path, device="cpu", log=lambda *_: None)
    assert len(list(csv.DictReader(open(tmp_path / "train_log.csv")))) == 2
    state = torch.load(tmp_path / "best_trainable.pt", weights_only=True)
    fresh = build_model(cfg, None, seed=0, sam=tiny_sam())
    load_trainable(fresh, state)
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in state.items())


def test_same_seed_same_result_different_seed_differs(setup, tmp_path):
    cfg, recs, prompts, split = setup
    c = copy.deepcopy(cfg); c["train"]["epochs"] = 1
    outs = []
    for seed, d in ((0, "a"), (0, "b"), (1, "c")):
        m = build_model(c, None, seed=seed, sam=tiny_sam())
        train(m, c, seed, recs, prompts, split, tmp_path / d, device="cpu", log=lambda *_: None)
        outs.append(trainable_state(m))
    k = next(iter(outs[0]))
    assert all(torch.equal(outs[0][n], outs[1][n]) for n in outs[0])
    assert not all(torch.equal(outs[0][n], outs[2][n]) for n in outs[0])


def test_a1_with_embedding_cache(setup, tmp_path):
    cfg, recs, prompts, split = setup
    c = copy.deepcopy(load_config("a1")); c["train"].update(epochs=1, batch_size=2, grad_accum=2)
    model = build_model(c, None, seed=0, sam=tiny_sam())
    loader = SampleLoader(model, recs, prompts, c)
    cache = cache_embeddings(model, recs, loader, device="cpu")
    assert next(iter(cache.values())).dtype == torch.float16
    enc_before = {k: v.clone() for k, v in model.image_encoder.named_parameters()}
    train(model, c, 0, recs, prompts, split, tmp_path, device="cpu", embed_cache=cache, log=lambda *_: None)
    assert all(torch.equal(v, enc_before[k]) for k, v in model.image_encoder.named_parameters())
