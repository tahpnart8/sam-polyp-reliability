"""PROTOCOL §2 (seeds) and configs/*.yaml."""

from __future__ import annotations

import copy

from samrel.config import config_sha256, load_config
from samrel.constants import ROOT_SEED
from samrel.seeding import derive_seed


def test_derive_seed_pure_and_label_sensitive():
    assert derive_seed(0, "train", "shuffle") == derive_seed(0, "train", "shuffle")
    assert derive_seed(0, "train", "shuffle") != derive_seed(1, "train", "shuffle")
    assert derive_seed(0, "train", "shuffle") != derive_seed(0, "train", "lora_init")
    assert 0 <= derive_seed(ROOT_SEED, "prompts", "etis/1") < 2**31


def test_derive_seed_frozen_value():
    # Freezes the derivation itself: changing it would silently change prompts and splits.
    assert derive_seed(0, "x") == 947497715
    assert derive_seed(20261029, "val_split", "kvasir_train") == 410999045


def test_configs_differ_only_in_model_and_train():
    base = load_config("base")
    for name in ("a0", "a1", "a2"):
        cfg = load_config(name)
        assert cfg["name"] == name
        for key in base:
            if key not in ("model", "train"):
                assert cfg[key] == base[key], (name, key)
    a1, a2 = load_config("a1"), load_config("a2")
    # Fair comparison: A1 and A2 share every training hyper-parameter except caching.
    t1 = {k: v for k, v in a1["train"].items() if k != "cache_embeddings"}
    t2 = {k: v for k, v in a2["train"].items() if k != "cache_embeddings"}
    assert t1 == t2
    assert a1["train"]["batch_size"] * a1["train"]["grad_accum"] == 4
    assert not a1["model"]["lora"]["enabled"] and a2["model"]["lora"]["enabled"]
    assert a2["model"]["lora"]["rank"] == 4 and a2["model"]["lora"]["targets"] == ["q", "v"]
    assert not load_config("a0")["train"]["enabled"]


def test_config_hash_order_independent():
    cfg = load_config("a2")
    shuffled = dict(reversed(list(copy.deepcopy(cfg).items())))
    assert config_sha256(cfg) == config_sha256(shuffled)
    assert config_sha256(cfg) != config_sha256(load_config("a1"))
