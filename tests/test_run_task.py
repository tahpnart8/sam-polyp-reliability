"""End-to-end A0 inference through scripts/run/run_task.py on fake data (random ViT-B, CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from samrel.config import load_config
from samrel.data import discover
from samrel.prompts import build_prompts

pytest.importorskip("segment_anything")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "run"))


@pytest.mark.slow
def test_run_task_infer_a0(fake_pranet, tmp_path):
    from run_task import main

    cfg = load_config("a0")
    recs = discover(fake_pranet, cfg["data"])
    (tmp_path / "prompts.json").write_text(json.dumps(build_prompts(recs, 1, 0.1, 127, 8, "x", 50)))
    (tmp_path / "split.json").write_text(json.dumps({"train": {}, "val": {}}))
    (tmp_path / "excl.json").write_text(json.dumps({"excluded": []}))
    out = tmp_path / "out"
    rc = main(["--task", "infer", "--config", "a0", "--seed", "0", "--run-id", "a0_s0_test", "--commit", "x",
               "--data-root", str(fake_pranet), "--checkpoint", "random", "--out", str(out),
               "--work", str(tmp_path / "w"), "--device", "cpu", "--limit-images", "1",
               "--prompts", str(tmp_path / "prompts.json"), "--split", str(tmp_path / "split.json"),
               "--excluded", str(tmp_path / "excl.json")])
    assert rc == 0
    df = pd.read_csv(out / "metrics" / "per_image" / "a0_s0_test.csv")
    assert len(df) == 5 and set(df["dataset"]) == {"kvasir", "clinicdb", "colondb", "etis", "cvc300"}
    cost = pd.read_csv(out / "metrics" / "cost" / "a0_s0_test.csv")
    assert cost.loc[0, "phase"] == "infer" and cost.loc[0, "n_images"] == 5
