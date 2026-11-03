"""PROTOCOL §9 (run_id, manifest, environment) and the Kaggle job templates."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone

import pytest

from samrel.manifest import environment, make_run_id, new_manifest


def test_run_id_format():
    when = datetime(2026, 11, 2, 8, 15, tzinfo=timezone.utc)
    assert make_run_id("a2", 0, "1a2b3c4d5e6f", when) == "a2_s0_1a2b3c4_20261102-0815"


def test_manifest_rejects_unknown_keys():
    m = new_manifest(run_id="x", seed=1)
    assert m["run_id"] == "x" and m["commit"] is None
    with pytest.raises(KeyError):
        new_manifest(credentials="no")


def test_environment_has_versions_and_no_hostname():
    import socket

    env = environment()
    assert env["python"].startswith("3.") and env["numpy"]
    assert socket.gethostname() not in json.dumps(env)


def test_make_job_renders_valid_files(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "kaggle"))
    from make_job import make_job

    when = datetime(2026, 11, 2, 8, 15, tzinfo=timezone.utc)
    out = make_job("acc2", "someuser", "train", "a2", "0", "owner/sam-reliability-code",
                   "owner/pranet", tmp_path, "1a2b3c4d5e6f", when)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "someuser/a2-s0-1a2b3c4-20261102-0815"
    assert meta["is_private"] == "true" and meta["enable_gpu"] == "true"
    assert meta["dataset_sources"] == ["owner/sam-reliability-code", "owner/pranet"]
    run = (out / "run.py").read_text()
    ast.parse(run)
    assert 'COMMIT = "1a2b3c4d5e6f"' in run and "{{" not in run
    audit = make_job("acc", "u", "audit", "base", "0", "o/c", "o/d", tmp_path, "abcdef0123", when)
    assert json.loads((audit / "kernel-metadata.json").read_text())["enable_gpu"] == "false"
    env = make_job("acc", "u", "env", "base", "0", "o/c", "o/d", tmp_path, "abcdef0123", when)
    assert json.loads((env / "kernel-metadata.json").read_text())["enable_gpu"] == "true"


def test_no_credential_material_in_templates():
    from pathlib import Path

    tdir = Path(__file__).parents[1] / "kaggle" / "templates"
    for p in tdir.iterdir():
        text = p.read_text(encoding="utf-8").lower()
        assert '"key"' not in text and "kaggle.json" not in text, p.name
