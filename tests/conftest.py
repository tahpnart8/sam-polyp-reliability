"""Runtime guard: no test may write into the real results directory (PROTOCOL §11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from samrel import paths
from samrel.mock import make_fake_pranet


@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    target = tmp_path / "results"
    monkeypatch.setenv("SAMREL_RESULTS_DIR", str(target))
    assert paths.results_dir() == target.resolve()
    assert paths.results_dir() != paths.DEFAULT_RESULTS.resolve()
    yield


@pytest.fixture(scope="session")
def fake_pranet(tmp_path_factory) -> Path:
    return make_fake_pranet(tmp_path_factory.mktemp("pranet"))
