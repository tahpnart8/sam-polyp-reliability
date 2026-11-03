"""Static guard: test sources never target results/ and never call writers without out_dir (PROTOCOL §11)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from samrel import paths

TESTS = Path(__file__).parent
WRITER = re.compile(r"^(write_|save_|run_)")


def _sources():
    for p in sorted(TESTS.glob("*.py")):
        if p.name != Path(__file__).name:
            yield p, p.read_text(encoding="utf-8")


def test_no_literal_results_path():
    for p, text in _sources():
        assert not re.search(r"""["'][^"']*\bresults[/\\]""", text), f"{p.name} names results/"
        assert "DEFAULT_RESULTS" not in text or p.name == "conftest.py", p.name


def test_writer_calls_pass_out_dir():
    for p, text in _sources():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
                if WRITER.match(name) and name != "write_text":
                    kws = {k.arg for k in node.keywords}
                    assert kws & {"out_dir", "path"} or len(node.args) >= 2, (
                        f"{p.name}:{node.lineno} calls {name} without an explicit output")


def test_runtime_guard_active():
    assert paths.results_dir() != paths.DEFAULT_RESULTS.resolve()
