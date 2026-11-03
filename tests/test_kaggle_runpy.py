"""The rendered Kaggle run.py passes the package path to every child process (bug of 2026-10-29)."""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / "kaggle" / "templates" / "run.py.tmpl"


def test_every_subprocess_gets_pythonpath():
    text = TEMPLATE.read_text(encoding="utf-8")
    calls = re.findall(r"subprocess\.check_call\((.*?)\)\n", text, flags=re.S)
    script_calls = [c for c in calls if "scripts/" in c or c.strip().startswith("cmd")]
    assert script_calls and all("env=SUB_ENV" in c for c in script_calls)
    assert '"PYTHONPATH": str(code_dir / "src")' in text
