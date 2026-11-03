"""Stage the code bundle for the private Kaggle Dataset ``sam-reliability-code``. Does not upload.

Usage:
    python scripts/kaggle/bundle_code.py --kaggle-user <owner> --out kaggle/_code_bundle

Then, by hand (first time ``datasets create``, later ``datasets version``):
    KAGGLE_CONFIG_DIR=<dir with kaggle.json> python -m kaggle datasets create -p kaggle/_code_bundle --dir-mode zip
    KAGGLE_CONFIG_DIR=... python -m kaggle datasets version -p kaggle/_code_bundle --dir-mode zip -m "<commit7>"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from samrel.manifest import git_commit
from samrel.paths import REPO_ROOT

# scripts/kaggle (local orchestration) is not shipped: changing it must not force a new bundle.
INCLUDE = ("src", "configs", "scripts/run", "scripts/data", "scripts/analysis", "data/prompts", "data/splits",
           "requirements-kaggle.txt", "docs/PROTOCOL.md", "pyproject.toml")


def bundle(out: Path, kaggle_user: str, commit: str) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for item in INCLUDE:
        src = REPO_ROOT / item
        dst = out / item
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (out / "COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    meta = (REPO_ROOT / "kaggle" / "templates" / "dataset-metadata.json.tmpl").read_text(encoding="utf-8")
    (out / "dataset-metadata.json").write_text(meta.replace("{{KAGGLE_USER}}", kaggle_user), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    commit, dirty = git_commit(paths=INCLUDE)
    if dirty or commit == "unknown":
        raise SystemExit("STOP: commit first; COMMIT.txt must name a clean commit")
    print(bundle(args.out, args.kaggle_user, commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
