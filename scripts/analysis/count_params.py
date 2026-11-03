"""Count trainable parameters of A0/A1/A2 on the real SAM ViT-B architecture (PROTOCOL §5.5; Table 3).

Counts do not depend on the weight values, so the architecture is built without the checkpoint.
A3 trains nothing; it holds three A2 models.

Usage:
    python scripts/analysis/count_params.py --out-dir results/metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from samrel.config import load_config
from samrel.constants import PARAM_COLUMNS
from samrel.lora import count_parameters
from samrel.manifest import git_commit
from samrel.models import build_model, param_row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    commit, dirty = git_commit()
    if dirty:
        raise SystemExit("STOP: commit first so the counts map to a clean commit")
    rows = []
    for name in ("a0", "a1", "a2"):
        cfg = load_config(name)
        rows.append(param_row(cfg, count_parameters(build_model(cfg, checkpoint=None)), commit))
    df = pd.DataFrame(rows, columns=list(PARAM_COLUMNS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "trainable_params.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
