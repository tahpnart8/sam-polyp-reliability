"""Render kaggle/templates/ into kaggle/jobs/<run_id>/ (PROTOCOL §9.1). Does not push.

Usage:
    python scripts/kaggle/make_job.py --account <name> --kaggle-user <username> --task audit \
        --config base --seed 0 --code-dataset <owner>/sam-reliability-code \
        --data-dataset <owner>/<pranet-dataset> --jobs-dir kaggle/jobs

Push afterwards by hand, with your Kaggle credentials:
    KAGGLE_CONFIG_DIR=<dir with kaggle.json> python -m kaggle kernels push -p <job dir>
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from samrel.manifest import git_commit, make_run_id
from samrel.paths import REPO_ROOT

TEMPLATES = REPO_ROOT / "kaggle" / "templates"
TASKS = ("env", "audit", "infer", "train", "ensemble", "benchmark", "perturb")
# env records the GPU/CUDA stack; ensemble runs on GPU too, since the CPU image ships torch +cpu
# and would fail the env lock (a3 job of 29/10 stopped there)
GPU_TASKS = {"env", "infer", "train", "ensemble", "benchmark", "perturb"}
_SLUG_OK = re.compile(r"^[a-z0-9-]{5,50}$")


def render(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in template: {re.findall(r'{{[A-Z_]+}}', text)}")
    return text


def make_job(account: str, kaggle_user: str, task: str, config: str, seed: str, code_dataset: str,
             data_dataset: str, jobs_dir: Path, commit: str, when: datetime | None = None,
             extra_args: list[str] | None = None, kernel_sources: list[str] | None = None,
             label: str | None = None) -> Path:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    label = label or {"env": "env", "audit": "audit", "ensemble": "a3", "benchmark": "cost"}.get(task, config)
    run_id = make_run_id(label, seed, commit, when)
    slug = run_id.replace("_", "-").lower()
    if not _SLUG_OK.match(slug):
        raise ValueError(f"bad kernel slug {slug!r}")
    values = {
        "RUN_ID": run_id, "TASK": task, "CONFIG": config, "SEED": str(seed), "COMMIT": commit,
        "ACCOUNT": account, "KAGGLE_USER": kaggle_user, "KERNEL_SLUG": slug,
        "CODE_DATASET": code_dataset, "DATA_DATASET": data_dataset,
        "ENABLE_GPU": "true" if task in GPU_TASKS else "false",
        "EXTRA_ARGS": json.dumps(list(extra_args or [])),
        "KERNEL_SOURCES": json.dumps(list(kernel_sources or [])),
    }
    out = jobs_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "kernel-metadata.json").write_text(
        render((TEMPLATES / "kernel-metadata.json.tmpl").read_text(encoding="utf-8"), values), encoding="utf-8")
    (out / "run.py").write_text(
        render((TEMPLATES / "run.py.tmpl").read_text(encoding="utf-8"), values), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--account", required=True)
    ap.add_argument("--kaggle-user", required=True)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--code-dataset", required=True)
    ap.add_argument("--data-dataset", required=True)
    ap.add_argument("--jobs-dir", required=True, type=Path)
    ap.add_argument("--extra", default="", help="extra run_task args, e.g. '--limit-images 10 --max-steps 20'")
    ap.add_argument("--kernel-source", action="append", default=[], help="earlier kernel whose output is mounted")
    ap.add_argument("--label", help="run_id prefix instead of the config (perturb: pp-a0a1, pp-a2a3)")
    args = ap.parse_args(argv)
    from bundle_code import INCLUDE

    head, dirty = git_commit(paths=INCLUDE)
    if dirty or head == "unknown":
        raise SystemExit("STOP: commit your changes first; every job must map to a clean commit")
    # The job must name the commit that is in the uploaded code dataset. Later commits that only
    # touch logs or the paper are fine; any change to shipped paths needs a new bundle first.
    commit = (REPO_ROOT / "kaggle" / "_code_bundle" / "COMMIT.txt").read_text().strip()
    import subprocess

    changed = subprocess.run(["git", "diff", "--quiet", commit, head, "--", *INCLUDE], cwd=REPO_ROOT)
    if changed.returncode != 0:
        raise SystemExit(f"STOP: shipped code changed since bundled commit {commit[:7]}; re-bundle and upload")
    out = make_job(args.account, args.kaggle_user, args.task, args.config, args.seed,
                   args.code_dataset, args.data_dataset, args.jobs_dir, commit,
                   extra_args=args.extra.split(), kernel_sources=args.kernel_source, label=args.label)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
