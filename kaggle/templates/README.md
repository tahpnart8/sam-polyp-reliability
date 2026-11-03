# Kaggle job templates

The runs of the paper were executed as Kaggle script kernels through the official Kaggle API (`python -m kaggle ...`).

| File | Role |
|---|---|
| `kernel-metadata.json.tmpl` | private script kernel, internet on (installs `segment-anything`, downloads the SAM checkpoint), GPU on for GPU tasks |
| `run.py.tmpl` | checks that `COMMIT.txt` of the code dataset matches the job's commit (stops otherwise), installs `requirements-kaggle.txt`, checks the environment lock, writes the environment record and manifest, runs the task |
| `dataset-metadata.json.tmpl` | private Kaggle Dataset holding the code (one version per commit) |

Workflow:

1. Commit, then `python scripts/kaggle/bundle_code.py --kaggle-user <owner> --out kaggle/_code_bundle` and upload it with `kaggle datasets create` (first time) or `kaggle datasets version`.
2. Upload the PraNet archives as a private Kaggle Dataset.
3. `python scripts/kaggle/make_job.py --account <name> --kaggle-user <user> --task <env|audit|infer|train|ensemble|benchmark|perturb> --config <a0|a1|a2|base> --seed <0|1|2> --code-dataset <owner>/sam-reliability-code --data-dataset <owner>/<pranet dataset> --jobs-dir kaggle/jobs` (add `--kernel-source` for earlier kernels whose outputs a job needs, `--extra` for task arguments).
4. Push with your own Kaggle credentials: `python -m kaggle kernels push -p kaggle/jobs/<run_id>`.
5. Follow with `kaggle kernels status <user>/<slug>` and download with `kaggle kernels output <user>/<slug>`; copy `out/metrics` and `out/logs` into `results/`.

Never put credentials in this repository.
