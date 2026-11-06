# Evidence record

Each table, figure and main statement of the manuscript, with the files behind it and the command that rebuilds it. All commands run from the repository root with `PYTHONPATH=src`, after `pip install -r requirements-local.txt`. No GPU is needed unless stated: everything starts from the stored per-image outputs.

Table and section numbers follow the manuscript version v14. Paths are relative to the repository root.

## Rebuild everything at one commit

```bash
python scripts/checks/reproduce_outputs.py --commit HEAD --work <empty tmp dir> --report <file>
```

This exports the analysis code of one commit, reruns it on the stored inputs and compares the outputs byte by byte (tables) and pixel by pixel (figures). The last run is in `results/logs/reproduce_report.md`, which also lists the separate rebuilds of the perturbation tables, the checks below and the figures.

## Research questions

RQ2 and RQ3 were reworded after the results were known, so that they ask what the results can answer. The primary RQ2 measures also changed after part of the test results were seen (`docs/PROTOCOL.md` §7.7). The RQ2 analysis is therefore exploratory.

| Question | Evidence |
|---|---|
| RQ1: accuracy and pixel-level calibration | `results/metrics/summary.csv`, `seed_stability.csv`, `paired_consistency.csv` |
| RQ2: image-level quality ranking, also under perturbed boxes | `results/metrics/perturb/rq2_spearman_table.md` (readable), `rq2_spearman_by_seed.csv`, `rq2_spearman_cells.csv`, `rq2_spearman_counts.csv`, `rq2_consistency.csv` |
| RQ3: ensemble against single LoRA, and cost | `results/metrics/paired_consistency.csv` (family `a3_vs_a2`), `results/metrics/cost.csv` |

## Tables and figures

| In the manuscript | Files | Rebuild |
|---|---|---|
| Table 2, datasets | `results/data_checks/dataset_characteristics.csv`, `image_lists.csv` | `python scripts/data/audit_dataset.py --data-root <images> --out-dir results/data_checks --splits-dir data/splits`, `python scripts/data/export_image_lists.py --audit results/data_checks/audit_per_image.csv --val-split data/splits/val_split.json --out-csv results/data_checks/image_lists.csv --data-root <images>` |
| Table 3, configurations | `results/metrics/trainable_params.csv` | `python scripts/analysis/count_params.py --out-dir results/metrics` |
| Table 4, accuracy and calibration | `results/metrics/summary.csv`, `seed_stability.csv` | `python scripts/analysis/aggregate.py --metrics-dir results/metrics --runs results/metrics/runs.csv` |
| Table 5, paired comparisons | `results/metrics/paired.csv`, `paired_consistency.csv` | `aggregate.py`, then `python scripts/analysis/finalize_tables.py --metrics-dir results/metrics` |
| Table 6, failures and ranking | `results/metrics/summary.csv`, `summary_secondary.csv` | `aggregate.py` |
| Fig. 2 | `results/figures/fig2_compact.*` | `python scripts/analysis/fig2_compact.py --metrics-dir results/metrics --runs results/figures/runs_shown.csv --out-dir results/figures` |
| Fig. 3 and other examples | `results/figures/fig3_one_row.*`, `examples_main.*`, `examples_selected.csv` | `python scripts/analysis/example_figure.py select --runs results/metrics/runs.csv --out results/figures/examples_selected.csv`, then `python scripts/analysis/example_figure.py draw --runs results/metrics/runs.csv --selected results/figures/examples_selected.csv --pred-root results/predictions_selected --data-root <images> --out-dir results/figures` |
| Prompt perturbation figure | `results/figures/fig_perturb.*` | `python scripts/analysis/fig_perturb.py --perturb-dir results/metrics/perturb --out-dir results/figures` |

## Prompt perturbation (RQ2)

Everything is computed from the per-image files of the 32 perturbation runs, `results/metrics/perturb/per_image/`.

| Statement | Files | Rebuild |
|---|---|---|
| Spearman rho of A0-A3 by dataset, level P0-P3, score and seed, with paired difference from A0 and 95% interval, marked higher, lower or undetermined | `rq2_spearman_by_seed.csv`; readable version `rq2_spearman_table.md`; reading per cell `rq2_spearman_cells.csv`; counts `rq2_spearman_counts.csv` | `python scripts/checks/rq2_spearman_audit.py --perturb-dir results/metrics/perturb --out-dir results/metrics/perturb`, then `python scripts/checks/rq2_spearman_table.py --perturb-dir results/metrics/perturb --out results/metrics/perturb/rq2_spearman_table.md` |
| A3 versus A0 on every RQ1 and RQ2 metric (exploratory, added after the results were seen) | `a3_vs_a0.csv` | `rq2_spearman_audit.py` |
| Delta from P0, difference from A0, the four comparison families | `delta*.csv`, `did*.csv`, `rq2_pairs.csv`, `rq2_consistency.csv` | `python scripts/analysis/aggregate_perturb.py --perturb-dir results/metrics/perturb` |
| AUROC is read only with at least 20 failures and 20 successes | every cell with its counts: `auroc_cells.csv`; every comparison and its reading: `auroc_comparisons.csv` | `python scripts/checks/auroc_cells.py --perturb-dir results/metrics/perturb` |
| Each statement of the manuscript on RQ2 against the files | `results/logs/rq2_claims_check.md`, `results/data_checks/rq2_claims_check.csv` | `python scripts/checks/verify_rq2_claims.py --metrics-dir results/metrics --out-csv results/data_checks/rq2_claims_check.csv --report results/logs/rq2_claims_check.md` |

Undetermined means the paired interval contains zero. It does not mean the two configurations are equivalent.

## Fig. 3: the image with the largest overconfidence

`results/data_checks/overconfidence_a2_s0_ranked.csv` lists all 380 CVC-ColonDB and 196 ETIS-LaribPolypDB images of A2 seed 0 with predicted IoU (`conf_iou_pred`), actual IoU (`iou`), `overconfidence` = predicted minus actual IoU, and rank. Rank 1 on CVC-ColonDB is `colondb/204` (0.903 predicted, 0.625 actual, 0.278); rank 2 is `colondb/270` (0.270).

```bash
python scripts/checks/overconfidence_check.py --metrics-dir results/metrics --selected results/figures/examples_selected.csv --ranked-csv results/data_checks/overconfidence_a2_s0_ranked.csv --out-csv results/data_checks/overconfidence_a2_s0.csv
```

The command prints rank 1 and rank 2 and exits with an error if the figure image is not rank 1. The rule was set after the results were known. It replaced an earlier rule that compared the predicted IoU with Dice (`docs/PROTOCOL.md` §13).

`results/data_checks/raw_prediction_check.csv` ties the tables to the raw predictions: Dice, IoU and predicted IoU of the 8 selected images were recomputed from the probability maps and masks for A0, A1 seed 0, A2 seeds 0-2 and A3 (rebuilt as the mean of its members); the largest gap is below 1e-7. The probability maps of these 8 images are in `results/predictions_selected/`:

```bash
python scripts/checks/raw_prediction_check.py --runs results/metrics/runs.csv --metrics-dir results/metrics --selected results/figures/examples_selected.csv --pred-root results/predictions_selected --data-root <images> --out-csv results/data_checks/raw_prediction_check.csv
```

## Data independence

| Check | Files | Rebuild |
|---|---|---|
| Train, validation and test images of each dataset, with sha256 of each image and mask | `results/data_checks/image_lists.csv`, `data/splits/val_split.json` | `export_image_lists.py`, see Table 2 |
| Near-duplicate search: every pair within the distance | `results/data_checks/overlap_pairs.csv`, `audit_per_image.csv` | `audit_dataset.py` (needs the images) |
| Pairs behind the 19 flagged CVC-ColonDB images: 36 pairs, 0 within 6 bits on both hashes | `results/data_checks/phash_flagged_pairs.csv` | `python scripts/checks/phash_flagged_pairs.py --data-checks results/data_checks --out-csv results/data_checks/phash_flagged_pairs.csv --data-root <images>`; recomputes the hashes from the image files |
| Montage of those pairs for visual inspection | Not in this repository: it holds dataset images, and the dataset pages state research use without addressing redistribution | Add `--montage <file.jpg>` to the command above |
| Video-sequence overlap of CVC-ClinicDB: all 62 test images share a sequence with training images | `results/data_checks/clinicdb_sequences.csv`, `clinicdb_test_sequence.csv` | `python scripts/checks/clinicdb_sequences.py --data-root <images> --out-dir results/data_checks` |

The sequence table is "Table 1" of the CVC-ClinicDB page on Grand Challenge (https://polyp.grand-challenge.org/CVCClinicDB/). The check assumes that a PraNet file name is the original frame number; the 550 train and 62 test names together are exactly 1 to 612, and no document confirms the assumption. CVC-ColonDB, ETIS-LaribPolypDB and Kvasir-SEG publish no image-to-sequence or image-to-patient map, so independence of the external sets rests mainly on their different hospitals and cannot be fully excluded at video or patient level.

[inspection record: pending]

## Statistics

| Check | Files | Rebuild |
|---|---|---|
| Pairing, runs entering each comparison, consistency rule | `results/logs/statistics_check.md` | `python scripts/checks/statistics_check.py --metrics-dir results/metrics --report results/logs/statistics_check.md` |
| Independent recomputation of all paired comparisons, with numpy only, and with a fresh index matrix (B = 10,000) | `results/logs/paired_independent_report.md`, `results/data_checks/paired_independent.csv` | `python scripts/checks/paired_independent.py --metrics-dir results/metrics --out-csv results/data_checks/paired_independent.csv --report results/logs/paired_independent_report.md` |
| Seed variation against sampling uncertainty | `results/data_checks/variance_sources.csv` | `python scripts/checks/stat_checks.py --metrics-dir results/metrics --out-dir results/data_checks` |
| Per-image ECE as a sensitivity check | `results/metrics/ece_per_image_*.csv` | `python scripts/checks/ece_per_image.py --metrics-dir results/metrics` |
| Integrity of runs | `results/logs/integrity_report.md`, `integrity_perturb.md` | `scripts/checks/integrity.py`, `integrity_perturb.py` |

Reading k of 3 in Table 5:

- A1 - A0 and A2 - A0 use the same single A0 run. A3 - A2 compares one A3 with each of its three members, and A3 is the mean of those members. These comparisons share a run and are not three independent repetitions.
- 3/3 means the three paired intervals exclude zero in one direction. With 1/3 or 2/3 the directions may differ: for ECE(Rroi) of A3 - A2 on CVC-ColonDB, one interval lies above zero and one below.
- Intervals resample images, not videos, and are not adjusted for multiple comparisons.

## Code, configuration, seeds and outputs

| Item | Where |
|---|---|
| Source code and tests | `src/samrel/`, `scripts/`, `tests/` |
| Configuration, environment lock | `configs/base.yaml`, `a0.yaml`, `a1.yaml`, `a2.yaml`, `env_kaggle.lock.json` |
| Seeds | root seed in `configs/base.yaml`, training seeds in `results/metrics/runs.csv`, derivation in `src/samrel/seeding.py` and `docs/PROTOCOL.md` §2 |
| Box prompts and validation split | `data/prompts/`, `data/splits/` |
| Run manifests, environment, training logs | `results/logs/manifests/`, `results/logs/environment/`, `results/logs/train/` |
| Per-image Dice, IoU and confidence scores of every run | `results/metrics/per_image/`, `results/metrics/perturb/per_image/` |
| Per-image ECE bin statistics | `results/metrics/ece_bins/`, `results/metrics/perturb/ece_bins/` |

## What is not in this repository

- **Images and masks.** Obtain them from the original sources (see `README.md`, "Data"); `image_lists.csv` lists the sha256 of every file used.
- **Pixel-level predictions (`.npz`) of the other images.** This repository holds the 8 example images for the 5 runs needed (`results/predictions_selected/`, 40 files, 12.3 MB). All predictions are outputs of private Kaggle kernels, so only the owner of the Kaggle account that ran them can download them. Their total size is about 1.6 GB; this is an estimate from the mean size of 60 downloaded files (281 KB) times 798 images and 7 models, not a measurement. The per-image values derived from all predictions are in this repository. Recomputing the per-image tables from images needs a GPU run with the stored prompts and the trained weights.
- **Trained weights (`.pt`).** They are kept in a private Kaggle dataset. Without them, A1, A2 and A3 can be retrained from `scripts/run/run_task.py`, within the limits on determinism in `docs/PROTOCOL.md` §9.3.
