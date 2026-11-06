# Accuracy versus Reliability: SAM Fine-Tuning for Cross-Center Polyp Segmentation

Code, protocol and results for the manuscript below (repository: https://github.com/tahpnart8/sam-polyp-reliability).

> **Accuracy versus Reliability: Evaluating Calibration and Failure Detection Across SAM Fine-Tuning Strategies for Cross-Center Polyp Segmentation**
> Quoc Hung Nguyen, Duc Phat Tran, Le Hai Long Nguyen, Thuy Ngan Nguyen, Anh Tu Tran, Hoang Phuc Nguyen.
> University of Economics Ho Chi Minh City (UEH).

**Status:** manuscript prepared for submission to ICICC-2027 (Springer LNNS). It has not been accepted or published. This README will be updated if that changes.

## Summary

Most studies of the Segment Anything Model (SAM) for colonoscopy polyp segmentation report Dice and IoU only. We ask whether fine-tuning SAM also makes its confidence more trustworthy on images from centers not seen during training. Four configurations of SAM ViT-B are compared under one controlled protocol:

| ID | Configuration | Updated weights |
|---|---|---|
| A0 | zero-shot | none |
| A1 | decoder-only fine-tuning | mask decoder |
| A2 | LoRA fine-tuning | LoRA (rank 4, q and v) in all 12 encoder blocks + mask decoder |
| A3 | LoRA ensemble | none; mean of three A2 models trained with different seeds |

Models are trained on Kvasir-SEG and CVC-ClinicDB (PraNet split) and tested on their held-out images and on two external datasets, CVC-ColonDB and ETIS-LaribPolypDB. We measure accuracy (Dice, IoU), pixel-level calibration (expected calibration error over the whole image, the region around the polyp and the polyp boundary) and image-level quality ranking (AURC, Spearman correlation between confidence and Dice), with binary failure detection (AUROC) as a supplementary analysis.

Research questions:

- **RQ1.** How do segmentation accuracy and pixel-level calibration change across SAM adaptation strategies (zero-shot, decoder-only fine-tuning and LoRA fine-tuning) on source-domain test sets and on test sets from unseen centers?
- **RQ2.** How reliably do image-level confidence scores rank segmentation quality across SAM adaptation strategies on unseen-center datasets, including under perturbed box prompts?
- **RQ3.** Does an ensemble of LoRA models improve segmentation accuracy, calibration, and image-level quality ranking over single LoRA models, and at what computational cost?

RQ2 and RQ3 were reworded after the results were known, and binary failure detection became a supplementary analysis. The RQ2 analysis is exploratory.

**All configurations receive box prompts derived from the ground-truth masks (oracle boxes).** The results describe SAM under oracle-box prompting, not a fully automatic system.

## Main findings (as reported in the manuscript)

On the external test sets and under oracle boxes:

- LoRA fine-tuning raised Dice (CVC-ColonDB 0.811 to 0.932; ETIS-LaribPolypDB 0.907 to 0.942) and reduced calibration error around the polyp.
- The fine-tuned confidence scores did not consistently rank images by segmentation quality better than zero-shot SAM (Spearman correlation), even though AURC decreased; AURC also reflects the lower overall error, so a lower AURC alone is not better failure detection.
- After fine-tuning almost no image fell below Dice 0.5, so AUROC could not support conclusions for most configurations (every cell with its counts: `results/metrics/perturb/auroc_cells.csv`). The main RQ2 metrics were changed to AURC and Spearman correlation **after** seeing these failure counts; this change is reported openly (see `docs/PROTOCOL.md` §7.7).
- The ensemble added little over single LoRA models (for example Dice +0.001 to +0.002 on CVC-ColonDB, with mixed results on calibration and ranking) at about three times the inference time on one Tesla T4.
- A prompt-perturbation check (§14) shifted and rescaled the oracle boxes at three predefined levels. Dice dropped by a similar amount for every configuration (IoU dropped somewhat more for the fine-tuned models); A1 and A2 were never consistently above zero-shot SAM in Spearman correlation at any level, with either score, while some AURC comparisons reversed. With the IoU-head score no point estimate of A1 or A2 was above zero-shot SAM. Paired intervals above zero-shot SAM occurred only with mean pixel confidence on CVC-ColonDB: A2 seed 1 at the oracle level, and A3 at the strongest level (A3 versus zero-shot SAM is an exploratory analysis added after the results were seen). Most other comparisons are undetermined, which does not mean equivalent: their interval half-widths are 0.07 to 0.13 (`results/metrics/perturb/rq2_spearman_counts.csv`).

These findings come from one backbone, one LoRA rank, three seeds and two external datasets, with confidence intervals not corrected for multiple comparisons.

## Evidence

`docs/EVIDENCE.md` maps each table, figure and main statement to its files and the command that rebuilds it.

## Repository layout

```
configs/            experiment configuration (base.yaml shared, a0/a1/a2.yaml differ only in model/training)
data/prompts/       fixed box prompts: oracle (prompts.json) and perturbed levels (prompts_pp10/20/30.json)
data/splits/        validation split and the (empty) exclusion list
docs/PROTOCOL.md    full experimental protocol; code comments refer to its sections
docs/EVIDENCE.md    each result of the manuscript, its files and the command that rebuilds it
kaggle/templates/   job templates used to run training and inference on Kaggle GPUs
results/
  data_checks/      dataset audit, near-duplicate pairs, ClinicDB sequence overlap, ECE weight check
  figures/          figures of the paper (PNG 600 dpi and PDF)
  logs/             run manifests, environment records, training logs, integrity and reproduction reports
  metrics/          per-image results of every run, ECE bin statistics, summary and comparison tables
  predictions_selected/  saved probability maps of the eight example images (all others: see docs/EVIDENCE.md)
  metrics/perturb/  prompt-perturbation runs and tables
scripts/
  data/             dataset audit, prompt and split generation
  run/              training and inference (run_task.py), ensemble and cost (run_post.py), perturbation
  analysis/         tables and figures from the per-run files
  checks/           integrity checks and the reproduction check
  kaggle/           job rendering and code bundling for Kaggle
src/samrel/         library code (data, prompts, models, LoRA, training, inference, metrics, bootstrap)
tests/              unit and integration tests (CPU, fake data)
```

## Installation

Python 3.11 was used for analysis and tests.

```bash
pip install -r requirements-local.txt
pip install -e .
```

Training and inference ran on Kaggle (Tesla T4, PyTorch 2.11, CUDA 12.8); the exact versions are in `configs/env_kaggle.lock.json` and `results/logs/environment/`.

## Data

The images are **not** included and must be obtained from their original sources. Download the PraNet training and test archives from the links in section 3.1 of the PraNet README (https://github.com/DengPingFan/PraNet) and unzip them into one folder that contains `TrainDataset/` and `TestDataset/`. The sha256 of both archives is recorded in `configs/base.yaml`.

The datasets are free for research and education. If you use them, cite the original publications:

- Kvasir-SEG: Jha et al., MMM 2020.
- CVC-ClinicDB: Bernal et al., Comput. Med. Imaging Graph. 43, 2015.
- CVC-ColonDB: Tajbakhsh et al., IEEE Trans. Med. Imaging 35(2), 2016 (as cited by PraNet for the 380-image version).
- ETIS-LaribPolypDB: Silva et al., Int. J. CARS 9(2), 2014.
- CVC-300 (EndoScene test part): Vázquez et al., J. Healthc. Eng. 2017.
- The split: Fan et al., PraNet, MICCAI 2020.

The SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) is downloaded from the official segment-anything release; its sha256 is checked on every run.

## Reproducing the results

The stored per-image files in `results/metrics/` are enough to rebuild every table and figure without a GPU:

```bash
python scripts/analysis/aggregate.py --metrics-dir results/metrics --runs results/metrics/runs.csv
python scripts/analysis/finalize_tables.py --metrics-dir results/metrics
python scripts/analysis/fig2_compact.py --metrics-dir results/metrics --runs results/figures/runs_shown.csv --out-dir results/figures
python scripts/analysis/aggregate_perturb.py --perturb-dir results/metrics/perturb
python scripts/analysis/fig_perturb.py --perturb-dir results/metrics/perturb --out-dir results/figures
python scripts/checks/ece_per_image.py --metrics-dir results/metrics
python scripts/checks/integrity_perturb.py --report results/logs/integrity_perturb.md
python scripts/checks/rq2_spearman_audit.py --perturb-dir results/metrics/perturb --out-dir results/metrics/perturb
python scripts/checks/paired_independent.py --metrics-dir results/metrics --out-csv results/data_checks/paired_independent.csv --report results/logs/paired_independent_report.md
python scripts/checks/overconfidence_check.py --metrics-dir results/metrics --selected results/figures/examples_selected.csv --ranked-csv results/data_checks/overconfidence_a2_s0_ranked.csv --out-csv results/data_checks/overconfidence_a2_s0.csv
python scripts/checks/rq2_spearman_table.py --perturb-dir results/metrics/perturb --out results/metrics/perturb/rq2_spearman_table.md
python scripts/checks/auroc_cells.py --perturb-dir results/metrics/perturb
python scripts/checks/verify_rq2_claims.py --metrics-dir results/metrics --out-csv results/data_checks/rq2_claims_check.csv --report results/logs/rq2_claims_check.md
python scripts/checks/statistics_check.py --metrics-dir results/metrics --report results/logs/statistics_check.md
```

`scripts/checks/reproduce_outputs.py --commit HEAD --work <tmp dir> --report <file>` exports the analysis code of one commit, reruns it on the stored inputs and compares the outputs byte by byte (tables) and pixel by pixel (figures).

To rerun the models (GPU needed), with `<data>` the PraNet folder:

```bash
# A0 (zero-shot inference on all test sets)
python scripts/run/run_task.py --task infer --config a0 --seed 0 --run-id <id> --commit <c> \
    --data-root <data> --checkpoint sam_vit_b_01ec64.pth --out <out> --work <work>
# A1 / A2 (training, then inference with the best validation epoch), seeds 0, 1, 2
python scripts/run/run_task.py --task train --config a2 --seed 0 --run-id <id> --commit <c> \
    --data-root <data> --checkpoint sam_vit_b_01ec64.pth --out <out> --work <work>
# A3 (ensemble of the three A2 runs) and the inference-cost benchmark
python scripts/run/run_post.py --task ensemble --run-id <id> --commit <c> --data-root <data> \
    --out <out> --search-root <dir with the A2 outputs> --a2-runs <a2_s0>,<a2_s1>,<a2_s2>
# Prompt perturbation (existing weights, oracle + three perturbed levels)
python scripts/run/run_perturb.py --commit <c> --stamp <yyyymmdd-hhmm> --data-root <data> \
    --checkpoint sam_vit_b_01ec64.pth --out <out> --search-root <dir with checkpoints> \
    --models <run ids> --levels data/prompts/prompts_pp10.json,data/prompts/prompts_pp20.json,data/prompts/prompts_pp30.json
```

The Kaggle jobs used for the paper were generated from `kaggle/templates/` (see the README there).

## Where each table and figure comes from

| In the manuscript | File(s) | Script |
|---|---|---|
| Table 2, dataset characteristics | `results/data_checks/dataset_characteristics.csv` | `scripts/data/audit_dataset.py` |
| Table 3, configurations and parameter counts | `results/metrics/trainable_params.csv` | `scripts/analysis/count_params.py` |
| Table 4, accuracy and calibration | `results/metrics/summary.csv`, `seed_stability.csv` | `scripts/analysis/aggregate.py` |
| Table 5, paired comparisons | `results/metrics/paired.csv`, `paired_consistency.csv` | `aggregate.py`, `finalize_tables.py` |
| Table 6, failures and quality ranking | `results/metrics/summary.csv` (AURC, Spearman, AUROC with `n_failures`), `summary_secondary.csv` | `aggregate.py` |
| Fig. 2, reliability and risk-coverage | `results/figures/fig2_compact.*` | `scripts/analysis/fig2_compact.py` |
| Fig. 3 and example segmentations | `results/figures/fig3_one_row.*`, `examples_main.*`, `examples_selected.csv`; values in `results/data_checks/overconfidence_a2_s0.csv` | `scripts/analysis/example_figure.py` (drawing needs the downloaded predictions), `scripts/checks/overconfidence_check.py` |
| Computational cost | `results/metrics/cost.csv` | `run_post.py --task benchmark`, `finalize_tables.py` |
| Prompt perturbation | `results/metrics/perturb/*.csv`, `results/figures/fig_perturb.*` | `run_perturb.py`, `aggregate_perturb.py`, `fig_perturb.py` |
| Spearman by dataset, level, score and seed, each against A0; A3 versus A0 (exploratory) | `results/metrics/perturb/rq2_spearman_{by_seed,cells,counts}.csv`, `a3_vs_a0.csv` | `scripts/checks/rq2_spearman_audit.py` |
| Independent recomputation of the paired comparisons | `results/logs/paired_independent_report.md`, `results/data_checks/paired_independent.csv` | `scripts/checks/paired_independent.py` |
| AUROC cells and comparisons | `results/metrics/perturb/auroc_cells.csv`, `auroc_comparisons.csv` | `scripts/checks/auroc_cells.py` |
| Per-image ECE check | `results/metrics/ece_per_image_*.csv` | `scripts/checks/ece_per_image.py` |
| Data independence checks | `results/data_checks/{image_lists,overlap_pairs,phash_flagged_pairs,clinicdb_sequences,clinicdb_test_sequence}.csv` | `audit_dataset.py`, `scripts/data/export_image_lists.py`, `scripts/checks/phash_flagged_pairs.py`, `scripts/checks/clinicdb_sequences.py` |

Table and figure numbers follow the current manuscript and may change in later versions. Fig. 1 (methodology diagram) and Table 1 (related work) do not come from code.

## Reproducibility notes

- Seeds are derived by a pure function of a root seed and a label (`src/samrel/seeding.py`); prompts and the validation split were generated once and are stored; the bootstrap indices are regenerated from that seed.
- The experiments were run from a private development repository. The commit hashes in run identifiers and manifests (for example `d60e848`, `85ca9c1`) refer to that repository, not to the history of this one. The code here is the same code, with comments and report text edited for publication; the computation was not changed.
- Integrity reports: `results/logs/integrity_report.md` (main runs) and `results/logs/integrity_perturb.md` (perturbation runs).
- Known limitations of the data: CVC-ClinicDB is split by image, and by the official frame-to-sequence table every one of its 62 test images shares a video sequence with training images (`results/data_checks/clinicdb_sequences.csv`); CVC-ColonDB and ETIS-LaribPolypDB publish no image-to-sequence mapping, so the bootstrap cannot account for frames from the same video.

## Model weights

The fine-tuned weights (A1 and A2, three seeds each) are not released in this repository. They are available from the authors on reasonable request.

## License

- Code: MIT License, see `LICENSE`.
- Results, tables and figures in `results/` and the protocol in `docs/`: Creative Commons Attribution 4.0 (CC BY 4.0), see `LICENSE-results`.
- Example figures show images from the public datasets above; those images remain under the terms of their original datasets.
- The code depends on, but does not include, Meta's segment-anything (Apache-2.0). The LoRA module is an independent implementation following the SAMed design.

## Citation

If you use this code or these results, please cite the manuscript (see `CITATION.cff`). Until it is published:

```
Q. H. Nguyen, D. P. Tran, L. H. L. Nguyen, T. N. Nguyen, A. T. Tran, H. P. Nguyen.
Accuracy versus Reliability: Evaluating Calibration and Failure Detection Across SAM
Fine-Tuning Strategies for Cross-Center Polyp Segmentation. Manuscript, 2026.
Code and results: https://github.com/tahpnart8/sam-polyp-reliability
```

Contact: Quoc Hung Nguyen (corresponding author), hungngq@ueh.edu.vn.
