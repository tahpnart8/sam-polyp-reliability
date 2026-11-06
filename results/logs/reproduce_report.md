# Rebuilding tables and figures from one commit

- Analysis code commit: `f3b7a9497fa95486f53e7ab4e043bd8b704c5c5d`
- Inputs: `results/metrics/{per_image,ece_bins,cost,runs.csv,trainable_params.csv}` (repository copy)
- Compared with the files committed in `results/`.

| File | Result |
|---|---|
| `results/metrics/summary.csv` | identical bytes |
| `results/metrics/summary_secondary.csv` | identical bytes |
| `results/metrics/seed_stability.csv` | identical bytes |
| `results/metrics/seed_stability_secondary.csv` | identical bytes |
| `results/metrics/paired.csv` | identical bytes |
| `results/metrics/paired_secondary.csv` | identical bytes |
| `results/metrics/paired_consistency.csv` | identical bytes |
| `results/metrics/paired_consistency_secondary.csv` | identical bytes |
| `results/metrics/cost.csv` | identical bytes |
| `results/figures/reliability_roi.png` | identical pixels |
| `results/figures/reliability_bnd.png` | identical pixels |
| `results/figures/reliability_img.png` | identical pixels |
| `results/figures/risk_coverage_iou_pred.png` | identical pixels |
| `results/figures/examples_selected.csv` | identical bytes |

PDFs are not compared (they embed a creation date). `example_figure.py draw` needs the prediction `.npz` files and is not rerun.

**0 differences**

## Additional rebuilds

Run from this repository into a temporary folder on 2026-11-06, then compared with the committed files. The analysis code is the same at `65746cc`, `f3b7a94` and later commits; those commits change only `docs/EVIDENCE.md`, and add `scripts/checks/a3_score_comparison.py` with its table.

| Output | Script | Result |
|---|---|---|
| `results/metrics/perturb/{summary,delta,did,rq2_pairs,delta_consistency,did_consistency,rq2_consistency}.csv` | `scripts/analysis/aggregate_perturb.py` | identical bytes (7 of 7) |
| `results/metrics/perturb/{rq2_spearman_by_seed,rq2_spearman_cells,rq2_spearman_counts,a3_vs_a0}.csv` | `scripts/checks/rq2_spearman_audit.py` | identical bytes (4 of 4) |
| `results/metrics/perturb/rq2_spearman_table.md` | `scripts/checks/rq2_spearman_table.py` | identical bytes |
| `results/metrics/perturb/{auroc_cells,auroc_comparisons}.csv` | `scripts/checks/auroc_cells.py` | identical bytes (2 of 2) |
| `results/logs/rq2_claims_check.md`, `results/data_checks/rq2_claims_check.csv` | `scripts/checks/verify_rq2_claims.py` | identical bytes; 24 of 24 statements match |
| `results/logs/statistics_check.md` | `scripts/checks/statistics_check.py` | identical bytes |
| `results/data_checks/paired_independent.csv`, `results/logs/paired_independent_report.md` | `scripts/checks/paired_independent.py` | identical bytes |
| `results/data_checks/overconfidence_a2_s0_ranked.csv`, `overconfidence_a2_s0.csv` | `scripts/checks/overconfidence_check.py` | identical bytes; the Fig. 3 image is rank 1 |
| `results/data_checks/image_lists.csv`, `phash_flagged_pairs.csv` | `scripts/data/export_image_lists.py`, `scripts/checks/phash_flagged_pairs.py` | identical bytes; 0 differences in 72 recomputed hash distances |
| `results/data_checks/raw_prediction_check.csv` | `scripts/checks/raw_prediction_check.py`, from `results/predictions_selected/` | identical bytes; largest gap 4e-08 |
| `results/metrics/a3_pairwise_vs_iou_head.csv` | `scripts/checks/a3_score_comparison.py` | identical bytes |
| `results/figures/fig3_one_row.png`, `examples_main.png` | `scripts/analysis/example_figure.py draw`, from `results/predictions_selected/` | identical pixels |
| `results/figures/fig2_compact.png`, `fig_perturb.png` | `scripts/analysis/fig2_compact.py`, `fig_perturb.py` | identical pixels |

Test suite: 83 passed, 1 skipped (needs the raw data), 0 failed. `scripts/checks/ece_per_image.py` and `integrity_perturb.py` were not rerun in this round; their code and inputs did not change.
