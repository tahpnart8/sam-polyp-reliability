# Rebuilding tables and figures from one commit

- Analysis code commit: `061163656a05b97385eb7be24c88c0c423f3e8fc`
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

## Additional rebuilds at the same commit

Run from this repository into a temporary folder, then compared with the committed files.

| Output | Script | Result |
|---|---|---|
| `results/metrics/perturb/{summary,delta,did,rq2_pairs,delta_consistency,did_consistency,rq2_consistency}.csv` | `scripts/analysis/aggregate_perturb.py` | identical bytes (7 of 7) |
| `results/metrics/perturb/{rq2_spearman_by_seed,rq2_spearman_cells,rq2_spearman_counts,a3_vs_a0}.csv` | `scripts/checks/rq2_spearman_audit.py` | identical content (4 of 4) |
| `results/data_checks/paired_independent.csv`, `results/logs/paired_independent_report.md` | `scripts/checks/paired_independent.py` | identical bytes |
| `results/data_checks/overconfidence_a2_s0.csv` | `scripts/checks/overconfidence_check.py` | identical bytes |
| `results/figures/fig3_one_row.png`, `examples_main.png` | `scripts/analysis/example_figure.py draw`, with the 60 downloaded `.npz` files of the selected images | identical pixels |
| `results/figures/fig2_compact.png`, `fig_perturb.png` | `scripts/analysis/fig2_compact.py`, `fig_perturb.py` | identical pixels |

Test suite at this commit: 83 passed, 1 skipped (needs the raw data), 0 failed.
