# Rebuilding tables and figures from one commit

- Analysis code commit: `4deead99500d05888017268742b8eb790c1f9c6b`
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
