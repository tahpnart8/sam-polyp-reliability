# How the paired comparisons are built

## Pairing

- Image order: all 8 main runs hold the same images in the same order: True. All 32 perturbation runs: True. Perturbation images are exactly the ColonDB and ETIS images of the main runs: True.
- Index matrix: `samrel.bootstrap.indices_for_dataset` seeds on the root seed, the label `bootstrap` and the dataset only, so every run of a dataset, at every perturbation level, uses one matrix (1000 x n). Main and perturbation matrices identical: True.
- A paired difference is computed per resample on the same image indices, in `samrel.aggregate.paired` and `samrel.perturb.diff_row`. Images stay paired.

## Which runs enter each comparison

- `a1_vs_a0`: a1_s0-a0_s0; a1_s1-a0_s0; a1_s2-a0_s0
- `a2_vs_a0`: a2_s0-a0_s0; a2_s1-a0_s0; a2_s2-a0_s0
- `a2_vs_a1`: a2_s0-a1_s0; a2_s1-a1_s1; a2_s2-a1_s2
- `a3_vs_a2`: a3_s012-a2_s0; a3_s012-a2_s1; a3_s012-a2_s2

Consequences for reading k of 3:
- A1 - A0 and A2 - A0 use one A0 run, which is deterministic. The three differences share A0 and vary only through the fine-tuned model.
- A3 - A2 uses one A3 and each of its three members. A3 is the pixel-wise mean of those members, so each difference compares a model with an ensemble that contains it. The three comparisons share A3 and are not three independent repetitions.
- A2 - A1 pairs runs with the same seed label. The seed sets minibatch order and, for A2, the LoRA initialisation; the pairing is a convention and the two runs share no trained weights.
- All comparisons on a dataset use the same resampled images, so intervals in different cells are correlated. No interval is adjusted for multiple comparisons. Images are resampled, not videos.

## Consistency rule

- `paired_consistency.csv` flags recomputed from `paired.csv`: 0 of 202 differ.
- Comparisons whose interval excludes 0 while the point difference has the opposite sign: 0 of 333.
- `consistent` needs all three intervals to exclude 0 with one sign. k counts the intervals that exclude 0 and does not require one sign when k is below 3.

## Table 5 cells with k of 1 or 2

| Comparison | Dataset | Metric | k of 3 | Above 0 | Below 0 |
|---|---|---|---|---|---|
| a2_vs_a0 | etis | ece_roi | 2 | 0 | 2 |
| a2_vs_a1 | colondb | spearman_iou_pred | 2 | 2 | 0 |
| a2_vs_a1 | etis | aurc_iou_pred | 2 | 0 | 2 |
| a3_vs_a2 | colondb | ece_roi | 2 | 1 | 1 |
| a3_vs_a2 | colondb | spearman_iou_pred | 2 | 2 | 0 |
| a3_vs_a2 | etis | aurc_iou_pred | 2 | 0 | 2 |
| a3_vs_a2 | etis | dice | 2 | 2 | 0 |
| a3_vs_a2 | etis | ece_roi | 2 | 2 | 0 |

`a3_vs_a2` rows checked: 150.
