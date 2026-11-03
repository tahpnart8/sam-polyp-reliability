# Experimental protocol

This document is the protocol behind the code. Comments in `src/`, `scripts/`, `configs/` and `tests/` point to its sections as `PROTOCOL §x.y`. It was written before the corresponding code and, except where a section says otherwise, before any model result on the test sets. Section numbers are stable; some numbers are skipped on purpose.

In the development repository this document was called `CONTRACT`; a few stored files (for example `data/splits/excluded_images.json`) still carry that name, and their content is kept unchanged so that the hashes in the run manifests stay valid.

---

## §1. Data

### §1.1 Datasets and roles
The standard PraNet split is used (training and test archives linked in the PraNet repository README).

| `dataset` key | PraNet folder | Images | Role |
|---|---|---|---|
| `kvasir_train` | `TrainDataset` (Kvasir part) | 900 | training |
| `clinicdb_train` | `TrainDataset` (ClinicDB part) | 550 | training |
| `kvasir` | `TestDataset/Kvasir` | 100 | source-domain test |
| `clinicdb` | `TestDataset/CVC-ClinicDB` | 62 | source-domain test |
| `colondb` | `TestDataset/CVC-ColonDB` | 380 | external test (main) |
| `etis` | `TestDataset/ETIS-LaribPolypDB` | 196 | external test (main) |
| `cvc300` | `TestDataset/CVC-300` | 60 | reference only, never pooled with `colondb` |

A count that differs from this table stops the audit. `image_id = <dataset>/<file stem>`.

### §1.2 Separating Kvasir and ClinicDB in TrainDataset
File stems that are pure digits belong to ClinicDB, all others to Kvasir (gives exactly 900/550).

### §1.3 Masks
Masks are read as grey levels and binarised with `value > 127` (Q-B1). Non-binary masks are counted for the data table, never edited.

### §1.4 Connected components
8-connectivity. Only components of at least 50 pixels produce a box prompt (Q-B2); the ground-truth mask used for the metrics is never changed, including 1-10 pixel annotation speckle.

### §1.5 Exclusion rule (near duplicates)
A ColonDB or ETIS image is excluded from the main analysis when some training image is within 6 bits Hamming distance on **both** a 64-bit pHash of the whole image and a pHash of the central 60% crop (Q-B10). With this rule no image was excluded. The whole-image hash alone flagged 19 ColonDB images; visual inspection of these pairs judged all of them different scenes (matches caused by the shared black octagonal frame and low-detail frames). No other image is excluded.

### §1.6 Data audit
`scripts/data/audit_dataset.py` writes `results/data_checks/audit_per_image.csv`, `dataset_characteristics.csv` (counts, sizes, mask quality, multi-polyp images, near duplicates) and `overlap_pairs.csv` (all pairs with whole-image pHash distance at most 10).

## §2. Seeds
`derive_seed(root, *labels)` takes the first 4 bytes of SHA-256 of `repr((root, *labels))`, so it does not depend on machine or call order. `ROOT_SEED = 20261029`.
- Prompts: `derive_seed(ROOT_SEED, "prompts", image_id)`; perturbed prompts: `("prompt_perturb", image_id, box_index)`.
- Validation split: `("val_split", dataset)`. Bootstrap: `("bootstrap", dataset)`, the same index matrix for every configuration.
- Training seeds {0, 1, 2} redraw the minibatch order and, for A2, the LoRA A initialisation. The decoder always starts from the pretrained SAM weights. Non-deterministic GPU kernels add further variation (§9.3).

## §3. Box prompts (oracle boxes)
Each component of §1.4 gives a tight box `[x0, y0, x1, y1]` (inclusive pixels). Every side is enlarged by `u * size` with `u ~ U(0, 0.10)` independently per side, rounded and clipped to the image. Boxes are generated **once** for every image and stored in `data/prompts/prompts.json` (with the tight boxes); every configuration reads the same file. These boxes are derived from the ground truth, so all results are **oracle-box** results, not those of an automatic system.

## §4. Validation split
10% of the training images, stratified by source (90 Kvasir, 55 ClinicDB), drawn once and stored in `data/splits/val_split.json`. No test image is used for any decision.

## §5. Configurations

### §5.1 Shared by all configurations
SAM ViT-B (`sam_vit_b_01ec64.pth`, sha256 checked on every run); SAM preprocessing (`ResizeLongestSide(1024)`, pixel normalisation, padding); no augmentation; `multimask_output=False`; low-resolution logits are upsampled with `postprocess_masks`, passed through a sigmoid to give `p`, and the mask is `p >= 0.5`. Several boxes in one image: each box is decoded separately, probabilities are merged by pixel-wise maximum, and the image-level predicted IoU is the mean over boxes. Fairness rule: configurations differ only in which weights are updated.

### §5.2 The four configurations
| Config | Updated weights | Seeds |
|---|---|---|
| A0 | none (zero-shot) | deterministic |
| A1 | mask decoder (encoder embeddings cached) | 0, 1, 2 |
| A2 | LoRA in the image encoder + mask decoder | 0, 1, 2 |
| A3 | none; ensemble of the three A2 seeds | - |

### §5.3 LoRA
LoRA on the q and v branches of the fused `qkv` projection in all 12 encoder blocks, rank 4, scale 1, A Kaiming-uniform, B zero (following SAMed; the code is an independent implementation).

### §5.4 A3
Pixel-wise mean of the three A2 probability maps (after the sigmoid); predicted IoU is the mean of the members. Pairwise Dice between member masks is an extra confidence score.

### §5.5 Parameter counts
Counted on the real architecture (`results/metrics/trainable_params.csv`); the paper reports these counts, not theoretical ones.

## §6. Training (A1, A2)
Loss = 20 focal (gamma 2, no alpha) + 1 Dice on 256x256 mask logits, plus 1 x MSE between predicted and true IoU (Q-B3). AdamW, learning rate 1e-4, weight decay 1e-4, cosine schedule per optimiser step to 0, no warm-up (Q-B4), 12 epochs, real batch 1 with gradient accumulation 4 (effective batch 4; batch 2 did not fit a T4 for A2, so both A1 and A2 use 1 x 4), fp16. Multi-polyp training images give one sample per box (Q-B5). The checkpoint with the highest mean validation Dice is kept (earliest on ties). Checkpoints are written every epoch.

## §7. Metrics

### §7.1 Image level
Dice and IoU with an empty prediction counted as 0. Failure = Dice < 0.5. Confidence scores: SAM predicted IoU (`conf_iou_pred`); mean confidence `max(p, 1-p)` inside the predicted mask (`conf_mean_inside`, 0 for an empty mask, Q-B6); for A3 the mean pairwise Dice of the members (`conf_pairwise_dice`).

### §7.2 Regional calibration (ECE)
Confidence `c = max(p, 1-p)`, correctness `[p >= 0.5] == y`. 15 equal bins on [0.5, 1] (Q-B7). Regions: `img` whole image; `roi` union of the tight ground-truth boxes enlarged by 10% of the box size on every side (Q-B8); `bnd` pixels within 1% of the image diagonal of the ground-truth boundary. ECE of a dataset is **pooled over pixels**: per-image bin statistics are summed, so images with large regions weigh more (see §7.9). The regions use the ground truth, so this is an offline evaluation.

### §7.3 Failure detection
AUROC with failures as positives and the negated confidence as score (undefined, reported as N/A, without failures). AURC: images sorted by decreasing confidence (ties by `image_id`), risk at coverage k = mean (1 - Dice) of the k most confident images, AURC = mean risk over k. AURC also depends on the overall error level, so a lower AURC is not by itself better failure detection.

### §7.4 Statistics
Image-level bootstrap, B = 1000, 95% percentile intervals (Q-B9), per dataset. Paired comparisons reuse the same resampled indices for both configurations. A1 and A2 are compared seed by seed (Q-B11). Seed variation (mean and SD over 3 seeds) is reported separately from the bootstrap intervals. No multiple-comparison correction.

### §7.5 Metric names
`dice, iou, ece_img, ece_roi, ece_bnd, auroc_<score>, aurc_<score>, spearman_<score>, n_failures, n_images`; secondary: `n_failures_07, auroc07_<score>`.

### §7.6 Sensitivity analysis: failure threshold 0.7
Added after the zero-shot results showed 1 failure out of 196 ETIS images, and before comparing configurations: `n_failures_07`, `auroc07_<score>`, written to `*_secondary.csv`.

### §7.7 Change of the main RQ2 metrics (made after seeing part of the results)
After fine-tuning almost no image had Dice < 0.5, so AUROC could not be computed for most configurations. **After seeing these failure counts on the test sets**, the main RQ2 metrics were changed to AURC and the Spearman correlation between confidence and Dice. AUROC is still computed and reported with the number of failures, but not used for conclusions. This change was not part of the original plan and is reported as such in the paper. Both metrics mainly measure how well confidence ranks images by segmentation quality.

### §7.8 Reading paired comparisons over seeds
A difference is called **consistent** only when the paired 95% intervals of all three seed pairs exclude 0 with the same sign (A3 versus each A2 seed in the same way). Otherwise it is reported as not consistent.

### §7.9 Per-image ECE (sensitivity analysis)
Because pooled ECE weights images by region size, ECE is also computed per image from each image's own bin statistics for `roi` and `bnd` on ColonDB and ETIS. Median, IQR and mean are reported, and the mean per-image ECE is compared with the rule of §7.8. Pooled ECE remains the main number.

## §8. Result files
| File | Content |
|---|---|
| `results/metrics/per_image/<run_id>.csv` | one row per test image: run, dataset, sizes, Dice, IoU, failure flag, confidence scores |
| `results/metrics/ece_bins/<run_id>.npz` | per-image ECE bin statistics `[n_images, 3 regions, 15 bins, (count, sum_conf, sum_correct)]` |
| `results/metrics/summary.csv` | value and bootstrap CI per configuration, seed, dataset, metric |
| `results/metrics/seed_stability.csv` | mean and SD over seeds (A1, A2) |
| `results/metrics/paired.csv`, `paired_consistency.csv` | paired differences and the §7.8 reading |
| `results/metrics/cost.csv` | training time, inference time and memory |
| `results/metrics/perturb/` | prompt-perturbation runs and tables (§14) |

Probability maps are not part of this repository.

## §9. Runs, manifests, environment

### §9.1 Run identifiers and manifests
`run_id = <config>_s<seed>_<commit7>_<yyyymmdd-hhmm>` (UTC). Each run has a manifest with the commit, the sha256 of the config, prompts, split, exclusion list and SAM checkpoint, and its start and end times. The commit hashes refer to the private development repository in which the runs were made; the Kaggle job checks that the uploaded code matches that commit before running.

### §9.2 Environment
`results/logs/environment/<run_id>.json`: Python, PyTorch, CUDA, GPU, library versions (hostname stored only as a hash).

### §9.3 Determinism
`cudnn.benchmark = False`, deterministic algorithms where available. Some GPU operations remain non-deterministic; this variation is part of the seed variation.

### §9.4 Library versions
Analysis and tests: `requirements-local.txt` (exact pins). Kaggle: the image's PyTorch plus `requirements-kaggle.txt`; the image versions were locked in `configs/env_kaggle.lock.json` after a first test job and checked by every later job.

## §10. Computational cost
Training time per run (GPU-hours = wall-clock hours, one GPU used). Inference cost in one dedicated job on one Tesla T4: ETIS images 6 to 55 by `image_id` (the first 5 are warm-up), batch 1, timed from an image in memory to the full-size probability map, for A0, A1, A2 and A3 (A3 = three full A2 passes plus averaging); peak memory from `torch.cuda.max_memory_allocated`.

## §11. Tests
Tests never write to `results/`: a fixture redirects the results directory, and a static check scans the test sources. Tests that need the real data are skipped without it; structural tests use a tiny fake dataset (`samrel.mock`).

## §12. Implementation details fixed before any model result
| Code | Point | Value used |
|---|---|---|
| Q-B1 | mask binarisation | `> 127` |
| Q-B2 | connectivity; box components | 8-connectivity; boxes only for components of at least 50 px |
| Q-B3 | focal gamma/alpha; IoU MSE weight; loss resolution | gamma 2, no alpha; 1.0; 256x256 |
| Q-B4 | learning-rate schedule | cosine per step, no warm-up |
| Q-B5 | multi-polyp training images | one sample per box |
| Q-B6 | confidence for empty masks | `conf_mean_inside = 0`; pairwise Dice of two empty masks = 1 |
| Q-B7 | ECE bins | 15 equal bins on [0.5, 1] |
| Q-B8 | ROI enlargement | 10% per side (20% in total per axis) |
| Q-B9 | bootstrap interval | percentile |
| Q-B10 | near-duplicate exclusion | whole-image and centre-crop pHash both at most 6 |
| Q-B11 | pairing with three seeds | seed by seed (A1-A0, A2-A0, A2-A1 same seed, A3 versus each A2 seed) |

## §13. Example figure
Images are chosen by rule, not by eye, on ColonDB and ETIS: `typical` (A0 Dice closest to the A0 median) and `overconfident` (largest `conf_iou_pred - Dice` for A2 seed 0); A1 and A2 are shown with seed 0, fixed in advance.

## §14. Prompt perturbation
Fixed before generating any perturbed prompt and before any inference with them. It tests whether the RQ2 findings depend on oracle boxes; it is not a new research question.

### §14.1 Perturbation levels
Perturbation is applied to the oracle box of §3 (so sigma = 0 is the oracle condition). For each box, four numbers `u ~ U(-1, 1)` are drawn once; at level sigma the centre moves by `sigma * u_x * w` and `sigma * u_y * h` and the size is multiplied by `1 + sigma * u_w` and `1 + sigma * u_h` (shrink or enlarge), then rounded and clipped to the image. Levels: P0 oracle, P1 sigma = 0.10, P2 sigma = 0.20, P3 sigma = 0.30, with the same `u` at every level. Each box of a multi-polyp image is perturbed independently; no box is added or removed. Levels are described by the IoU between perturbed and oracle boxes, computed from the prompt files only.

### §14.2 Generation
Generated once into `data/prompts/prompts_pp10.json`, `prompts_pp20.json`, `prompts_pp30.json`, for ColonDB and ETIS. The tight ground-truth boxes (used for the ECE `roi` region) are unchanged.

### §14.3 Models
The existing A0, A1 (3 seeds) and A2 (3 seeds) weights, no retraining; A3 is the mean of the three A2 members at the same level. Each image is encoded once and decoded for P0 to P3. P0 must reproduce the original runs (maximum absolute difference 1e-4, otherwise the job stops).

### §14.4 Test sets and metrics
ColonDB and ETIS only. Dice, IoU, failures at 0.5 and 0.7, AUROC with counts, AURC and Spearman rho for `conf_iou_pred` and `conf_mean_inside` (A3 also `conf_pairwise_dice`). ECE is computed and stored but not used for conclusions.

### §14.5 Reading rules
Paired bootstrap as in §7.4, same indices for every configuration and level; consistency over seeds as in §7.8.
1. Effect of perturbation within a configuration: `Delta = m(c, P_k) - m(c, P0)`.
2. More or less sensitive than zero-shot: `DiD = Delta(c) - Delta(A0)` on absolute scales. For Dice, IoU and rho, a consistent positive DiD means less sensitive and a consistent negative DiD more sensitive (reversed for AURC); otherwise "not distinguishable". Relative drops are descriptive only.
3. Does the RQ2 finding hold: at every level, the four comparison families of `paired_consistency.csv` (`a1_vs_a0`, `a2_vs_a0`, `a2_vs_a1`, `a3_vs_a2`) on AURC and rho are classified as consistently higher, consistently lower or not consistent, and compared with P0.
4. All three levels are reported; no level is chosen as the main one; monotonic trends are not required.

### §14.6 Cost
About 1.5 GPU-hours on a Tesla T4 (two parallel jobs).

### §14.7 Outputs
`results/metrics/perturb/`: `prompt_levels.csv`, per-run `per_image/` and `ece_bins/`, `runs_*.csv`, `summary.csv`, `delta.csv`, `did.csv`, `rq2_pairs.csv`, `*_consistency.csv`, `p0_check_*.csv`. Run ids: `<config>-pp<NN>_s<seed>_<commit7>_<stamp>`. Figure: `results/figures/fig_perturb.*`.

### §14.8 When AUROC may be used
A cell (configuration, seed, dataset, level, failure threshold 0.5 or 0.7) supports a conclusion from AUROC only when both the failure and the success group contain at least 20 images (`auroc_eligible` in `summary.csv`). Otherwise AUROC is reported with the counts only. Delta needs both cells eligible, DiD all four, and a between-configuration comparison both; for A1 and A2 all three seeds must be eligible.
