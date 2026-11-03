"""Fixed constants (PROTOCOL §1.1, §2, §7.5)."""

from __future__ import annotations

ROOT_SEED = 20261029  # PROTOCOL §2

# dataset key -> (role, expected image count); PROTOCOL §1.1
DATASETS: dict[str, tuple[str, int]] = {
    "kvasir_train": ("train", 900),
    "clinicdb_train": ("train", 550),
    "kvasir": ("source_test", 100),
    "clinicdb": ("source_test", 62),
    "colondb": ("external_test", 380),
    "etis": ("external_test", 196),
    "cvc300": ("reference", 60),
}
TRAIN_DATASETS = ("kvasir_train", "clinicdb_train")
TEST_DATASETS = ("kvasir", "clinicdb", "colondb", "etis", "cvc300")
PRIMARY_DATASETS = ("colondb", "etis")
# Only these may lose images to the near-duplicate rule (PROTOCOL §1.5).
EXCLUSION_DATASETS = ("colondb", "etis")

CONFIGS = ("a0", "a1", "a2", "a3")
REGIONS = ("img", "roi", "bnd")
BIN_COLUMNS = ("count", "sum_conf", "sum_correct")

METRIC_NAMES = (
    "dice", "iou", "ece_img", "ece_roi", "ece_bnd",
    "auroc_iou_pred", "auroc_mean_inside", "auroc_pairwise_dice",
    "aurc_iou_pred", "aurc_mean_inside", "aurc_pairwise_dice",
    "spearman_iou_pred", "spearman_mean_inside", "spearman_pairwise_dice",  # main metric since §7.7
    "n_failures", "n_images",
)
# Secondary metrics (sensitivity analysis, PROTOCOL §7.6), reported in a separate table only.
SECONDARY_METRIC_NAMES = ("n_failures_07", "auroc07_iou_pred", "auroc07_mean_inside", "auroc07_pairwise_dice")

PER_IMAGE_COLUMNS = (
    "run_id", "config", "seed", "dataset", "image_id", "img_h", "img_w", "n_boxes",
    "gt_area", "pred_area", "dice", "iou", "failed",
    "conf_iou_pred", "conf_mean_inside", "conf_pairwise_dice", "excluded",
)
SUMMARY_COLUMNS = (
    "config", "seed", "dataset", "metric", "value", "ci_low", "ci_high",
    "n_images", "n_failures", "run_ids", "commit",
)
PAIRED_COLUMNS = ("config_a", "config_b", "dataset", "metric", "diff", "ci_low", "ci_high", "excludes_zero", "B")
SEED_STABILITY_COLUMNS = ("config", "dataset", "metric", "mean", "sd", "n_seeds", "run_ids")
PARAM_COLUMNS = (
    "config", "n_total", "n_trainable", "n_trainable_lora", "n_trainable_decoder",
    "n_trainable_other", "lora_targets", "lora_rank", "commit",
)
COST_COLUMNS = (
    "run_id", "config", "seed", "phase", "gpu_name", "gpu_count", "amp", "batch_size", "n_images",
    "wall_seconds", "gpu_hours", "sec_per_image_mean", "sec_per_image_median", "peak_gpu_mem_mb", "commit",
)
TRAIN_LOG_COLUMNS = (
    "epoch", "step", "train_loss", "train_focal", "train_dice", "train_iou_mse",
    "val_dice", "lr", "seconds", "peak_mem_mb",
)
AUDIT_PER_IMAGE_COLUMNS = (
    "dataset", "role", "image_id", "image_file", "mask_file", "img_h", "img_w", "mask_h", "mask_w",
    "size_mismatch", "mask_unique_values", "mask_nonbinary", "mask_empty", "n_components",
    "n_small_components", "n_box_components", "fg_fraction", "phash", "phash_center",
)
DATASET_CHARACTERISTICS_COLUMNS = (
    "dataset", "role", "n_expected", "n_images", "n_masks", "n_images_without_mask",
    "n_masks_without_image", "count_ok", "size_min_hw", "size_max_hw", "size_mode_hw",
    "n_distinct_sizes", "n_mask_nonbinary", "n_mask_empty", "n_size_mismatch", "n_multi_polyp",
    "max_components", "n_small_components", "fg_fraction_median", "n_near_dup_within",
    "n_near_dup_in_train", "n_near_dup_in_train_center", "n_excluded",
)
OVERLAP_PAIR_COLUMNS = ("dataset_a", "image_a", "dataset_b", "image_b", "hamming", "hamming_center")
MANIFEST_KEYS = (
    "run_id", "config", "seed", "commit", "commit_dirty", "config_sha256", "prompts_sha256",
    "val_split_sha256", "excluded_sha256", "data_hash", "checkpoint_sha256", "kaggle_account",
    "kernel_slug", "started_utc", "finished_utc", "status",
)
