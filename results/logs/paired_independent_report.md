# Independent recomputation of paired comparisons

- Rows compared: 732 (540 from paired.csv, 192 from perturb/rq2_pairs.csv).
- Metrics: dice, iou, ece_img, ece_roi, ece_bnd, aurc_iou_pred, aurc_mean_inside, spearman_iou_pred, spearman_mean_inside.
- IoU = Dice/(2-Dice) and failed = Dice < 0.5 on every per-image row of the 8 runs: 0 violations.
- Largest absolute gap, point difference: 7.10e-16.
- Largest absolute gap, CI low: 1.16e-15; CI high: 4.76e-08.
- Rows whose sign class differs from the official one: 0.
- paired_consistency.csv `consistent` rows that differ: 0 of 180.

## Fresh index matrix, B = 10000 (external sets)

- Seed-level rows compared: 408; sign class changed: 3.
- Family readings compared: 136; changed: 1.

- rq2_pairs.csv a2_vs_a1 pp20 etis aurc_iou_pred: official -1, fresh matrix +0
