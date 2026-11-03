# Integrity checks: prompt perturbation (PROTOCOL §14)

| Check | Result | Detail |
|---|---|---|
| index: 32 runs | PASS | 32 |
| index: one commit | PASS | 85ca9c1b0dca78c001ac5a2c2df414526f5dcd64 |
| index: 8 runs per level | PASS | {'pp00': 8, 'pp10': 8, 'pp20': 8, 'pp30': 8} |
| index: prompt sha256 = files | PASS |  |
| prompts pp10: tight/oracle boxes unchanged, sigma | PASS |  |
| prompts pp20: tight/oracle boxes unchanged, sigma | PASS |  |
| prompts pp30: tight/oracle boxes unchanged, sigma | PASS |  |
| a0-pp00_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a0-pp00_s0_85ca9c1_20261102-1602: pp00 = original a0_s0_d60e848_20261029-1028 (max |diff| <= 0.0001) | PASS | 0.00e+00 |
| a0-pp10_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a0-pp20_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a0-pp30_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s0_85ca9c1_20261102-1602: pp00 = original a1_s0_d60e848_20261029-1027 (max |diff| <= 0.0001) | PASS | 6.93e-08 |
| a1-pp10_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp20_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp30_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s1_85ca9c1_20261102-1602: pp00 = original a1_s1_d60e848_20261029-1027 (max |diff| <= 0.0001) | PASS | 2.18e-07 |
| a1-pp10_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp20_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp30_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp00_s2_85ca9c1_20261102-1602: pp00 = original a1_s2_d60e848_20261029-1028 (max |diff| <= 0.0001) | PASS | 1.48e-07 |
| a1-pp10_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp20_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a1-pp30_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s0_85ca9c1_20261102-1602: pp00 = original a2_s0_d60e848_20261029-1027 (max |diff| <= 0.0001) | PASS | 4.01e-07 |
| a2-pp10_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp20_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp30_s0_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s1_85ca9c1_20261102-1602: pp00 = original a2_s1_d60e848_20261029-1028 (max |diff| <= 0.0001) | PASS | 1.05e-07 |
| a2-pp10_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp20_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp30_s1_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp00_s2_85ca9c1_20261102-1602: pp00 = original a2_s2_d60e848_20261029-1028 (max |diff| <= 0.0001) | PASS | 1.47e-07 |
| a2-pp10_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp20_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a2-pp30_s2_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a3-pp00_s012_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a3-pp00_s012_85ca9c1_20261102-1602: pp00 = original a3_s012_2f6ac15_20261029-1722 (max |diff| <= 0.0001) | PASS | 1.42e-07 |
| a3-pp10_s012_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a3-pp20_s012_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| a3-pp30_s012_85ca9c1_20261102-1602: 576 images, no NaN, GT/ROI same as original | PASS | n=576 nan_ok=True gt=True px=True |
| p0_check files: 8 runs checked in the kernels | PASS | 8 |
| a3-pp00_s012_85ca9c1_20261102-1602: predicted IoU = mean of 3 A2 members | PASS |  |
| a3-pp10_s012_85ca9c1_20261102-1602: predicted IoU = mean of 3 A2 members | PASS |  |
| a3-pp20_s012_85ca9c1_20261102-1602: predicted IoU = mean of 3 A2 members | PASS |  |
| a3-pp30_s012_85ca9c1_20261102-1602: predicted IoU = mean of 3 A2 members | PASS |  |
| pp00 summary (value, CI) = original summary.csv | PASS | 198 rows, max |diff| 2.52e-08 |

**53 PASS, 0 FAIL**
