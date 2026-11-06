# Spearman rho between confidence and Dice, by dataset, perturbation level, score and seed

A0: rho [95% bootstrap interval]. Other rows: rho; paired difference from A0 [95% paired bootstrap interval], then a mark: **higher** or **lower** when the interval excludes 0, **undetermined** when it contains 0. Undetermined does not mean equivalent. Intervals are per image, not adjusted for multiple comparisons, and do not account for frames from the same video.

A larger rho means better ranking of images by segmentation quality. A3 versus A0 and mean pixel confidence under perturbed boxes are exploratory analyses added after the results were seen.

## CVC-ColonDB, IoU-head score

| Run | P0 (oracle) | P1 (0.10) | P2 (0.20) | P3 (0.30) |
|---|---|---|---|---|
| A0 | 0.724 [0.668, 0.777] | 0.682 [0.611, 0.744] | 0.706 [0.644, 0.764] | 0.617 [0.542, 0.686] |
| A1 seed 0 | 0.646; -0.079 [-0.155, -0.008] lower | 0.633; -0.049 [-0.129, +0.031] undetermined | 0.550; -0.156 [-0.233, -0.075] lower | 0.555; -0.063 [-0.159, +0.029] undetermined |
| A1 seed 1 | 0.595; -0.129 [-0.215, -0.047] lower | 0.602; -0.080 [-0.168, +0.001] undetermined | 0.556; -0.150 [-0.236, -0.068] lower | 0.595; -0.022 [-0.120, +0.072] undetermined |
| A1 seed 2 | 0.617; -0.107 [-0.189, -0.028] lower | 0.618; -0.064 [-0.155, +0.021] undetermined | 0.593; -0.113 [-0.192, -0.041] lower | 0.615; -0.002 [-0.096, +0.088] undetermined |
| A2 seed 0 | 0.707; -0.017 [-0.093, +0.054] undetermined | 0.660; -0.022 [-0.104, +0.057] undetermined | 0.561; -0.145 [-0.232, -0.062] lower | 0.577; -0.040 [-0.137, +0.058] undetermined |
| A2 seed 1 | 0.671; -0.053 [-0.129, +0.017] undetermined | 0.657; -0.025 [-0.110, +0.058] undetermined | 0.557; -0.149 [-0.238, -0.068] lower | 0.571; -0.046 [-0.145, +0.046] undetermined |
| A2 seed 2 | 0.668; -0.056 [-0.138, +0.022] undetermined | 0.661; -0.022 [-0.106, +0.062] undetermined | 0.590; -0.116 [-0.199, -0.041] lower | 0.607; -0.010 [-0.103, +0.083] undetermined |
| A3 | 0.717; -0.007 [-0.078, +0.064] undetermined | 0.702; +0.019 [-0.060, +0.098] undetermined | 0.606; -0.100 [-0.181, -0.024] lower | 0.620; +0.003 [-0.087, +0.094] undetermined |
| **A1 reading** | lower (0 higher, 3 lower of 3) | not consistent (0 higher, 0 lower of 3) | lower (0 higher, 3 lower of 3) | not consistent (0 higher, 0 lower of 3) |
| **A2 reading** | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) | lower (0 higher, 3 lower of 3) | not consistent (0 higher, 0 lower of 3) |
| **A3 reading (one run)** | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | lower (0 higher, 1 lower of 1) | not consistent (0 higher, 0 lower of 1) |

## CVC-ColonDB, mean pixel confidence

| Run | P0 (oracle) | P1 (0.10) | P2 (0.20) | P3 (0.30) |
|---|---|---|---|---|
| A0 | 0.458 [0.367, 0.544] | 0.463 [0.374, 0.544] | 0.500 [0.417, 0.578] | 0.462 [0.369, 0.544] |
| A1 seed 0 | 0.483; +0.025 [-0.077, +0.124] undetermined | 0.450; -0.013 [-0.119, +0.090] undetermined | 0.433; -0.067 [-0.160, +0.037] undetermined | 0.415; -0.047 [-0.151, +0.065] undetermined |
| A1 seed 1 | 0.465; +0.006 [-0.093, +0.105] undetermined | 0.418; -0.045 [-0.149, +0.059] undetermined | 0.430; -0.070 [-0.163, +0.039] undetermined | 0.455; -0.007 [-0.110, +0.108] undetermined |
| A1 seed 2 | 0.503; +0.044 [-0.061, +0.148] undetermined | 0.506; +0.043 [-0.055, +0.145] undetermined | 0.517; +0.017 [-0.080, +0.116] undetermined | 0.495; +0.033 [-0.070, +0.147] undetermined |
| A2 seed 0 | 0.464; +0.006 [-0.104, +0.116] undetermined | 0.467; +0.004 [-0.105, +0.114] undetermined | 0.461; -0.039 [-0.142, +0.063] undetermined | 0.467; +0.006 [-0.101, +0.121] undetermined |
| A2 seed 1 | 0.571; +0.113 [+0.009, +0.211] higher | 0.505; +0.042 [-0.059, +0.144] undetermined | 0.513; +0.013 [-0.094, +0.123] undetermined | 0.535; +0.074 [-0.035, +0.192] undetermined |
| A2 seed 2 | 0.443; -0.015 [-0.141, +0.099] undetermined | 0.394; -0.068 [-0.184, +0.044] undetermined | 0.426; -0.074 [-0.183, +0.039] undetermined | 0.467; +0.006 [-0.098, +0.124] undetermined |
| A3 | 0.556; +0.098 [-0.010, +0.198] undetermined | 0.497; +0.034 [-0.076, +0.144] undetermined | 0.532; +0.032 [-0.071, +0.138] undetermined | 0.573; +0.111 [+0.004, +0.228] higher |
| **A1 reading** | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) |
| **A2 reading** | not consistent (1 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 0 lower of 3) |
| **A3 reading (one run)** | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | higher (1 higher, 0 lower of 1) |

## ETIS-LaribPolypDB, IoU-head score

| Run | P0 (oracle) | P1 (0.10) | P2 (0.20) | P3 (0.30) |
|---|---|---|---|---|
| A0 | 0.743 [0.668, 0.798] | 0.738 [0.656, 0.803] | 0.718 [0.623, 0.787] | 0.738 [0.656, 0.796] |
| A1 seed 0 | 0.613; -0.130 [-0.234, -0.026] lower | 0.563; -0.175 [-0.287, -0.067] lower | 0.456; -0.262 [-0.393, -0.136] lower | 0.475; -0.263 [-0.395, -0.138] lower |
| A1 seed 1 | 0.652; -0.090 [-0.182, -0.005] lower | 0.613; -0.125 [-0.221, -0.035] lower | 0.566; -0.152 [-0.250, -0.055] lower | 0.541; -0.197 [-0.301, -0.091] lower |
| A1 seed 2 | 0.658; -0.085 [-0.166, -0.008] lower | 0.554; -0.184 [-0.284, -0.085] lower | 0.468; -0.250 [-0.365, -0.140] lower | 0.537; -0.201 [-0.320, -0.093] lower |
| A2 seed 0 | 0.666; -0.077 [-0.178, +0.015] undetermined | 0.621; -0.117 [-0.223, -0.009] lower | 0.528; -0.190 [-0.318, -0.072] lower | 0.534; -0.204 [-0.322, -0.071] lower |
| A2 seed 1 | 0.662; -0.080 [-0.182, +0.016] undetermined | 0.594; -0.144 [-0.256, -0.031] lower | 0.534; -0.183 [-0.308, -0.069] lower | 0.583; -0.155 [-0.255, -0.046] lower |
| A2 seed 2 | 0.687; -0.056 [-0.140, +0.032] undetermined | 0.633; -0.105 [-0.209, -0.006] lower | 0.554; -0.164 [-0.285, -0.051] lower | 0.566; -0.172 [-0.286, -0.054] lower |
| A3 | 0.705; -0.038 [-0.126, +0.050] undetermined | 0.669; -0.069 [-0.161, +0.029] undetermined | 0.570; -0.148 [-0.263, -0.035] lower | 0.598; -0.140 [-0.241, -0.025] lower |
| **A1 reading** | lower (0 higher, 3 lower of 3) | lower (0 higher, 3 lower of 3) | lower (0 higher, 3 lower of 3) | lower (0 higher, 3 lower of 3) |
| **A2 reading** | not consistent (0 higher, 0 lower of 3) | lower (0 higher, 3 lower of 3) | lower (0 higher, 3 lower of 3) | lower (0 higher, 3 lower of 3) |
| **A3 reading (one run)** | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | lower (0 higher, 1 lower of 1) | lower (0 higher, 1 lower of 1) |

## ETIS-LaribPolypDB, mean pixel confidence

| Run | P0 (oracle) | P1 (0.10) | P2 (0.20) | P3 (0.30) |
|---|---|---|---|---|
| A0 | 0.489 [0.377, 0.596] | 0.507 [0.394, 0.607] | 0.533 [0.422, 0.642] | 0.626 [0.520, 0.716] |
| A1 seed 0 | 0.363; -0.126 [-0.253, -0.012] lower | 0.345; -0.162 [-0.291, -0.049] lower | 0.460; -0.073 [-0.192, +0.044] undetermined | 0.463; -0.163 [-0.274, -0.044] lower |
| A1 seed 1 | 0.325; -0.164 [-0.289, -0.048] lower | 0.366; -0.140 [-0.249, -0.031] lower | 0.416; -0.117 [-0.244, +0.007] undetermined | 0.403; -0.223 [-0.354, -0.094] lower |
| A1 seed 2 | 0.422; -0.067 [-0.193, +0.055] undetermined | 0.395; -0.112 [-0.234, +0.007] undetermined | 0.410; -0.123 [-0.250, +0.007] undetermined | 0.520; -0.106 [-0.215, +0.008] undetermined |
| A2 seed 0 | 0.157; -0.331 [-0.457, -0.205] lower | 0.195; -0.312 [-0.436, -0.175] lower | 0.243; -0.290 [-0.439, -0.147] lower | 0.347; -0.278 [-0.400, -0.153] lower |
| A2 seed 1 | 0.343; -0.146 [-0.276, -0.019] lower | 0.439; -0.068 [-0.175, +0.040] undetermined | 0.485; -0.048 [-0.170, +0.074] undetermined | 0.578; -0.048 [-0.159, +0.065] undetermined |
| A2 seed 2 | 0.192; -0.297 [-0.428, -0.159] lower | 0.210; -0.297 [-0.426, -0.159] lower | 0.242; -0.291 [-0.438, -0.151] lower | 0.342; -0.283 [-0.412, -0.149] lower |
| A3 | 0.382; -0.107 [-0.216, +0.004] undetermined | 0.444; -0.063 [-0.190, +0.049] undetermined | 0.476; -0.057 [-0.195, +0.074] undetermined | 0.593; -0.033 [-0.147, +0.087] undetermined |
| **A1 reading** | not consistent (0 higher, 2 lower of 3) | not consistent (0 higher, 2 lower of 3) | not consistent (0 higher, 0 lower of 3) | not consistent (0 higher, 2 lower of 3) |
| **A2 reading** | lower (0 higher, 3 lower of 3) | not consistent (0 higher, 2 lower of 3) | not consistent (0 higher, 2 lower of 3) | not consistent (0 higher, 2 lower of 3) |
| **A3 reading (one run)** | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) | not consistent (0 higher, 0 lower of 1) |

## Counts over all cells

A cell is one dataset x score x level. A1, A2: higher or lower only when all three seeds agree. A3: a single paired interval.

| Dataset | Score | Config. | Cells | Cells higher | Cells lower | Cells not consistent | Runs higher | Runs lower | Runs undetermined | Point estimates above A0 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | all | A1 | 16 | 0 | 6 | 10 | 0 | 24 | 24 | 6/48 |
| all | all | A2 | 16 | 0 | 5 | 11 | 1 | 21 | 26 | 8/48 |
| all | all | A3 | 16 | 1 | 3 | 12 | 1 | 3 | 12 | 6/16 |
| colondb | all | A1 | 8 | 0 | 2 | 6 | 0 | 6 | 18 | 6/24 |
| colondb | all | A2 | 8 | 0 | 1 | 7 | 1 | 3 | 20 | 8/24 |
| colondb | all | A3 | 8 | 1 | 1 | 6 | 1 | 1 | 6 | 6/8 |
| etis | all | A1 | 8 | 0 | 4 | 4 | 0 | 18 | 6 | 0/24 |
| etis | all | A2 | 8 | 0 | 4 | 4 | 0 | 18 | 6 | 0/24 |
| etis | all | A3 | 8 | 0 | 2 | 6 | 0 | 2 | 6 | 0/8 |
| all | iou_pred | A1 | 8 | 0 | 6 | 2 | 0 | 18 | 6 | 0/24 |
| all | iou_pred | A2 | 8 | 0 | 4 | 4 | 0 | 12 | 12 | 0/24 |
| all | iou_pred | A3 | 8 | 0 | 3 | 5 | 0 | 3 | 5 | 2/8 |
| all | mean_inside | A1 | 8 | 0 | 0 | 8 | 0 | 6 | 18 | 6/24 |
| all | mean_inside | A2 | 8 | 0 | 1 | 7 | 1 | 9 | 14 | 8/24 |
| all | mean_inside | A3 | 8 | 1 | 0 | 7 | 1 | 0 | 7 | 4/8 |
| colondb | iou_pred | A1 | 4 | 0 | 2 | 2 | 0 | 6 | 6 | 0/12 |
| colondb | iou_pred | A2 | 4 | 0 | 1 | 3 | 0 | 3 | 9 | 0/12 |
| colondb | iou_pred | A3 | 4 | 0 | 1 | 3 | 0 | 1 | 3 | 2/4 |
| colondb | mean_inside | A1 | 4 | 0 | 0 | 4 | 0 | 0 | 12 | 6/12 |
| colondb | mean_inside | A2 | 4 | 0 | 0 | 4 | 1 | 0 | 11 | 8/12 |
| colondb | mean_inside | A3 | 4 | 1 | 0 | 3 | 1 | 0 | 3 | 4/4 |
| etis | iou_pred | A1 | 4 | 0 | 4 | 0 | 0 | 12 | 0 | 0/12 |
| etis | iou_pred | A2 | 4 | 0 | 3 | 1 | 0 | 9 | 3 | 0/12 |
| etis | iou_pred | A3 | 4 | 0 | 2 | 2 | 0 | 2 | 2 | 0/4 |
| etis | mean_inside | A1 | 4 | 0 | 0 | 4 | 0 | 6 | 6 | 0/12 |
| etis | mean_inside | A2 | 4 | 0 | 1 | 3 | 0 | 9 | 3 | 0/12 |
| etis | mean_inside | A3 | 4 | 0 | 0 | 4 | 0 | 0 | 4 | 0/4 |
