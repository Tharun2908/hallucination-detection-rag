# Post-thesis calibration and threshold fit — 2026-09-20

The operator reports successful read-only replay of both complete reserved scoring
arms, followed by one accepted fit at revision
`c22ae9257eada6220875895a5f41fac15c308e43`. Cached inspection did not fit again and
returned the identical report SHA256:
`2885eb400a8b5662ffd30382926773b85bbcf5055c1144394d893ecc072610f9`.
The [console-derived JSON record](development_fit_cluster_20260920.json) retains
these findings. The author has not received the full private fit artifact or
independently reconstructed its hash. The read-only freeze checker verifies it
on the pod before subsequent use.

This is post-thesis development work, outside the submitted thesis. No TEST or
HaluBench inference, model calls, model/prompt changes or post-outcome adjustment
to the preregistered fitting rules occurred.

## Accepted calibration

For raw unsupported log-odds `m`, use:

```text
p_cal = sigmoid(0.3808096416654668 * m + 0.3575745057816423)
```

The optimizer succeeded in eight iterations and ten function evaluations. The
objective decreased from 0.7540429127157294 to 0.5577270851005413. The largest
absolute analytic gradient is about 2.09e-10, and both parameters are safely
inside the registered bounds. All acceptance checks passed; no fallback fit
was needed.

The slope below one compresses log-odds and the positive intercept shifts them
upward. This supports describing the raw scores as too extreme on these
calibration data, not claiming that calibration will generalize to every domain.
A positive-slope map preserves mathematical ranking. Use raw margins for AUROC
and average precision to avoid sigmoid saturation and rounding effects.

| Arm | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| Calibration, 600 rows | 0.224561 | 0.189173 | 0.178012 | 0.040184 |
| Operating threshold, 600 rows | 0.222392 | 0.199611 | 0.160977 | 0.058079 |

Calibration-arm values are in-sample diagnostics. The separate threshold arm
shows improved reliability under this fixed map, but remains development data
used to select the operating point. These are not independent final-pipeline
validation or cross-domain findings. No uncertainty intervals were computed from
these console summaries.

## Frozen operating point

Predict unsupported when **raw margin >= -1.25**. Preserve the exact float
`-0x1.4000000000000p+0`; do not classify using rounded calibrated probabilities.
The transformed threshold is approximately 0.470425177, for description only.

On the threshold-selection arm: TP 209, FP 104, FN 67, TN 220; F1 0.709677,
precision 0.667732, recall 0.757246, accuracy 0.715. These counts sum to 600 and
match the reserved 276 positive / 324 negative labels. F1 is the exact ratio
418/589. It is selection performance, not a test estimate or an optimal deployment
claim. No threshold was chosen using TEST outcomes.

## Next step

Hold the [judge decision configuration](../../../post_thesis/llm_judge/configs/frozen_judge_v1.json)
fixed. Apply the [evaluation protocol](../../../post_thesis/llm_judge/EVALUATION_PROTOCOL.md)
when building benchmark and baseline alignment manifests. Do not refit on either
benchmark. The HaluBench canonical 8k split remains unchanged.
