# Post-thesis TRAIN pilot: v1 versus v2 — 2026-09-19

**Development analysis, excluded from the submitted thesis. No held-out test
performance or cross-domain improvement is established.** v2 was designed after
inspecting v1 outcomes on these same 50 examples. This is not an independent
validation set, and the bundled prompt changes do not isolate one causal factor.

## Evidence and execution

The operator supplied both the v2 execution summary and the 50 paired predictions
in [the accompanying CSV](ragtruth_pilot_v1_v2_20260919.csv). IDs, labels, task types
and v1 predictions match the [preserved v1 report](ragtruth_pilot_v1_20260919.md)
and its CSV. Metrics below are calculated from the supplied rows. Raw cluster
journals and response payloads were not independently downloaded.

- Scoring commit: `662af1dd2e94059f6cdb6f5eacf14c84577fe293`.
- Run ID: `qwen3-ragtruth-train-pilot-50-v2`.
- Prompt: `faithfulness-development-v2`; SHA256
  `7771610009b40b5cce476fa4abda9ecb6bd025ecf71d635d7d3e7feb3c70135a`.
- Token audit SHA256:
  `fb28d1bab2bc2cd64d650af26dce5fad8c7374b5997acf8d6afe720d59bbeaae`.
- Existing preparation manifest SHA256:
  `ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
- Model, input selection and decoding remain unchanged from v1. See the
  [frozen v2 execution plan](../../../post_thesis/llm_judge/configs/ragtruth_pilot_50_v2.json).

The operator reported 50/50 scores, 50 new attempts, zero terminal failures or
pending examples, and complete token usage: 62,674 input + 504 output = 63,178.
Charged client time was 62.199092883151025 seconds; unused allowance was
537.800907116849 seconds. This includes client preflight and bookkeeping, excludes
server startup/idle time, and is not a controlled speed comparison with v1's
111.38075984921306 seconds. Warmup/server state and per-attempt timings have not
been matched. No rental-cost figure is available. Unused allowance does not
authorize another run or repeat stability experiment.

## Paired descriptive results

Higher scores mean more likely unsupported. Positive-label prevalence is 24/50
(0.48); the pilot was not label-balanced and is not a row-uniform benchmark sample.

| Metric | v1 | v2 |
| --- | ---: | ---: |
| AUROC | 0.6810897435897435 | 0.7564102564102564 |
| Average precision (stepwise PR area) | 0.606881427707199 | 0.7009933574879226 |
| Brier score (lower is better) | 0.37335 | 0.30830 |
| Mean predicted probability | 0.099 | 0.166 |
| Mean among positive labels | 0.12916666666666668 | 0.2375 |
| Mean among negative labels | 0.07115384615384615 | 0.1000 |

AUROC increased by 0.0753205128205128 and average precision by
0.0941119297807236. This indicates better ranking on the inspected development
sample, rather than only a uniform upward probability shift. No statistical
significance or held-out improvement is claimed. Brier score improved, but the
probabilities still substantially understate the observed positive rate.

| v2 score | Label 0 | Label 1 | Total |
| --- | ---: | ---: | ---: |
| 0.0 | 6 | 1 | 7 |
| 0.05 | 3 | 0 | 3 |
| 0.1 | 10 | 7 | 17 |
| 0.2 | 6 | 11 | 17 |
| 0.25 | 1 | 2 | 3 |
| 0.7 | 0 | 1 | 1 |
| 0.8 | 0 | 2 | 2 |

23 scores increased, one decreased and 26 were unchanged. 21 of the 24 positives
still score at or below 0.25. At the **illustrative, unfitted** threshold 0.5, v2
has TN=26, FP=0, FN=21 and TP=3 (recall 0.125); v1 had TN=26, FP=0, FN=24 and
TP=0. This is not threshold selection. No threshold search or calibration fitting
was performed, and no final benchmark metrics are reported.

## Evidence review

Selection for this review was label-informed: inspect all three examples crossing
0.5 and revisit the five v1 positives scored zero. This is not blinded annotation.
The source observations below were checked against the pinned TRAIN parquet.

| ID | v1 → v2 | Observation |
| --- | --- | --- |
| 15849 | 0.2 → 0.8 | Incorrect passage attributions and a vaccine-specific assertion unsupported by the supplied passages. |
| 6946 | 0.2 → 0.8 | Multiple explicit conflicts about business attributes, including WiFi, reservations and parking. |
| 9256 | 0.2 → 0.7 | Unknown WiFi, reservation and takeout fields become factual availability claims; review-platform attribution is also unsupported. |
| 6040 | 0.0 → 0.1 | The previously identified unsupported sandwich addition remains weakly scored. |
| 6486 | 0.0 → 0.1 | Unknown WiFi/music values still become negative availability claims. |
| 12052 | 0.0 → 0.1 | The lowest-to-highest numerical relationship error remains weakly scored. |
| 9350 | 0.0 → 0.0 | The answer asserts outdoor seating despite an unknown field and no support in the reviews. |
| 12892 | 0.0 → 0.1 | The prior concern about an oil-related annotated span remains unresolved; this does not adjudicate the whole answer. |

The three high scores align with genuine evidence problems, but a one-field
probability response does not reveal which claim the judge detected. Original
labels remain unchanged. Disputed annotations are not removed to improve metrics.

## Decision

Retain v2 as the current development candidate, without declaring the final
prompt frozen. Do not immediately add a v3 prompt or expand to benchmark tests.
Prepare the [separate synthetic diagnostic](../../../post_thesis/llm_judge/DIAGNOSTIC.md)
to examine unknown attributes and numerical direction in short versus embedded
claims. It keeps v2 and the serving profile fixed, has its own budget, and uses
manually authored text rather than benchmark answers. Its outcomes can localize
failures on those constructed cases but cannot establish generalization or
fully explain model internals. No diagnostic GPU results are available yet.
