# Post-thesis TRAIN pilot v1 — 2026-09-19

**Development evidence only. Not submitted-thesis or final benchmark results.**
The operator supplied the execution summary and 50 score/label rows in
[the accompanying CSV](ragtruth_pilot_v1_20260919.csv). Metrics below were computed
from those rows. IDs, task types and labels were checked against the pinned
TRAIN parquet; the raw cluster journal and per-response payloads were not
independently inspected for this report. Full answer/context text stays private.

## Provenance and execution

- Scoring commit: `b19b52d08c6eb342f95a01cd0187b3c594e7cf20`.
- Run: `qwen3-ragtruth-train-pilot-50-v1`.
- Prompt: `faithfulness-development-v1`, SHA256
  `e9e218764a4c7171f85468fd8f38ed698003f7538766ef692163f21cacc483c5`.
- Pilot manifest SHA256:
  `ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
- Token audit SHA256:
  `08dee554dbedde9c53354862af9d2f718c1f52895b33b32379fe4a9971a2b9f0`.
- Model/profile and resource limits:
  [frozen execution plan](../../../post_thesis/llm_judge/configs/ragtruth_pilot_50_v1.json).
  Qwen3-32B BF16, non-thinking, temperature 0, one H200, sequential requests.

The operator reported 50/50 valid scores, 50 new attempts, zero terminal failures
or pending rows, and complete token usage: 48,924 input + 477 output = 49,401.
Charged client execution was 111.38075984921306 seconds; remaining allowance was
488.61924015078694 seconds. This is a client window including preflight and
bookkeeping, not mean inference latency or total rented GPU time. Server startup
and idle time are separate. Rental cost and final server duration are not supplied.

## Descriptive development diagnostics

There are 24 positive and 26 negative labels. Selection used one response from
each of 50 hash-ranked context groups, without label balancing. These are prompt
development examples, not a new held-out test partition.

| Diagnostic | Value |
| --- | ---: |
| AUROC (higher score = unsupported) | 0.6810897435897435 |
| Average precision (stepwise PR area) | 0.606881427707199 |
| Positive-label prevalence | 0.48 |
| Mean predicted probability | 0.099 |
| Brier score | 0.37335 |
| Mean score among positive labels | 0.12916666666666668 |
| Mean score among negative labels | 0.07115384615384615 |

Only four score values appeared:

| Score | Label 0 | Label 1 | Total |
| --- | ---: | ---: | ---: |
| 0.0 | 13 | 5 | 18 |
| 0.05 | 3 | 0 | 3 |
| 0.1 | 3 | 7 | 10 |
| 0.2 | 7 | 12 | 19 |

At the **illustrative, unfitted** threshold 0.5, every prediction is negative:
TN=26, FP=0, FN=24, TP=0; recall is zero. This is not an operating threshold
selection. No threshold search or calibration fitting was performed. The narrow
score range retains some ranking signal but understates the positive rate on
this small sample. Neither a lower threshold nor an upward shift alone proves
better claim verification. Do not compare these development metrics directly
with thesis metrics on different test examples.

## Manual inspection: all five positive-label examples scored zero

Selection rule for this review: inspect every label-1 example with score exactly
zero after seeing the pilot results. This is development error analysis, not a
blinded independent annotation or an exhaustive check of all 50 examples.

| ID | Observation from the pinned TRAIN evidence/annotations |
| --- | --- |
| 6040 | The answer introduces sandwiches, unsupported by the supplied business data/reviews. |
| 6486 | Missing WiFi/music values become claims that those amenities are unavailable. Missing data does not establish absence. |
| 9350 | A missing outdoor-seating value becomes a claim that outdoor seating is offered. |
| 12052 | The answer changes a numerical relationship from lowest 75% to highest 75%. |
| 12892 | The oil-related annotated span appears supported by the supplied passage. This raises an annotation concern about that span; it does not adjudicate the entire answer, which may contain other issues. |

Original labels remain unchanged. Log potential annotation-definition or
annotation-quality issues separately; do not relabel examples to improve scores.

## Development decision

Preserve v1 and introduce one explicitly versioned v2 revision with general
instructions for claim-by-claim checking, unknown versus false values, numerical
relationships/negation, and response-level ANY-unsupported probability. Do not
include these IDs, answers, labels, benchmark names or generator identities in
the prompt. Keep model, output schema, decoding and the selected inputs fixed.

This revision is motivated by inspected TRAIN outcomes and synthetic checks.
The same 50 examples remain development data. v2 bundles several changes; a
paired improvement would not identify which individual change caused it and
would not establish held-out or cross-domain improvement. Before scoring v2,
audit its new formatted lengths and freeze a separate execution budget. No v2
judge scores, thresholds, or final test results are claimed here.
