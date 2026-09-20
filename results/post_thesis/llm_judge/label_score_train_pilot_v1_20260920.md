# Post-thesis TRAIN label-score pilot — 2026-09-20

These experiments were conducted after thesis submission and are not part of the
submitted thesis results. This is the same adaptively inspected 50-example TRAIN
pilot used during earlier prompt development, not a held-out benchmark sample.

## Provenance and engineering outcome

The [CSV](label_score_train_pilot_v1_20260920.csv) preserves all 50 operator-supplied
score rows, unchanged in the cached replay. The [JSON](label_score_train_pilot_v1_20260920.json)
records both console report hashes, the frozen plan, row-aligned original labels,
metric definitions and comparisons. Labels/tasks were independently joined from
the checksum-verified pinned TRAIN parquet, reproducing the original manifest
hash exactly. No TEST file was read. The assistant has not independently inspected
the cluster's private journal or raw A/B log probabilities; the console report
hashes are operator-supplied provenance, not independently recomputed attestations.

- Code revision: `ccd8dd02b5a9c22717e2026ce4f9759f4012ceb5`.
- Fresh report: `fcc71604bfd7732cef74c883480eb4e821f45a8714bf12b7a8fe8a1d865b3248`.
- Cached report: `39b3d4e7119aca1725c2924e662c559d39c7796ee242020127902a63856fe820`.
- Valid scores: 50/50; terminal failures: 0; pending: 0; off-label tokens: 0.
- Attempts: 50 fresh, 0 cached. The derived report includes invocation-specific
  attempt counts, so the differing report hashes do not imply changed scores.
- Known tokens: 60,574 input and 50 output; no unknown usage.
- Charged client time: 50.16903266776353 seconds; remaining: 549.8309673322365.
  This includes preflight and client overhead, not just model inference. The
  console alone does not provide isolated latency percentiles or total rental
  cost. Currency cost remains unknown; unused budget is not permission to rerun.

The fixed prompt is `faithfulness-label-score-v1`, primary A=supported /
B=unsupported, with the pinned Qwen3-32B BF16 model and raw-logprob serving profile.
The run uses one generated token per example and the existing strict response
parser. Raw A/B probabilities, responses and per-attempt latencies stay in the
private journal; do not reconstruct them as if they had been directly observed.

## Descriptive development findings

The positive class is unsupported, with 24 positive and 26 negative original
labels. Classification below uses the emitted A/B token, without fitting a
threshold. AUROC uses raw log odds with exact ties counted as one half; average
precision uses stepwise PR area with exact-score tie groups, not trapezoidal area.

| Metric | Label-score pilot |
| --- | ---: |
| AUROC | 0.705128 |
| Average precision | 0.754326 |
| Precision | 0.600000 |
| Recall | 0.500000 |
| F1 | 0.545455 |
| Accuracy | 0.600000 |
| TP / FP / TN / FN | 12 / 8 / 18 / 12 |
| Brier score of uncalibrated relative score | 0.239817 |

The score range is approximately 0.000261–0.998299. A broad score range and valid
transport do not establish calibration. Brier score is descriptive squared error
against the labels, not a fitted calibration result. ECE binning was not fixed
for this pilot, so no pilot ECE is added retrospectively; register binning and
edge conventions before held-out evaluation. Tiny near-zero class-token log
masses, including small positive floating-point values within the existing
tolerance, are preserved without clipping.

Task slices are small and have different positive-label rates:

| Task | Examples | Positive labels | AUROC | F1 |
| --- | ---: | ---: | ---: | ---: |
| Data2txt | 24 | 15 | 0.674074 | 0.571429 |
| QA | 18 | 7 | 0.610390 | 0.500000 |
| Summary | 8 | 2 | 0.666667 | 0.500000 |

These slices are descriptive only. No task-specific threshold, prompt or score
mapping is chosen from them.

## Comparison with preserved development candidates

The [binary pilot](ragtruth_binary_pilot_20260919.md) had F1 0.650000, with
13 true positives, three false positives, 23 true negatives and 11 false negatives.
The new candidate matches 30/50 original labels, versus 36/50 for the binary pilot.
All 14 previous binary errors persist; six previously correct decisions change:

| ID | Original label | Binary verdict | Label-score verdict |
| --- | ---: | --- | --- |
| 11000 | 1 | unsupported | supported |
| 17143 | 0 | supported | unsupported |
| 10409 | 0 | supported | unsupported |
| 9180 | 0 | supported | unsupported |
| 14682 | 0 | supported | unsupported |
| 7469 | 0 | supported | unsupported |

No previous binary error is corrected. Persistent reviewed misses 6040, 6486,
12052 and 9350 remain supported predictions. Original labels are retained,
including previously disputed cases; this comparison is label agreement, not a
new exhaustive semantic adjudication.

The [earlier verbalized-score v2 pilot](ragtruth_pilot_v1_v2_20260919.md) had AUROC
0.756410 and average precision 0.700993. The current AUROC is lower while average
precision is higher. These metrics emphasize different aspects of ranking, so
this is mixed evidence, not uniform ranking improvement. The prompt/output format
and extraction differ between candidates; this is not an isolated causal test
of logprob extraction. No statistical significance or held-out superiority is claimed.

## Decision and next boundary

Preserve the candidate and negative findings. Keep the prompt and A/B mapping
unchanged; do not add more prompt variants, a label swap ensemble or a threshold
search on these 50 examples. The pilot demonstrates a working continuous-score
pipeline, not a benchmark-ready quality guarantee. A verifier can remain a useful
research baseline even when it does not beat the earlier development candidate.

The next step under the [continuous-score protocol](../../../post_thesis/llm_judge/CONTINUOUS_SCORE_PROTOCOL.md)
is to resolve native-source grouping and overlap, then register disjoint source
calibration/development data and the final evaluation procedure. The current
shared-context grouping is still only a proxy; do not claim native-source
independence. Keep these 50 and source-linked relatives outside threshold or
calibration fitting. Preserve the canonical HaluBench split. No new model calls,
TEST/HaluBench scoring, calibration fitting or final prompt freeze is authorized
by this results record.
