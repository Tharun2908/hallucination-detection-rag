# Post-thesis fresh S2 TEST inference and agreement

Operator-supplied cluster results received on 2026-09-28. These are post-thesis
baseline preparation observations, not new thesis results or TEST accuracy
metrics. The report author did not receive the full private prediction journal
or independently execute the weights.

## Completed run and replay

- Valid scores: 2,700 / 2,700; terminal failures and pending examples: zero.
- Completed sentence pairs: 362,918; forward batches: 20,576.
- Initial invocation: 2,700 attempted examples. Replay: zero new attempts.
- Truncated answer pairs: zero; truncated context pairs: 159.
  These are pair counts, not counts of unique examples.
- Known example time: 453.95719652296975 seconds; known CUDA-forward time:
  201.14531651698053 seconds. Unknown-timing examples: zero.
- Shared GPU, model loading excluded. These timings do not establish a
  controlled efficiency comparison with other verifiers.

Initial and replay report SHA256:
`8960c4a8c5d2c873b6f68241bb3c05810d6848fab392b9e8cf5407cdda4a013b`.
Private path: `s2-ragtruth-test-fresh-v1/summary.json` under the post-thesis
artifact root. No TEST label metrics, normalization fitting, fusion fitting,
judge calls or legacy-file changes were reported.

## Preserved legacy raw-min feature comparison

The [exact supplied agreement record](s2_legacy_score_agreement_20260928.json)
compares `raw_min_relevance`, the feature consumed by metadata-free fusion.
Its canonical report hash was independently recomputed from the supplied values.

| Observation | Value |
| --- | ---: |
| Examples | 2,700 |
| Matching answer/context sentence counts | 2,700 |
| Matching after rounding fresh features to four decimals | 2,669 |
| Mean absolute difference | 0.000025215495720508416 |
| Median absolute difference | 0.00002534465789805207 |
| Maximum absolute difference | 0.000054437255859296485 |
| Differences above 0.001 | 0 |
| Differences above 0.01 | 0 |

All sentence counts agree and score differences are small. The 31 four-decimal
mismatches are consistent with numerical differences near rounding boundaries;
the aggregates alone cannot establish the cause. No S2 tuning is warranted by
this result. Score agreement does not prove historical model/input provenance.

Next, apply the existing fusion coefficients, TRAIN normalization and operating
threshold to the fresh S2/S4 features. The registered application preserves the
historical four-decimal feature representation without overwriting full-precision
source scores. No refitting or label-based TEST selection is introduced.
