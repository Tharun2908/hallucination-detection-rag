# Post-thesis synthetic label-score findings — 2026-09-20

These are post-submission development observations, not submitted thesis results
or benchmark evaluation. Source: operator console output from the H200 run and
its cached replay; the assistant has not independently inspected private server
journals. The [CSV](label_score_synthetic_v1_20260920.csv) preserves all 30 displayed
score rows. The [JSON](label_score_synthetic_v1_20260920.json) records provenance
and accounting. Raw A/B log probabilities and server identities remain in private
records; they were not supplied in the console table and are not reconstructed here.

## Engineering outcome

- All 30 requests returned valid scores: 14 primary, 14 swapped, one repeat and
  one temperature control. No failures, pending work or off-label emitted tokens.
- The cached replay made zero new attempts; all displayed scores and accounting
  were unchanged. Its summary hash differs because `new_attempts` changes from
  30 to 0; preserve both hashes.
- Repeat and temperature controls returned identical raw A/B log probabilities
  on the single supported-short control input. This supports the checked transport
  behavior; it is not a general determinism or temperature-invariance study.
- Known usage: 18,786 input tokens, 30 output tokens, no unknown usage.
  Charged client time: 37.90494146896526 seconds. This includes client execution
  overhead and is not isolated model latency or total GPU rental time. Currency
  cost is unknown.

## Semantic outcome and limits

Both mappings match 12/14 expected greedy classes. Both incorrectly predict
support for `absence-unknown-short` and `absence-unknown-embedded`: an unknown
field does not establish that outdoor seating is absent.

| Case | Primary unsupported score | Swapped unsupported score |
| --- | ---: | ---: |
| Unknown absence, short | 0.132964 | 0.320821 |
| Unknown absence, embedded | 0.164516 | 0.148047 |

These examples still rank above supported examples within this deliberately tiny
suite. No threshold is selected from this observation. Swapping the label mapping
changes some margins substantially even when emitted classes agree; it is only a
diagnostic, not a candidate selected or averaged after seeing outcomes.

Eleven of 14 scores in each mapping lie outside [0.01, 0.99]. Relative token
likelihoods are uncalibrated; confidence near an endpoint does not establish
correctness or calibration. Class-token log mass is approximately zero. Small
positive values (maximum about 5.03e-8) fit the existing numerical tolerance;
retain the original values rather than silently clipping them.

## Decision

Keep `faithfulness-label-score-v1`, primary A=supported / B=unsupported, the
pinned Qwen3-32B revision and raw-logprob extraction fixed. Record the two absence
failures rather than retuning against this suite. Proceed only to the
[CPU token audit](../../../post_thesis/llm_judge/LABEL_SCORE_PILOT_AUDIT.md) of the
unchanged original 50 TRAIN examples. A separate bounded execution plan follows
that audit. No TEST/HaluBench calls, prompt freeze for evaluation, threshold
fitting, calibration or comparison with benchmark baselines is claimed.
