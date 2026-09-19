# Post-thesis LLM judge results

Results collected after thesis submission. These development artifacts are
excluded from the submitted thesis results. **No final test results are available.**

- [Synthetic H200 smoke findings](synthetic_smoke_20260919.md).
- [First 50-example TRAIN pilot report](ragtruth_pilot_v1_20260919.md) and
  [operator-supplied predictions](ragtruth_pilot_v1_20260919.csv).
- [Paired v1/v2 TRAIN development report](ragtruth_pilot_v1_v2_20260919.md) and
  [paired predictions](ragtruth_pilot_v1_v2_20260919.csv).
- [Probability diagnostic findings](synthetic_diagnostic_v1_20260919.md) and
  [operator-supplied results](synthetic_diagnostic_v1_20260919.json).
- [Binary diagnostic findings](binary_diagnostic_v1_20260919.md) and
  [operator-supplied verdicts](binary_diagnostic_v1_20260919.json).

- [Binary TRAIN pilot findings](ragtruth_binary_pilot_20260919.md),
  [operator-supplied verdicts](ragtruth_binary_pilot_20260919.csv) and
  [metrics/provenance](ragtruth_binary_pilot_20260919.json).
- [Focused binary error review](ragtruth_binary_error_review_20260919.md).

Probability pilots, synthetic diagnostics, the binary TRAIN pilot and the focused
error review are complete. All remain adaptively inspected development results. Raw answer/context text, request caches and execution
journals remain in ignored private artifact directories.

Future run directories must use unique run IDs. Include protocol/configuration,
model/prompt identities, input manifest references, successful-scoring coverage,
metrics, and cost/latency summaries. Preserve frozen runs rather than overwriting
them. Raw request caches belong in the ignored artifact directory described in
[the experiment README](../../../post_thesis/llm_judge/README.md).

Existing canonical results elsewhere in `results/` remain comparison inputs;
do not overwrite them with this extension's outputs.
