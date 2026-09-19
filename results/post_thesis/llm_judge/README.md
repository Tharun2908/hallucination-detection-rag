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
error review are complete. All remain adaptively inspected development results.
The [structured evidence v1 findings](evidence_diagnostic_v1_20260919.md) and
[operator-supplied responses](evidence_diagnostic_v1_20260919.json) record 11/14
valid synthetic outputs and one explanation concern. The next
[schema-only v2 candidate](../../../post_thesis/llm_judge/EVIDENCE_SCHEMA_V2.md)
passed its [native compatibility check](evidence_schema_v2_native_20260919.md) at 38/38. Its [live synthetic result](evidence_diagnostic_v2_20260919.md) has 14/14 valid records with two false positives and a persistent explanation concern. The [verdict-first native check](evidence_schema_v3_native_20260919.md) passed 44/44. The [v3 live findings](evidence_diagnostic_v3_20260919.md) and [parsed observations](evidence_diagnostic_v3_20260919.json) record 14/14 valid, correctly ordered outputs and 13/14 expected verdicts. A [token-only original TRAIN pilot audit](../../../post_thesis/llm_judge/EVIDENCE_PILOT.md) is prepared; evidence TRAIN scoring remains pending. Raw answer/context text, request caches and execution
journals remain in ignored private artifact directories.

Future run directories must use unique run IDs. Include protocol/configuration,
model/prompt identities, input manifest references, successful-scoring coverage,
metrics, and cost/latency summaries. Preserve frozen runs rather than overwriting
them. Raw request caches belong in the ignored artifact directory described in
[the experiment README](../../../post_thesis/llm_judge/README.md).

Existing canonical results elsewhere in `results/` remain comparison inputs;
do not overwrite them with this extension's outputs.

The [evidence-v3 original TRAIN token audit](ragtruth_evidence_v3_audit_20260919.md)
is complete: 50/50 counted, no overlength inputs and zero generation calls.
The [bounded scoring plan](../../../post_thesis/llm_judge/EVIDENCE_PILOT_RUN.md)
was executed: the [evidence TRAIN findings](ragtruth_evidence_pilot_v3_20260919.md) record 37/50 valid outputs, 13 quote failures and targeted semantic review. These are adaptive development results; full claim-level review remains incomplete.

The [paired evidence prompt findings](evidence_prompt_pair_v1_20260919.md) record
26/26 valid outputs per prompt, 24 versus 25 expected verdict matches, one
corrected absence verdict, one v2 rationale regression and one shared missed
false-absence claim. All paired exported evidence and assistant review tags are
preserved in the accompanying JSON. This remains adaptive synthetic development.
