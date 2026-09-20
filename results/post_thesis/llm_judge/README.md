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
passed its [native compatibility check](evidence_schema_v2_native_20260919.md) at 38/38. Its [live synthetic result](evidence_diagnostic_v2_20260919.md) has 14/14 valid records with two false positives and a persistent explanation concern. The [verdict-first native check](evidence_schema_v3_native_20260919.md) passed 44/44. The [v3 live findings](evidence_diagnostic_v3_20260919.md) and [parsed observations](evidence_diagnostic_v3_20260919.json) record 14/14 valid, correctly ordered outputs and 13/14 expected verdicts. The original TRAIN evidence audit and scoring runs are complete; see the findings below. Raw answer/context text, request caches and execution
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

The [evidence prompt-v2 TRAIN findings](ragtruth_evidence_prompt_v2_pilot_20260920.md)
and [exported evidence with paired outcomes](ragtruth_evidence_prompt_v2_pilot_20260920.json)
record 43/50 valid outputs, seven retained quote failures and targeted semantic
review. All six recovered records match positive labels, but shared-valid label
agreement changes from 27/37 to 26/37. Preserve these mixed development findings;
evidence-prompt tuning is paused while the continuous-score protocol is designed.

The [label-score setup observations](label_score_setup_20260920.md) record the
passed cluster tokenizer and installed-source checks. The separate bounded
synthetic score experiment is complete (see findings below); its outputs are
not thesis or benchmark results.

The [synthetic label-score findings](label_score_synthetic_v1_20260920.md), [operator record](label_score_synthetic_v1_20260920.json) and [30 score rows](label_score_synthetic_v1_20260920.csv) record the completed transport check and cached replay. Both mappings miss the two unknown-as-absence cases. The [cluster original-50 TRAIN token audit](label_score_pilot_audit_20260920.json) subsequently matched all local counts; the [completed TRAIN label-score pilot and replay](label_score_train_pilot_v1_20260920.md) have 50 valid scores and zero replay calls. The [operator/analysis record](label_score_train_pilot_v1_20260920.json) and [50 score rows](label_score_train_pilot_v1_20260920.csv) preserve the mixed development findings. Native-source grouping and disjoint calibration/evaluation remain pending.

The [assistant CPU native-source audit](source_audit_local_20260920.json) checks pinned TRAIN identity and declared exact-overlap rules. No additional pilot-linked exclusions were found. It is a pre-release local check; cluster reproduction and broader disjointness checks remain pending.

The [completed cluster TRAIN source audit](source_audit_cluster_20260920.json) matches the local check, including the independently reconstructed full report hash. The [next native cross-split local check](cross_split_audit_local_20260920.json) proposes 12 additional TRAIN candidate exclusions from exact shared evidence, without using TEST answers or labels. Its [completed cluster result](cross_split_audit_cluster_20260920.md) matches, including independent reconstruction of the full report hash. Broader disjointness remains unestablished. The [development reservation design](../../../post_thesis/llm_judge/DEVELOPMENT_SPLIT_PROTOCOL.md) records deterministic 100-component calibration and threshold roles; no allocation or fitting has run.

The [local development reservation check](development_reservation_local_20260920.md) and [aggregate/hash record](development_reservation_local_20260920.json) implement that committed design: 600 rows per selected arm, 312 exclusions, 13,578 unallocated rows, zero model calls. Cluster reproduction is pending; no calibration or threshold fitting has occurred.

The [completed cluster reservation and identical replay](development_reservation_cluster_20260920.md) now reproduce all counts and the independently reconstructed full manifest hash. The [cluster record](development_reservation_cluster_20260920.json) pins the data roles for the [calibration/threshold design](../../../post_thesis/llm_judge/CALIBRATION_THRESHOLD_PROTOCOL.md). No new scores, fitted calibrator or threshold are available.

The [local reserved-arm token audit](development_token_audit_local_20260920.md) and [hash/count record](development_token_audit_local_20260920.json) cover all 1,200 inputs with the actual pinned tokenizer. All fit without truncation; cached replay adds no development tokenizations. Cluster reproduction is pending and there is no scoring allowance.

The [completed cluster development token audit and replay](development_token_audit_cluster_20260920.md) and [hash/count record](development_token_audit_cluster_20260920.json) reproduce all 1,200 input lengths. The [bounded scoring plan](../../../post_thesis/llm_judge/DEVELOPMENT_SCORING_RUN.md) is implemented; no live reserved-arm scores or fitted parameters have been collected by this patch.

- [Reserved TRAIN scoring completed](development_scoring_cluster_20260920.md): operator-reported full coverage and cached reuse in both 600-row arms; actual CPU fitting remains pending. See the [exact console record](development_scoring_cluster_20260920.json).
