# Post-thesis LLM judge results

**Status — 2026-11-02: complete.** Both frozen benchmark evaluations, cached
replays, the selected qualitative review and inference accounting are recorded.
Release tag: `post-thesis-llm-judge-v1` at `c1b1cd6`.
These findings are separate from the submitted thesis results.

## Final findings

| Report | What it contains |
| --- | --- |
| [Final overview](FINAL_OVERVIEW.md) | Cross-benchmark results, paired intervals, figures and interpretation |
| [RAGTruth evaluation](ragtruth_frozen_evaluation_20260928.md) · [JSON](ragtruth_frozen_evaluation_20260928.json) | Four fresh systems on 2,700 responses / 450 groups |
| [HaluBench evaluation](halubench_frozen_evaluation_20260929.md) · [JSON](halubench_frozen_evaluation_20260929.json) | Same frozen rules on 8,000 responses / 7,198 bootstrap components |
| [Qualitative review](qualitative_review_20261102.md) · [JSON](qualitative_review_20261102.json) | 127 selected cases; AI-assisted judgments collectively confirmed before unblinding |
| [Verified inference accounting](efficiency_accounting_20261102.md) · [JSON](efficiency_accounting_20261102.json) | Timing scopes, token usage, interrupted-attempt accounting and cost limits |
| [Figures and regeneration](../../../figures/post_thesis/llm_judge/README.md) | CPU-only plotting from committed aggregates |

The judge trails every fresh baseline on RAGTruth AUROC, AP and F1. It leads
aggregate HaluBench AUROC and F1, while MiniCheck retains higher precision and
accuracy; the judge-minus-MiniCheck AP interval includes zero. Source-level
weaknesses and differing evidence visibility remain central to interpretation.

The qualitative review is selected-case analysis, not independent annotation or
label correction. Historical baseline provenance remains limited, MiniCheck
fresh/legacy drift remains unexplained, and monetary cost is unknown.

## Reading the dated records

The records below preserve observations **at each development stage**. A dated
report may say that the following experiment was pending; that is historical
context, not the current release status. Original JSON status fields, failures,
request identities and budget records are not rewritten after later completion.
The [documentation index](../../../post_thesis/llm_judge/DOCUMENTATION_INDEX.md)
maps all procedures to their completed outcomes.

Probability, binary and evidence pilots were adaptively inspected development
runs. They did not supply the final benchmark scores or held-out model-selection
evidence. The final study used the frozen class-token label-score judge.

## Study design and final evaluation

| Record | Outcome at this stage |
| --- | --- |
| [Evaluation mathematics](evaluation_math_cluster_20260920.md) | Synthetic checks passed; benchmark loaders subsequently used the frozen core |

## Historical diagnostics

| Record | Outcome at this stage |
| --- | --- |
| [Initial serving and smoke checks](synthetic_smoke_20260919.md) | Completed; the final judge uses the separate label-score-v1 profile |
| [Initial probability TRAIN pilot](ragtruth_pilot_v1_20260919.md) · [CSV](ragtruth_pilot_v1_20260919.csv) | Completed adaptive 50-example development run |
| [Probability prompt v2](ragtruth_pilot_v1_v2_20260919.md) · [CSV](ragtruth_pilot_v1_v2_20260919.csv) | Completed paired TRAIN development comparison |
| [Probability synthetic diagnostic](synthetic_diagnostic_v1_20260919.md) · [JSON](synthetic_diagnostic_v1_20260919.json) | Completed; embedded errors received diluted scores |
| [Binary synthetic diagnostic](binary_diagnostic_v1_20260919.md) · [JSON](binary_diagnostic_v1_20260919.json) | Completed: 10/10 expected verdicts |
| [Binary TRAIN pilot](ragtruth_binary_pilot_20260919.md) · [JSON](ragtruth_binary_pilot_20260919.json) · [CSV](ragtruth_binary_pilot_20260919.csv) | Completed; focused error review retained semantic misses |
| [Evidence contract](evidence_diagnostic_v1_20260919.md) · [JSON](evidence_diagnostic_v1_20260919.json) | Implemented and exercised; not selected as the final benchmark scorer |
| [Evidence schema-v2 checks](evidence_schema_v2_native_20260919.md) · [JSON](evidence_schema_v2_native_20260919.json) | Completed: 38/38 native grammar checks |
| [Evidence schema-v2 live run](evidence_diagnostic_v2_20260919.md) · [JSON](evidence_diagnostic_v2_20260919.json) | Completed: 14/14 valid records, with two false positives |
| [Verdict-first schema checks](evidence_schema_v3_native_20260919.md) · [JSON](evidence_schema_v3_native_20260919.json) | Completed: 44/44 native order checks |
| [Verdict-first live run](evidence_diagnostic_v3_20260919.md) · [JSON](evidence_diagnostic_v3_20260919.json) | Completed: 14/14 valid records and 13/14 expected verdicts |
| [Evidence TRAIN token audit](ragtruth_evidence_v3_audit_20260919.md) · [JSON](ragtruth_evidence_v3_audit_20260919.json) | Completed: 50 inputs and 77,574 input tokens |
| [Evidence TRAIN run](ragtruth_evidence_pilot_v3_20260919.md) · [JSON](ragtruth_evidence_pilot_v3_20260919.json) | Completed: 37/50 valid records; targeted semantic review recorded |
| [Evidence prompt-v2 candidate](ragtruth_evidence_prompt_v2_pilot_20260920.md) · [JSON](ragtruth_evidence_prompt_v2_pilot_20260920.json) | Completed paired synthetic and original-TRAIN checks; tuning paused |
| [Paired evidence prompts](evidence_prompt_pair_v1_20260919.md) · [JSON](evidence_prompt_pair_v1_20260919.json) | Completed: 26 cases per arm, with mixed semantic findings |
| [Evidence prompt-v2 token audit](ragtruth_evidence_prompt_v2_audit_20260920.md) · [JSON](ragtruth_evidence_prompt_v2_audit_20260920.json) | Completed: 50 inputs and 77,724 input tokens |

## Label-score development and freeze

| Record | Outcome at this stage |
| --- | --- |
| [Label-score contract and tokenizer](label_score_setup_20260920.md) · [JSON](label_score_setup_20260920.json) | Completed offline, tokenizer and installed-source checks |
| [Synthetic label scores](label_score_synthetic_v1_20260920.md) · [JSON](label_score_synthetic_v1_20260920.json) · [CSV](label_score_synthetic_v1_20260920.csv) | Completed: 30/30 valid scores and cached replay |
| [Label-score TRAIN token audit](label_score_pilot_audit_20260920.json) | Completed: 50 inputs and 60,574 input tokens |
| [Label-score TRAIN pilot](label_score_train_pilot_v1_20260920.md) · [JSON](label_score_train_pilot_v1_20260920.json) · [CSV](label_score_train_pilot_v1_20260920.csv) | Completed: 50/50 valid scores and cached replay |
| [Native TRAIN provenance](source_audit_cluster_20260920.json) | Completed scoped exact-overlap audit |
| [Native cross-split overlap](cross_split_audit_cluster_20260920.md) · [JSON](cross_split_audit_cluster_20260920.json) | Completed; 12 additional TEST-linked TRAIN candidate rows excluded |
| [Development reservation design](development_reservation_cluster_20260920.md) · [JSON](development_reservation_cluster_20260920.json) | Implemented without changing the original design |
| [Calibration and threshold design](development_fit_cluster_20260920.md) · [JSON](development_fit_cluster_20260920.json) | Implemented; fitted parameters and threshold frozen |
| [Reserved-arm token audit](development_token_audit_cluster_20260920.md) · [JSON](development_token_audit_cluster_20260920.json) | Completed: all 1,200 inputs fit |
| [Reserved-arm scoring](development_scoring_cluster_20260920.md) · [JSON](development_scoring_cluster_20260920.json) | Completed: 600/600 valid scores per arm and cached replay |

## RAGTruth preparation and inference

| Record | Outcome at this stage |
| --- | --- |
| [TEST input alignment](fresh_comparison_manifest_20260928.md) | Completed; final comparison uses fresh baseline input provenance |
| [Legacy baseline audit](baseline_provenance_cluster_20260920.md) · [JSON](baseline_provenance_cluster_20260920.json) | Audit completed; historical inference/training provenance remains limited |
| [Metadata-free fusion recovery](fusion_recovery_cluster_20260920.md) · [JSON](fusion_recovery_cluster_20260920.json) | Completed CPU reconstruction and zero-fit replay |
| [TEST token audit](test_token_audit_cluster_20260920.md) · [JSON](test_token_audit_cluster_20260920.json) | Completed: all 2,700 inputs fit |
| [S4 checkpoint compatibility](s4_checkpoint_cluster_20260920.md) | Passed CPU synthetic compatibility checks |
| [Fresh S4 inference](s4_fresh_test_cluster_20260920.md) | Completed: 2,700/2,700 valid scores |
| [Fresh S2 inference](s2_fresh_test_cluster_20260928.md) | Completed: 2,700/2,700 valid features and cached replay |
| [Fusion on fresh features](fresh_fusion_cluster_20260928.md) | Completed: 2,700/2,700 scores and identical replay |
| [MiniCheck preparation](minicheck_synthetic_v2_20260928.md) | Completed; later synthetic v2 check validated the runtime |
| [Fresh MiniCheck inference](minicheck_fresh_test_cluster_20260928.md) | Completed: 2,700/2,700 scores and identical replay |
| [Frozen Qwen inference](qwen_ragtruth_test_cluster_20260928.md) | Completed: 2,700/2,700 valid scores and identical replay |

## HaluBench preparation and inference

| Record | Outcome at this stage |
| --- | --- |
| [Canonical TEST inputs](halubench_inputs_cluster_20260928.json) | Completed: unchanged 8,000-row membership; passage overlap disclosed |
| [Canonical TEST token audit](halubench_token_audit_cluster_20260928.json) | Completed: all 8,000 inputs fit and replay added no tokenizations |
| [Fresh S4 inference](s4_halubench_cluster_20260928.json) | Completed: 8,000/8,000 scores and identical replay |
| [Fresh S2 inference](s2_halubench_cluster_20260928.json) | Completed: 8,000 features; 1,886 original raw-zero fallbacks retained |
| [Fusion on fresh features](halubench_fusion_cluster_20260928.json) | Completed: 8,000/8,000 scores without refitting |
| [Fresh MiniCheck inference](minicheck_halubench_cluster_20260928.json) | Completed: 8,000/8,000 scores and identical replay |
| [Original frozen Qwen run](halubench_budget_stop_20260929.json) | Parent stopped at 7,652 valid scores; separate continuation completed the population |
| [Explicit Qwen continuation](halubench_continuation_complete_20260929.json) | Completed: 348/348 child scores, 8,000/8,000 combined, identical replay |

## Additional audit and provenance records

These supplement the primary reports above; local checks and earlier inventories
do not supersede later cluster results or fresh inference provenance.

| Record | Artifact |
| --- | --- |
| [cross split audit local 20260920](cross_split_audit_local_20260920.json) | Machine-readable record |
| [development reservation local 20260920](development_reservation_local_20260920.json) | Machine-readable record |
| [Post-thesis TRAIN development reservation: local validation](development_reservation_local_20260920.md) | Narrative |
| [development token audit local 20260920](development_token_audit_local_20260920.json) | Machine-readable record |
| [Post-thesis reserved TRAIN token audit: local validation](development_token_audit_local_20260920.md) | Narrative |
| [evaluation math local 20260920](evaluation_math_local_20260920.json) | Machine-readable record |
| [frozen fit verification 20260920](frozen_fit_verification_20260920.json) | Machine-readable record |
| [halubench inputs local 20260928](halubench_inputs_local_20260928.json) | Machine-readable record |
| [halubench judge requests local 20260928](halubench_judge_requests_local_20260928.json) | Machine-readable record |
| [halubench token audit local 20260928](halubench_token_audit_local_20260928.json) | Machine-readable record |
| [legacy baseline inventory 20260920](legacy_baseline_inventory_20260920.json) | Machine-readable record |
| [minicheck environment 20260928](minicheck_environment_20260928.json) | Machine-readable record |
| [minicheck halubench workload local 20260928](minicheck_halubench_workload_local_20260928.json) | Machine-readable record |
| [Post-thesis MiniCheck fresh/legacy score comparison](minicheck_legacy_score_agreement_20260928.md) | Narrative |
| [Post-thesis binary pilot: focused source review — 2026-09-19](ragtruth_binary_error_review_20260919.md) | Narrative |
| [s2 halubench workload local 20260928](s2_halubench_workload_local_20260928.json) | Machine-readable record |
| [s2 legacy score agreement 20260928](s2_legacy_score_agreement_20260928.json) | Machine-readable record |
| [s4 legacy score agreement 20260920](s4_legacy_score_agreement_20260920.json) | Machine-readable record |
| [source audit local 20260920](source_audit_local_20260920.json) | Machine-readable record |
| [test alignment local 20260920](test_alignment_local_20260920.json) | Machine-readable record |
| [test token audit local 20260920](test_token_audit_local_20260920.json) | Machine-readable record |

## Preservation and reproduction

Raw answer/context text, prompts, request journals, checkpoint fingerprints,
private evaluations and review packets/keys remain outside Git under the private
artifact root. Public aggregate reports are not a substitute for those records.
The HaluBench parent and continuation are both required: the parent retains its
budget stop and one interrupted attempt with unknown usage; the child completes
the 8,000-example population without overwriting the parent.

Use the [experiment README](../../../post_thesis/llm_judge/README.md) for CPU replay
and artifact locations. Preserve all frozen runs and submitted-thesis artifacts.
Any new inference, cascade, prompt revision or fit needs a distinct experiment
identity; none is an unfinished task for this release.
