# Post-thesis documentation index

**Status — 2026-11-02:** the frozen study is complete. Start with the
[experiment README](README.md) or [final overview](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md).
This index links every run/design guide to its recorded outcome.

## How to read the guides

- Run commands, budgets, revision checks and failure policies describe the
  original procedures. They are retained for reproduction, not a request to
  start new model calls or repeat fitting.
- A command that performs only tokenization or preparation still grants no
  scoring allowance. Its scope has not changed because later stages completed.
- Original JSON status fields and dated result records are historical snapshots.
  A `pending`, `comparison_ready: false` or `generation_enabled: false` field in
  an earlier artifact is not a current release task. Later, separately pinned
  reports establish completion; the original artifacts remain unchanged.
- Completed diagnostics can retain failures and unresolved semantic concerns.
  The release uses the frozen label-score judge, not those earlier candidates.
- Full replay needs private artifacts and the guide's recorded environment and
  revision requirements. Public summaries alone do not reproduce raw inference.

## Study design and final evaluation

| Guide | Recorded outcome | Evidence |
| --- | --- | --- |
| [Study design and development history](PROTOCOL.md) | Completed study; later score and evaluation contracts govern final results | [Record](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md) |
| [Continuous label-score design](CONTINUOUS_SCORE_PROTOCOL.md) | Implemented, calibrated and evaluated on both benchmarks | [Record](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md) |
| [Frozen evaluation rules](EVALUATION_PROTOCOL.md) | Both frozen benchmark evaluations and replays completed | [Record](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md) |
| [Evaluation mathematics](EVALUATION_MATH.md) | Synthetic checks passed; benchmark loaders subsequently used the frozen core | [Record](../../results/post_thesis/llm_judge/evaluation_math_cluster_20260920.md) |
| [RAGTruth evaluation](TEST_EVALUATION.md) | Completed: 2,700 shared responses and 450 groups | [Record](../../results/post_thesis/llm_judge/ragtruth_frozen_evaluation_20260928.md) |
| [HaluBench evaluation](HALUBENCH_EVALUATION.md) | Completed: 8,000 shared responses and 7,198 bootstrap components | [Record](../../results/post_thesis/llm_judge/halubench_frozen_evaluation_20260929.md) |
| [Qualitative review](DISAGREEMENT_REVIEW.md) | Completed selected-case review; 41 cases remain uncertain | [Record](../../results/post_thesis/llm_judge/qualitative_review_20261102.md) |

## Historical diagnostics

| Guide | Recorded outcome | Evidence |
| --- | --- | --- |
| [Initial serving and smoke checks](SERVING.md) | Completed; the final judge uses the separate label-score-v1 profile | [Record](../../results/post_thesis/llm_judge/synthetic_smoke_20260919.md) |
| [Initial probability TRAIN pilot](PILOT.md) | Completed adaptive 50-example development run | [Record](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_20260919.md) |
| [Probability prompt v2](PROMPT_V2.md) | Completed paired TRAIN development comparison | [Record](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_v2_20260919.md) |
| [Probability synthetic diagnostic](DIAGNOSTIC.md) | Completed; embedded errors received diluted scores | [Record](../../results/post_thesis/llm_judge/synthetic_diagnostic_v1_20260919.md) |
| [Binary synthetic diagnostic](BINARY_DIAGNOSTIC.md) | Completed: 10/10 expected verdicts | [Record](../../results/post_thesis/llm_judge/binary_diagnostic_v1_20260919.md) |
| [Binary TRAIN pilot](BINARY_PILOT.md) | Completed; focused error review retained semantic misses | [Record](../../results/post_thesis/llm_judge/ragtruth_binary_pilot_20260919.md) |
| [Evidence contract](EVIDENCE_DIAGNOSTIC.md) | Implemented and exercised; not selected as the final benchmark scorer | [Record](../../results/post_thesis/llm_judge/evidence_diagnostic_v1_20260919.md) |
| [Evidence schema-v1 run](EVIDENCE_RUN.md) | Completed: 11/14 valid synthetic records | [Record](../../results/post_thesis/llm_judge/evidence_diagnostic_v1_20260919.md) |
| [Evidence schema-v2 checks](EVIDENCE_SCHEMA_V2.md) | Completed: 38/38 native grammar checks | [Record](../../results/post_thesis/llm_judge/evidence_schema_v2_native_20260919.md) |
| [Evidence schema-v2 live run](EVIDENCE_V2_RUN.md) | Completed: 14/14 valid records, with two false positives | [Record](../../results/post_thesis/llm_judge/evidence_diagnostic_v2_20260919.md) |
| [Verdict-first schema checks](EVIDENCE_ORDER_V3.md) | Completed: 44/44 native order checks | [Record](../../results/post_thesis/llm_judge/evidence_schema_v3_native_20260919.md) |
| [Verdict-first live run](EVIDENCE_V3_RUN.md) | Completed: 14/14 valid records and 13/14 expected verdicts | [Record](../../results/post_thesis/llm_judge/evidence_diagnostic_v3_20260919.md) |
| [Evidence TRAIN token audit](EVIDENCE_PILOT.md) | Completed: 50 inputs and 77,574 input tokens | [Record](../../results/post_thesis/llm_judge/ragtruth_evidence_v3_audit_20260919.md) |
| [Evidence TRAIN run](EVIDENCE_PILOT_RUN.md) | Completed: 37/50 valid records; targeted semantic review recorded | [Record](../../results/post_thesis/llm_judge/ragtruth_evidence_pilot_v3_20260919.md) |
| [Evidence prompt-v2 candidate](EVIDENCE_PROMPT_V2.md) | Completed paired synthetic and original-TRAIN checks; tuning paused | [Record](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_pilot_20260920.md) |
| [Paired evidence prompts](EVIDENCE_PAIR_RUN.md) | Completed: 26 cases per arm, with mixed semantic findings | [Record](../../results/post_thesis/llm_judge/evidence_prompt_pair_v1_20260919.md) |
| [Evidence prompt-v2 token audit](EVIDENCE_PROMPT_V2_PILOT.md) | Completed: 50 inputs and 77,724 input tokens | [Record](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_audit_20260920.md) |
| [Evidence prompt-v2 TRAIN run](EVIDENCE_PROMPT_V2_PILOT_RUN.md) | Completed: 43/50 valid records; targeted semantic review recorded | [Record](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_pilot_20260920.md) |

## Label-score development and freeze

| Guide | Recorded outcome | Evidence |
| --- | --- | --- |
| [Label-score contract and tokenizer](LABEL_SCORE_OFFLINE.md) | Completed offline, tokenizer and installed-source checks | [Record](../../results/post_thesis/llm_judge/label_score_setup_20260920.md) |
| [Synthetic label scores](LABEL_SCORE_RUN.md) | Completed: 30/30 valid scores and cached replay | [Record](../../results/post_thesis/llm_judge/label_score_synthetic_v1_20260920.md) |
| [Label-score TRAIN token audit](LABEL_SCORE_PILOT_AUDIT.md) | Completed: 50 inputs and 60,574 input tokens | [Record](../../results/post_thesis/llm_judge/label_score_pilot_audit_20260920.json) |
| [Label-score TRAIN pilot](LABEL_SCORE_PILOT_RUN.md) | Completed: 50/50 valid scores and cached replay | [Record](../../results/post_thesis/llm_judge/label_score_train_pilot_v1_20260920.md) |
| [Native TRAIN provenance](SOURCE_AUDIT.md) | Completed scoped exact-overlap audit | [Record](../../results/post_thesis/llm_judge/source_audit_cluster_20260920.json) |
| [Native cross-split overlap](CROSS_SPLIT_AUDIT.md) | Completed; 12 additional TEST-linked TRAIN candidate rows excluded | [Record](../../results/post_thesis/llm_judge/cross_split_audit_cluster_20260920.md) |
| [Development reservation design](DEVELOPMENT_SPLIT_PROTOCOL.md) | Implemented without changing the original design | [Record](../../results/post_thesis/llm_judge/development_reservation_cluster_20260920.md) |
| [Development allocation](DEVELOPMENT_RESERVATION.md) | Completed: 600 rows per arm, 312 exclusions | [Record](../../results/post_thesis/llm_judge/development_reservation_cluster_20260920.md) |
| [Calibration and threshold design](CALIBRATION_THRESHOLD_PROTOCOL.md) | Implemented; fitted parameters and threshold frozen | [Record](../../results/post_thesis/llm_judge/development_fit_cluster_20260920.md) |
| [Reserved-arm token audit](DEVELOPMENT_TOKEN_AUDIT.md) | Completed: all 1,200 inputs fit | [Record](../../results/post_thesis/llm_judge/development_token_audit_cluster_20260920.md) |
| [Reserved-arm scoring](DEVELOPMENT_SCORING_RUN.md) | Completed: 600/600 valid scores per arm and cached replay | [Record](../../results/post_thesis/llm_judge/development_scoring_cluster_20260920.md) |
| [Offline fitting and fit verification](OFFLINE_FITTING.md) | Completed: accepted calibration, margin threshold -1.25, identical replay | [Record](../../results/post_thesis/llm_judge/development_fit_cluster_20260920.md) |

## RAGTruth preparation and inference

| Guide | Recorded outcome | Evidence |
| --- | --- | --- |
| [TEST input alignment](TEST_ALIGNMENT.md) | Completed; final comparison uses fresh baseline input provenance | [Record](../../results/post_thesis/llm_judge/fresh_comparison_manifest_20260928.md) |
| [Legacy baseline audit](BASELINE_PROVENANCE.md) | Audit completed; historical inference/training provenance remains limited | [Record](../../results/post_thesis/llm_judge/baseline_provenance_cluster_20260920.md) |
| [Metadata-free fusion recovery](FUSION_RECOVERY.md) | Completed CPU reconstruction and zero-fit replay | [Record](../../results/post_thesis/llm_judge/fusion_recovery_cluster_20260920.md) |
| [TEST token audit](TEST_TOKEN_AUDIT.md) | Completed: all 2,700 inputs fit | [Record](../../results/post_thesis/llm_judge/test_token_audit_cluster_20260920.md) |
| [S4 checkpoint compatibility](S4_CHECKPOINT_CHECK.md) | Passed CPU synthetic compatibility checks | [Record](../../results/post_thesis/llm_judge/s4_checkpoint_cluster_20260920.md) |
| [Fresh S4 inference](S4_TEST_RUN.md) | Completed: 2,700/2,700 valid scores | [Record](../../results/post_thesis/llm_judge/s4_fresh_test_cluster_20260920.md) |
| [Fresh S2 inference](S2_TEST_RUN.md) | Completed: 2,700/2,700 valid features and cached replay | [Record](../../results/post_thesis/llm_judge/s2_fresh_test_cluster_20260928.md) |
| [Fusion on fresh features](FRESH_FUSION.md) | Completed: 2,700/2,700 scores and identical replay | [Record](../../results/post_thesis/llm_judge/fresh_fusion_cluster_20260928.md) |
| [MiniCheck preparation](MINICHECK_PREPARATION.md) | Completed; later synthetic v2 check validated the runtime | [Record](../../results/post_thesis/llm_judge/minicheck_synthetic_v2_20260928.md) |
| [MiniCheck synthetic compatibility](MINICHECK_LIVE_CHECK.md) | V1 failed; v2 passed and cached replay made no new attempt | [Record](../../results/post_thesis/llm_judge/minicheck_synthetic_v2_20260928.md) |
| [Fresh MiniCheck inference](MINICHECK_TEST_RUN.md) | Completed: 2,700/2,700 scores and identical replay | [Record](../../results/post_thesis/llm_judge/minicheck_fresh_test_cluster_20260928.md) |
| [Fresh baseline comparison manifest](COMPARISON_MANIFEST.md) | Verified all three baselines; later judge inference/evaluation completed | [Record](../../results/post_thesis/llm_judge/fresh_comparison_manifest_20260928.md) |
| [Frozen Qwen inference](TEST_LABEL_RUN.md) | Completed: 2,700/2,700 valid scores and identical replay | [Record](../../results/post_thesis/llm_judge/qwen_ragtruth_test_cluster_20260928.md) |

## HaluBench preparation and inference

| Guide | Recorded outcome | Evidence |
| --- | --- | --- |
| [Canonical TEST inputs](HALUBENCH_INPUTS.md) | Completed: unchanged 8,000-row membership; passage overlap disclosed | [Record](../../results/post_thesis/llm_judge/halubench_inputs_cluster_20260928.json) |
| [Canonical TEST token audit](HALUBENCH_TOKEN_AUDIT.md) | Completed: all 8,000 inputs fit and replay added no tokenizations | [Record](../../results/post_thesis/llm_judge/halubench_token_audit_cluster_20260928.json) |
| [Fresh S4 inference](HALUBENCH_S4_RUN.md) | Completed: 8,000/8,000 scores and identical replay | [Record](../../results/post_thesis/llm_judge/s4_halubench_cluster_20260928.json) |
| [Fresh S2 inference](HALUBENCH_S2_RUN.md) | Completed: 8,000 features; 1,886 original raw-zero fallbacks retained | [Record](../../results/post_thesis/llm_judge/s2_halubench_cluster_20260928.json) |
| [Fusion on fresh features](HALUBENCH_FUSION_RUN.md) | Completed: 8,000/8,000 scores without refitting | [Record](../../results/post_thesis/llm_judge/halubench_fusion_cluster_20260928.json) |
| [Fresh MiniCheck inference](HALUBENCH_MINICHECK_RUN.md) | Completed: 8,000/8,000 scores and identical replay | [Record](../../results/post_thesis/llm_judge/minicheck_halubench_cluster_20260928.json) |
| [Original frozen Qwen run](HALUBENCH_JUDGE_RUN.md) | Parent stopped at 7,652 valid scores; separate continuation completed the population | [Record](../../results/post_thesis/llm_judge/halubench_budget_stop_20260929.json) |
| [Explicit Qwen continuation](HALUBENCH_CONTINUATION.md) | Completed: 348/348 child scores, 8,000/8,000 combined, identical replay | [Record](../../results/post_thesis/llm_judge/halubench_continuation_complete_20260929.json) |

## Engineering reference and public artifacts

- [Historical probability-judge interface](LEGACY_INTERFACE.md): the generic
  JSON runner, parser, cache/retry behavior and accounting semantics.
- [Results index](../../results/post_thesis/llm_judge/README.md): final findings
  and dated development records.
- [Figure reproduction](../../figures/post_thesis/llm_judge/README.md): CPU-only
  plotting from committed aggregate metrics.
- [Verified inference accounting](../../results/post_thesis/llm_judge/efficiency_accounting_20261102.md):
  timing scopes, token usage, unknown interrupted usage and cost limitations.

The frozen release tag is `post-thesis-llm-judge-v1` at `c1b1cd6`. Subsequent
navigation/status edits do not change that tag, frozen inputs, decisions or metrics.
