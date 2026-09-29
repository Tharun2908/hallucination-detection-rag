# Post-thesis LLM judge results

Results collected after thesis submission. These post-thesis artifacts are
excluded from the submitted thesis results. **Frozen RAGTruth TEST evaluation is complete.** The judge trails all three baselines on AUROC, AP and F1; see the [findings](ragtruth_frozen_evaluation_20260928.md) and [reported metrics](ragtruth_frozen_evaluation_20260928.json). The canonical HaluBench 8k cross-domain evaluation is in progress. [Fresh S2 inference and replay](s2_halubench_cluster_20260928.json) are complete; [frozen CPU fusion application and replay](halubench_fusion_cluster_20260928.json) are also complete. [Fresh MiniCheck inference and replay](minicheck_halubench_cluster_20260928.json) are complete. The [bounded frozen Qwen run](../../../post_thesis/llm_judge/HALUBENCH_JUDGE_RUN.md) is prepared; cross-domain metrics remain pending.

The [HaluBench overnight stop](halubench_budget_stop_20260929.json) preserves 7,652 valid scores and the identical offline replay. An [explicit 348-request continuation](../../../post_thesis/llm_judge/HALUBENCH_CONTINUATION.md) has [completed with identical replay](halubench_continuation_complete_20260929.json). All 8,000 judge scores are available; [CPU evaluation](../../../post_thesis/llm_judge/HALUBENCH_EVALUATION.md) is implemented and benchmark metrics remain pending.

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

- [Accepted development fit and cached replay](development_fit_cluster_20260920.md): positive-slope calibration, raw threshold -1.25, reliability diagnostics on both reserved arms; no TEST results. [Console-derived JSON](development_fit_cluster_20260920.json).

- [Frozen fit checker](frozen_fit_verification_20260920.json): operator-reported checksum match; zero refitting/model calls or source-file changes.
- [Local RAGTruth TEST alignment](test_alignment_local_20260920.json): all 2,700 native answers, 2,694 exact contexts and six pinned single-space differences; cluster preparation and baseline provenance pending.

- [Legacy baseline artifact inventory](legacy_baseline_inventory_20260920.json): located six full-size caches and six S4 config paths; cluster TEST manifest hash independently reconstructed. Cache bytes and checkpoint linkage remain unverified by the report author; on-pod provenance audit pending.

- [Completed cluster legacy-baseline audit](baseline_provenance_cluster_20260920.md) and [operator observations](baseline_provenance_cluster_20260920.json): complete caches, matching OOF assignments and historical TRAIN thresholds; legacy input/model provenance remains incomplete. Metadata-free fusion reconstruction is prepared, not yet run on cluster caches.

- [Completed metadata-free fusion reconstruction and replay](fusion_recovery_cluster_20260920.md), with [operator observations](fusion_recovery_cluster_20260920.json). The historical TRAIN reference matches; 2,700 predictions were saved. No new TEST metrics; legacy comparison provenance remains limited.

- [Completed TEST token audit and replay](test_token_audit_cluster_20260920.md), with [observations and independent hash reconstruction](test_token_audit_cluster_20260920.json): 2,700 inputs fit, no generation or benchmark metrics.

- [Completed cluster evaluation-math check and CI fix](evaluation_math_cluster_20260920.md): artificial numerical results and identical replay verified; baseline checkpoint compatibility is next.

- [Completed S4 checkpoint CPU compatibility](s4_checkpoint_cluster_20260920.md): clean loading and four saved-tokenizer matches; both answer and context truncation observed. Fresh S4 TEST inference is now complete (see below).

- [Fresh S4 TEST run and legacy-score agreement](s4_fresh_test_cluster_20260920.md): 2,700 valid scores, no failures, explicit truncation counts and maximum score difference below 0.000053; no label-based TEST metrics. [Exact agreement record](s4_legacy_score_agreement_20260920.json). Fresh S2 inference is prepared in [the run guide](../../../post_thesis/llm_judge/S2_TEST_RUN.md).

- [Fresh S2 TEST run, replay and legacy-score agreement](s2_fresh_test_cluster_20260928.md): all 2,700 scores valid, identical replay hash, sentence counts matched throughout and maximum raw-min difference below 0.000055. [Exact agreement record](s2_legacy_score_agreement_20260928.json). The frozen fusion can now be applied to fresh features without refitting; cluster application remains pending.

- [Frozen fusion on fresh S2/S4 features](fresh_fusion_cluster_20260928.md): all 2,700 scores valid, identical replay hash and no decision changes at the frozen 0.45 threshold. No fitting or TEST label metrics. MiniCheck preparation is next.

- [New-pod MiniCheck environment inventory](minicheck_environment_20260928.json): supplied payload hash independently reproduced; GPU free, model snapshot present, cached bytes/live compatibility not yet verified. The [synthetic check guide](../../../post_thesis/llm_judge/MINICHECK_LIVE_CHECK.md) records a locally observed tokenizer incompatibility and the explicit saved-tokenizer adapter.

- MiniCheck synthetic v1 startup failed after tokenizer checks and weight loading; replay made no new attempt. Report SHA256: `525e75192fb7b2e588a3b3e659228eaf30ece5478280e5c1c1a30da53c50ec89`. No synthetic scores returned. The [v2 sampler correction](../../../post_thesis/llm_judge/MINICHECK_LIVE_CHECK.md) is prepared, not yet live-verified.

- [MiniCheck v2 synthetic compatibility passed](minicheck_synthetic_v2_20260928.md): 14 encoding checks, six generation requests, four matching verdicts, cached replay without another model attempt. [Fresh TEST execution](../../../post_thesis/llm_judge/MINICHECK_TEST_RUN.md) is prepared but not yet run.

- [Fresh MiniCheck TEST completion and replay](minicheck_fresh_test_cluster_20260928.md): 2,700 valid examples, 18,935 sentence requests, no truncated prompts, identical replay hash. All planned fresh baseline predictions are present. The subsequent comparison is recorded below; the frozen judge TEST run remains pending.

- [MiniCheck fresh/legacy score drift](minicheck_legacy_score_agreement_20260928.md): mean absolute difference 0.02947, 100/2,700 changed decisions at the unchanged historical TRAIN threshold. No label metrics or attribution of cause. The [fresh comparison manifest](../../../post_thesis/llm_judge/COMPARISON_MANIFEST.md) records the fresh-runtime scope and retained provenance limits; cluster verification is recorded below.

- [Fresh baseline comparison manifest verified](fresh_comparison_manifest_20260928.md): all three baselines aligned on 2,700 examples / 450 groups, fixed thresholds and limitations preserved. [Bounded Qwen TEST execution](../../../post_thesis/llm_judge/TEST_LABEL_RUN.md) is prepared, not yet executed.

- [Frozen Qwen RAGTruth TEST run and replay](qwen_ragtruth_test_cluster_20260928.md): 2,700/2,700 valid scores, complete usage, zero new replay attempts and identical report hash. [CPU evaluation](../../../post_thesis/llm_judge/TEST_EVALUATION.md) is implemented under the unchanged numerical contract; [cluster findings](ragtruth_frozen_evaluation_20260928.md) now record the first frozen TEST comparison.

- [Canonical HaluBench input preparation](../../../post_thesis/llm_judge/HALUBENCH_INPUTS.md) and [local CPU counts](halubench_inputs_local_20260928.json): saved 8k membership preserved; exact-passage overlap disclosed, legacy comparison readiness unresolved. Pod manifest and inference remain pending.

- [HaluBench pod manifest](halubench_inputs_cluster_20260928.json): all 8,000 canonical rows preserved; exact manifest hash independently reconstructed. [CPU token audit](../../../post_thesis/llm_judge/HALUBENCH_TOKEN_AUDIT.md) is implemented; pod token counts and inference remain pending.

- [Local HaluBench token audit](halubench_token_audit_local_20260928.json): 8,000/8,000 inputs fit, 7,794,485 input tokens, maximum 7,298; identical zero-new-input-tokenization replay. Pod audit hash and live inference remain pending.

- [HaluBench pod token audit and identical replay](halubench_token_audit_cluster_20260928.json): all 8,000 inputs fit; zero new replay tokenizations. [Bounded fresh S4 HaluBench inference](../../../post_thesis/llm_judge/HALUBENCH_S4_RUN.md) is implemented with the unchanged RAGTruth checkpoint and inference policy; live S4 run remains pending.

- [Completed fresh S4 HaluBench run and identical replay](s4_halubench_cluster_20260928.json): all 8,000 scores valid; 850 contexts and zero answers truncated. [Fresh S2 plan](../../../post_thesis/llm_judge/HALUBENCH_S2_RUN.md) pins the [locally verified workload](s2_halubench_workload_local_20260928.json), including 1,886 empty-pair cases under the unchanged original policy. No new benchmark metrics.
