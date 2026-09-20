# Post-thesis frozen-judge evaluation protocol

Version 1, recorded after development fitting and before collecting judge TEST
scores. This extension is not part of the submitted thesis. Historical thesis
benchmark results are already known; these are not previously unseen benchmarks.

## Fixed judge and decision rule

The [completed fit](../../results/post_thesis/llm_judge/development_fit_cluster_20260920.md)
and [decision configuration](configs/frozen_judge_v1.json) fix Qwen3-32B, model
revision, primary A/B mapping, prompt hash, raw-logprob extraction, BF16/non-thinking
serving profile, accepted calibration and raw threshold. Preserve all fields.

- Rank by unrounded float64 `logP(B) - logP(A)`.
- Report raw sigmoid scores and the accepted calibrated scores separately.
- Predict unsupported at raw margin `>= -1.25`, equality positive.
- Keep the raw-margin zero reference secondary; never choose between policies
  using benchmark results.
- No prompt/model/mapping selection, recalibration, new threshold, or HaluBench
  adaptation. Later changes motivated by TEST are separate exploratory studies.

This freezes model and decision choices. Dataset identities, aligned baseline
caches, token audits and exact bounded execution manifests are still pending.
This document and the freeze checker make **zero generation calls**.

## Benchmark input manifest before inference

RAGTruth: preserve all 2,700 rows of the original processed TEST partition,
`output` as answer and `context` as evidence. Positive label is
`evident_conflict > 0 OR baseless_info > 0`; labels remain offline. Pin dataset
revision, downloaded file hash, row IDs/order, answer/context hashes and mapping
code. Join native response/source IDs and existing exact-overlap component IDs.
Retain the 12 previously identified TEST-linked TRAIN exclusions; do not remove
TEST rows to improve scores. Preserve historical pilot/fit reservations.

HaluBench: pin an upstream revision whose content/order can be established against
the existing caches; remove `source_ds == RAGTruth` and use the existing canonical
`test_filtered_indices` (8,000 rows) in
`results/cross_domain/halubench_groupfix/halubench_group_split.json` (SHA256
`9d78af5b4623893cfc7d761dc6be4c2e0f64278faaba11b2d10e6f829e67bb4b`).
No new outer split or use of the 6,000-row adaptation pool. Map `answer`/`passage`
to model input and FAIL/PASS to offline labels 1/0; reject unexpected labels.
Retain source and canonical grouping metadata offline. An index match alone does
not establish text identity: verify original IDs and text hashes or equivalent
content-level provenance before using baseline scores.

Each model request includes only the frozen instruction plus answer/context.
Neither original question as a separate field nor labels, task, generator,
benchmark/source names or other metadata enter the prompt. Naturally occurring
text inside answer/context remains untouched.

Run CPU tokenization with the frozen tokenizer/template, no truncation, 32,768
model limit and one output token. Record every overlength input and complete
coverage before fixing request references and a cumulative execution budget.
Do not silently omit overlength rows or alter prompt/evidence to force a fit.
If any occur, resolve them through an explicit protocol amendment before
inference and without inspecting judge TEST scores.

## Baselines and provenance

Main comparisons are S4 DeBERTa, MiniCheck-7B, metadata-free S2+S4, and the judge.
Use one named, pinned variant of each. Record checkpoint/software, input hashes,
score direction/transformation, threshold value/comparator, training-side
threshold provenance, truncation/chunking and actual evidence visibility.
Verify per-example IDs/text alignment; aggregate results cannot supply paired
metrics. Stop a comparison when alignment is unproven rather than guessing.

The existing fusion decomposition explicitly separates `S2_S4` (metadata false,
historical RAGTruth AUROC about 0.8494, OOF threshold 0.45) from
`S2_S4_metadata` (metadata true, AUROC about 0.8749, OOF threshold 0.40).
Use the former for the main deployment-oriented comparison. The latter is an
optional, separately labelled in-distribution metadata-aware result, not an
interchangeable baseline. These historical numbers are provenance references,
not new evaluation results.

Audit MiniCheck support-score inversion and equality handling explicitly:
a threshold in support space is not numerically the same threshold in
unsupported space. Preserve each baseline's training-selected operating policy.
Do not take a test-maximizing threshold or reuse a value without its transform
and comparator. Existing baseline exposure to RAGTruth training may include the
judge's development rows; disclose unequal training/development exposure.

The main comparison is between practical systems with their original evidence
limits. If evidence visibility differs, do not attribute differences solely to
architecture. A common-evidence control would need separate predictions and
its own plan. Do not overwrite any thesis cache, figure or table.

## Metrics, coverage and uncertainty

Use Python 3.12, NumPy 1.26.4, SciPy 1.14.1 and scikit-learn 1.5.2 for the later
metric implementation. Positive means unsupported throughout.

- AUROC: `roc_auc_score(y, raw_margin)` for the judge, oriented continuous scores
  for each baseline. Tied scores receive the standard tied-rank treatment.
- AUPRC means **average precision**, `average_precision_score`; do not substitute
  trapezoidal integration. Store the unambiguous key `average_precision`.
- F1, precision, recall and confusion counts use the frozen policies. Zero
  denominators yield zero; report class counts. One-class samples have undefined
  AUROC and AP for this protocol, recorded as null with a reason.
- Brier and ECE use the already registered ten-bin implementation, including
  zero/one endpoints and empty-bin counts. Report raw and calibrated judge
  reliability together; never choose the better one after seeing TEST.
- Report each benchmark separately, with sample counts/prevalence. Include all
  source-specific HaluBench results and RAGTruth task/generator breakdowns as
  descriptive slices; no post-hoc choice of only favorable slices.

Keep the original attempted denominator. Failed/invalid requests have missing
scores, not 0, 0.5 or an invented label. Report individual valid-subset metrics
with coverage, then use the same all-main-systems intersection for primary paired
comparisons; disclose the shared N and class counts. Pairwise larger intersections
may be supplementary and must be labelled. No silent complete-case claims or
substitution of a different model. Full coverage is the goal, not an assumption.
Bounded request failures and interruption handling belong to the execution plan.

Uncertainty: 2,000 cluster-bootstrap replicates, NumPy Generator(PCG64(20260920)),
resampling K whole groups with replacement from K sorted eligible group IDs. Keep
all rows and multiplicity within a sampled group; equal response weighting within
each replicate. RAGTruth groups use the audited native/exact-overlap components.
HaluBench groups use the canonical normalized source/question/passage relation,
merged when exact nonempty passages are shared. Do not change canonical split
membership when defining these resampling groups. Freeze the resulting group map
before judge inference; partial/fuzzy document overlap remains unproven.

Use identical resampled rows for all compared systems, with no refitting or
threshold selection within replicates. Preserve the original 2,000 draws; for
undefined metrics skip that draw only for the affected metric/pair and report
valid/invalid counts. Report no interval if fewer than 1,900 draws are valid.
Use 2.5/97.5 percentile endpoints with NumPy `method='linear'`. For differences,
report judge minus baseline and specify whether larger or smaller is better.
Intervals are nominal, unadjusted for multiple comparisons, conditional on fitted
systems and this test sample. They do not quantify model-training variation or
prove absence of all document dependence. Source/task slices remain descriptive
without additional bootstrap intervals in this version.

## Efficiency and release artifacts

Keep fresh attempt latencies, retries, failed-attempt time, token usage, client
wall time, concurrency and server startup/idle duration distinct. Cached reads
are not fresh inference latency. Report median/p95 and aggregate measured time,
with timing method and warmup status. Historical costs measured on different
hardware are descriptive, not a controlled speed comparison.

Compute monetary cost only with a documented hourly rate and billable duration
(or dated provider prices and actual token totals). Unknown cost/usage remains
unknown, not zero. No judge cascade or escalation tuning is included here.

Store manifests and private raw responses under `.artifacts/post_thesis/llm_judge`.
Publish aggregate findings under `results/post_thesis/llm_judge` and figures under
`figures/post_thesis/llm_judge`. Pin all evaluated artifact hashes in the eventual
result record. Freeze the evaluation implementation and bounded run configuration
before collecting judge TEST scores. Release/tag and final README findings follow
completed evaluation, not this development result.

Input-alignment implementation note (before judge TEST scoring): [the native/processed audit](TEST_ALIGNMENT.md) identified a one-ASCII-space formatting difference for six responses to source 14347. Both exact context hashes, IDs and offset are pinned. Original processed inputs and labels are preserved; other context mismatches fail. This does not alter the metric, model, calibration, threshold or split rules.

Pre-inference progress: TEST input alignment, legacy cache/threshold checks and metadata-free fusion reconstruction are complete, with explicitly unresolved historical input/checkpoint/evidence provenance. The [TEST token audit](TEST_TOKEN_AUDIT.md) can proceed as a CPU length check without promoting those baselines to comparison-ready status. This implementation note changes no comparison acceptance rule, model, metric or threshold.
