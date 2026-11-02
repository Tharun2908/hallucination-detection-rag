# Post-thesis LLM judge protocol

**Protocol ID:** `post_thesis_llm_judge_v1`

**Current stage — complete:** the final label-score judge, calibration and
operating threshold are frozen. Both benchmark evaluations, cached replay,
selected qualitative review and inference accounting are complete. The release
is tagged `post-thesis-llm-judge-v1` at `c1b1cd6`.

The sections below retain the study's design boundaries. Later protocols specify
the implemented [label score](CONTINUOUS_SCORE_PROTOCOL.md),
[development fit](CALIBRATION_THRESHOLD_PROTOCOL.md) and
[benchmark evaluation](EVALUATION_PROTOCOL.md). The
[documentation index](DOCUMENTATION_INDEX.md) links each development/run guide
to its completed outcome; earlier candidates are not outstanding experiments.

**Scope:** research conducted after thesis submission, excluded from submitted
thesis results. Preserve the thesis artifacts and their separate provenance.

## 1. Question, claim, and baseline

Research question: under a fixed evaluation protocol, when does a prompted LLM
judge improve cross-domain faithfulness detection, and which failures remain
shared with trained verifiers?

Start with one judge and one prompt. The contribution is the controlled
comparison and failure analysis, not a claim that LLM judging itself is new.

The implementation base is recorded in [baseline_manifest.json](baseline_manifest.json).
Its commit and artifact hashes identify existing comparison material. They do
not identify the original thesis-submission snapshot. That historical snapshot
must be verified separately before it is tagged or described as the submission.

No extension output may overwrite canonical thesis scores, tables, or figures.
Keep later engineering corrections and extension findings distinguishable from
submitted-thesis evidence.

## 2. Input and prediction contract

Each request contains only a fixed instruction/prompt plus:

- `answer`: generated response text;
- `context`: supplied retrieved evidence text.

Never interpolate labels, a gold answer, dataset/source names, task type,
generator identity, or the original question as a separate input. The context
and answer may naturally contain such text; do not rewrite benchmark content
merely to remove it. IDs and metadata are retained outside the request for
alignment and analysis.

The target is **whether the answer contains at least one factual claim that is
unsupported by, or contradicts, the supplied context**. It is not the fraction
of unsupported claims, severity, answer quality, or real-world truthfulness.

### Final benchmark score

The final judge uses the frozen `faithfulness-label-score-v1` prompt and one
class token: A=supported, B=unsupported. Rank by raw `logP(B) - logP(A)`;
apply the frozen TRAIN-fitted calibration separately. The operating rule is
raw margin `>= -1.25`. Exact identities and parameters are in
[configs/frozen_judge_v1.json](configs/frozen_judge_v1.json).
This replaces the initial verbalized-probability candidate for final evaluation.

### Historical probability interface

The initial development interface requested:

```json
{"unsupported_probability": 0.73}
```

Its finite numeric value in `[0, 1]` was not assumed calibrated. Its parser
rejects booleans, missing/duplicate keys, invalid JSON, non-finite/out-of-range
values and API refusals rather than manufacturing a score. This preserved
[legacy interface](LEGACY_INTERFACE.md) is not the final label-score scorer.

Do not automatically exclude disclaimers or evidence-absence claims. Assess any
factual assertions they contain against the context. A terminal API failure,
invalid input, and an uncertain but valid probability estimate are distinct.
Treat instructions embedded in benchmark text as data, not judge instructions.

## 3. Dataset and label contract

Preserve the repository's existing target direction: `1 = hallucination or
unsupported`, `0 = supported`.

| Dataset | Existing input mapping | Existing binary target | Evaluation partition |
| --- | --- | --- | --- |
| RAGTruth (`wandb/RAGTruth-processed`) | `output` -> answer; `context` -> context | `evident_conflict > 0 OR baseless_info > 0` in `hallucination_labels_processed` | Original 2,700-example test partition |
| HaluBench (`PatronusAI/HaluBench`) | `answer` -> answer; `passage` -> context | `FAIL` -> 1; `PASS` -> 0; reject unexpected labels | Existing canonical 8,000 filtered test indices |

These mappings follow [S4 preparation](../../signals/signal4_finetune.py) and
[HaluBench preparation](../../cross_domain/halubench_curve_groupfix.py).
They do not establish that the datasets' annotation definitions are identical.
Document those differences when interpreting transfer.

For HaluBench, remove rows with `source_ds == "RAGTruth"` before resolving the
canonical filtered indices. Reuse
`results/cross_domain/halubench_groupfix/halubench_group_split.json`.
Do not sample a new outer split or use its 6,000-example adaptation pool to tune
the judge in the zero-shot transfer experiment.

Before scoring, pin dataset revisions and create an ID/content-hash manifest.
Validate row counts, unique IDs, split membership, and cache/text alignment.
Index equality alone is insufficient if the upstream dataset order changed.
Record source/group metadata offline for analysis; never send it to the judge.
Stop on an alignment discrepancy instead of silently pairing different examples.

## 4. Development and freeze sequence

1. Select 50–100 RAGTruth TRAIN examples for prompt development. Record selection
   method, seed, IDs, and input hashes. Label inspection here makes this development
   data. Repeat a small subset to measure stability, accounting for its extra cost.
2. Use a disjoint source TRAIN subset for operating-threshold fitting and any
   optional probability calibration. Keep shared source documents/duplicate inputs
   together when allocating development subsets. Freeze the exact manifest,
   subset sizes, and overlap checks before the corresponding judging runs.
3. Inspect support interpretation, absence/numerical claims, malformed outputs,
   saturation, latency, and cost. Record prompt revisions and their reasons.
4. Freeze model/version, prompt/hash, inference configuration, input-length policy,
   retry policy, and evaluation rules before either final benchmark is scored.
5. Select the judge's threshold using source development data only. Record the
   objective, search rule, tie rule, and selected value before TEST. Raw scores
   are the primary output. If calibration is fitted, record its method/data and
   report calibrated results separately.
6. Transfer the same frozen configuration and numeric threshold to HaluBench.
   Any later HaluBench-label-based tuning is a separately named adaptation study.

Existing thesis test outcomes are already known. This is a new protocol fixed
before collecting judge test scores, not a claim that the historical benchmarks
have never been inspected. Any later change motivated by test results must be
logged as exploratory and cannot overwrite the original frozen run.

## 5. Comparisons and evidence visibility

Main systems: S4 DeBERTa, MiniCheck-7B, metadata-free S2+S4, and the LLM judge.
Metadata-aware fusion is optional and explicitly identified as benchmark-aware.
The distilled student may be an additional comparator once aligned scores exist;
do not claim a distilled-verifier comparison without those results.

Use existing baseline thresholds chosen on training-side data. Record score
orientation and provenance for each baseline; invert support scores where
necessary before comparing hallucination-positive rankings. Missing RAGTruth
per-example caches must be obtained or regenerated before paired analysis;
aggregate JSON results cannot substitute for them.

Record context/answer lengths, tokenizer, truncation/chunking strategy, and exact
text seen by each verifier. Distinguish practical full-context system comparisons
from a control using the same evidence text, where feasible. Any changed evidence
control needs its own baseline predictions. Do not describe unequal evidence
visibility as an architecture-only comparison.

## 6. Evaluation and failure accounting

- Discrimination: AUROC and AUPRC, separately for each benchmark.
- Operating point: F1, precision, recall at the frozen source-selected threshold.
- Probability estimates: Brier score, ECE and reliability plots; document binning
  and distinguish raw from calibrated results.
- HaluBench: also report source-specific metrics, sample counts, and prevalence.
- Uncertainty: paired bootstrap differences on aligned examples, preserving
  relevant grouping. Freeze resampling unit, replicates, seed, and interval
  definition before TEST. Report nominal intervals without implying correction
  for multiple comparisons or uncertainty over unseen model-training runs.
- Execution: successful-scoring coverage, error categories, retries, latency
  distribution, concurrency, input/output tokens, and total/per-example API cost.
  Record price schedule/date and distinguish estimated from billed cost.

After bounded retries, preserve a terminal failure record with no score. Never
silently drop it, turn it into 0.5, or substitute a different model. Report total
attempted and valid counts; give paired metrics on shared successfully scored
examples and disclose that failure-related selection can bias that subset.
Cached local reads are not fresh API latency. Include retry costs and unknown
usage explicitly instead of treating unreported usage as zero.

## 7. Engineering and output provenance

Persist per-attempt records incrementally and resume completed valid requests.
The request/cache key includes exact input, prompt version/hash, requested model,
provider/API configuration, and inference parameters. Store the returned model
identifier with each answer; never use shared mutable last-model state.
Reproducibility may be limited when providers offer mutable aliases only.

Every run summary includes `study_stage: post_thesis`, protocol ID, Git commit,
model and prompt identities, dataset revisions/manifests, configuration, scoring
coverage, and cost/latency accounting. Every figure identifies the post-thesis run.
Store outputs only under the locations in [README.md](README.md).

## 8. Error analysis and evidence sensitivity

Analyze disagreements even if the judge underperforms. Inspect unsupported
additions, contradictions, partial support, long-context cases, evidence-absence
claims, numerical claims, paraphrases, and source-specific failures. Use a
recorded selection method and human review; the evaluated judge is not its own
ground-truth adjudicator.

Construct a small manually checked set of pairs: keep an answer fixed and change
evidence so a particular claim's support changes. Document which other claims
remain, the resulting response-level label, and expected score direction.
Pairs derived from TEST are post-hoc diagnostics, not development examples or
an independent confirmation set. Context shuffling alone does not prove a
mechanism. Unknown foundation-model benchmark exposure also limits unseen-data
and causal claims.

## 9. Optional cascade and release

After standalone evaluation, consider lightweight S2+S4 -> LLM escalation.
Select routing rules on development data, then report quality, escalation rate,
latency, and API cost on held-out data. The best test-observed escalation rate is
not an independently validated deployment optimum.

Publish positive and negative findings in a separate post-thesis report. Link the
frozen run configuration, prompt, artifact hashes, and limitations. A later
release/tag must explicitly identify the extension and preserve thesis provenance.

## 10. Recorded freeze points

| Decision | Completed record |
| --- | --- |
| Judge model, tokenizer and serving profile | [Frozen judge](configs/frozen_judge_v1.json); separate bounded run plans |
| Adaptive pilot membership, prompts and budgets | [Development history](DOCUMENTATION_INDEX.md#historical-diagnostics) |
| Final prompt and inference settings | [Continuous-score protocol](CONTINUOUS_SCORE_PROTOCOL.md) |
| Separate calibration and threshold membership | [Development reservation](DEVELOPMENT_RESERVATION.md) |
| Calibration method, threshold rule and fitted values | [Fit protocol](CALIBRATION_THRESHOLD_PROTOCOL.md), [accepted fit](../../results/post_thesis/llm_judge/development_fit_cluster_20260920.md) |
| Metrics, bootstrap groups, evidence visibility and missing scores | [Evaluation protocol](EVALUATION_PROTOCOL.md) and completed benchmark reports |
| Post-thesis release | Tag `post-thesis-llm-judge-v1` at `c1b1cd6` |

The exact historical thesis-submission snapshot remains unverified. The extension
tag does not claim to identify that submission. Operational configurations and
earlier status fields remain preserved as historical records; this documentation
update does not grant a new inference budget or change the evaluated systems.

### Backend deployment clarification

The backend may be a hosted API or a self-hosted open-weight model. This does not
change the answer/context input contract, development splits, evaluation rules,
or post-thesis boundary. Select one backend for the initial pilot; GPU access
does not itself select a model or authorize a full benchmark run.

For self-hosting, pin the checkpoint and tokenizer revisions, precision or
quantization, serving engine/version, generation configuration, GPU type/count,
and concurrency. Record token usage and latency plus measured GPU-hours and any
declared rental-cost basis. For APIs, record the price schedule/date and estimated
versus billed charges. Unknown cost remains unknown, not zero. The compute/time
budget replaces an API-spend budget when self-hosting; freeze the applicable
resource limit before the pilot.

## Historical implementation sequence

The initial probability interface used Qwen3-32B, BF16, non-thinking, one H200
and sequential requests. Its original 128-token profile is preserved in
[configs/qwen3_32b_h200.json](configs/qwen3_32b_h200.json). Structured-evidence
experiments used a separate 512-token profile. The final `label-score-v1`
profile generates one token. These are distinct request and run identities.

| Stage | Recorded outcome |
| --- | --- |
| Probability and binary pilots | Completed adaptive development; embedded-error dilution and persistent semantic misses motivated later diagnostics |
| Evidence schema v1/v2/v3 | Completed synthetic checks; structural validity and field order did not establish evidence correctness |
| Evidence prompt-v2 TRAIN pilot | 43/50 valid records; seven quote failures and mixed semantic findings; evidence tuning paused |
| Class-token compatibility and original TRAIN pilot | Completed transport checks and 50/50 valid pilot scores; uncalibrated development results retained |
| Native source and cross-split audits | Completed scoped exact-overlap checks; 312 TRAIN rows excluded from subsequent reservation |
| Reserved development arms | 600 calibration and 600 threshold rows, scored completely under separate bounded plans |
| Fitting and decision freeze | Positive-slope calibration accepted and raw-margin threshold -1.25 frozen before new TEST scores |
| RAGTruth and HaluBench | Completed fresh four-system inference, cached replay and frozen evaluation |
| Review, accounting and release | Completed selected-case review and verified accounting; tagged release preserved |

The [documentation index](DOCUMENTATION_INDEX.md) provides every original stage
guide and its result record. Historical commands describe their recorded revisions
and budgets; they do not authorize reopening development after TEST inspection.
No attempt failures, quote failures, prior scores or labels were silently repaired.

Native/exact-overlap component separation does not prove strict document
independence. Historical cache/checkpoint linkage, verifier training exposure,
MiniCheck fresh/legacy drift and unknown foundation-model exposure remain
limitations. The [final overview](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md)
explains their effect on interpretation. A cascade or another prompt/fit is
future research, not an unfinished part of this release.
