# Post-thesis LLM judge protocol

**Protocol ID:** `post_thesis_llm_judge_v1`

**Stage:** design recorded; offline interface/parser and resumable runner
implemented. Qwen3-32B on a self-hosted vLLM/H200 profile is the initial development
candidate. The operator completed two six-example H200 synthetic runs and a
cache reuse check; a reproducible absence-claim failure remains. TRAIN pilot
manifest is prepared, the cluster token audit passed, and the first bounded
50-example TRAIN scoring run is complete. Inspected development outcomes
motivated a separate v2 prompt revision. Its token audit and 50-example scoring
run are complete. Paired development results improve ranking but leave clear
errors. A completed [synthetic diagnostic](DIAGNOSTIC.md) found sharply lower
probabilities for embedded unsupported claims. A [binary-output diagnostic](BINARY_DIAGNOSTIC.md)
matched all ten synthetic expectations. The next step is a separate
[binary TRAIN token audit](BINARY_PILOT.md), without changing the binary formulation
or v2. Binary outputs do not resolve the final continuous-score design.
The final benchmark configuration remains pending.
See [PILOT.md](PILOT.md) for the pinned selection and grouping limitations.
Study design unchanged.

**Scope:** research conducted after thesis submission; excluded from submitted
thesis results. The defence is pending.

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

Planned structured response:

```json
{"unsupported_probability": 0.73}
```

The value is a finite numeric estimate in `[0, 1]`, with higher meaning more
likely unsupported. A verbalized probability is not assumed calibrated.
The parser must reject booleans, missing/duplicate keys, invalid JSON,
non-finite/out-of-range values, and API refusals rather than manufacture a score.

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

## 10. Decisions still required before a paid pilot or final run

| Decision | Freeze point |
| --- | --- |
| Judge provider/model, available version pin, API budget | Before paid pilot |
| Pilot IDs/seed, initial prompt, input limits, bounded retry policy, concurrency | Before paid pilot; record pilot revisions |
| Final prompt and inference settings | After pilot, before final evaluation |
| Disjoint development IDs/count, threshold rule, optional calibration method | Before fitting thresholds/calibration |
| Metric binning, bootstrap/grouping, context control, missing-score reporting | Before final evaluation |
| Exact historical submission snapshot | Before creating any thesis-submission tag or claiming snapshot identity |

This file freezes the study design baseline, not the unresolved operational
values above. Record revisions in Git and retain the first frozen evaluation.

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

### Initial development backend

The first candidate is `Qwen/Qwen3-32B`, BF16, non-thinking, temperature zero,
with one H200 and sequential requests. Exact model/tokenizer revisions and the
vLLM version are in [configs/qwen3_32b_h200.json](configs/qwen3_32b_h200.json).
These are development settings, not a final evaluation freeze or a performance
claim. The 32,768-token serving limit includes the complete formatted input and
output allowance; overlength examples fail explicitly without truncation.
Audit benchmark lengths before selecting the final evidence-visibility policy.

The six manually constructed examples in `smoke.py` are infrastructure checks,
not benchmark examples or a substitute for the 50–100-example TRAIN pilot.
Synthetic GPU execution is recorded in the [post-thesis smoke observations](../../results/post_thesis/llm_judge/synthetic_smoke_20260919.md).
The first 50-example TRAIN selection rule is recorded in [PILOT.md](PILOT.md).
Shared-context groups are a source proxy; native-source overlap auditing remains
required before claiming strict source-disjoint threshold development. The
[first TRAIN pilot report](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_20260919.md)
preserves v1 findings. See [PILOT.md](PILOT.md) for its immutable artifact hashes
and committed 50-example/one-attempt/600-second cumulative client budget.
[Prompt v2](PROMPT_V2.md) completed its separate 50-example/one-attempt/600-second
scoring run; see the [paired report](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_v2_20260919.md).
Unused pilot allowances are not transferred to the synthetic diagnostic.
The token audit's separate 50-request/300-second invocation limits are not
the scoring budget.
