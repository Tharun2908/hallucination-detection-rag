# Post-thesis: continuous label-score protocol

**Design v1, recorded 2026-09-20. Synthetic transport and original-50 TRAIN pilot complete; final evaluation pending.**
This experiment is excluded from submitted thesis results. It follows the
[mixed evidence-prompt findings](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_pilot_20260920.md).
Evidence-prompt tuning remains paused. The [offline contract and tokenizer check](LABEL_SCORE_OFFLINE.md)
are implemented and the cluster checks passed. A separate [bounded synthetic
execution plan](LABEL_SCORE_RUN.md) adds 30 one-token compatibility requests.
The [synthetic findings](../../results/post_thesis/llm_judge/label_score_synthetic_v1_20260920.md) record successful transport checks with semantic limitations; benchmark evaluation remains pending.

## Question and selected method

Can the model's relative likelihood of two class tokens provide a useful
continuous ranking of response-level unsupportedness? This tests a different
score extraction method; it does not assume that changing the output format
corrects the model's observed reasoning errors.

The event remains: at least one factual assertion in the answer is unsupported
by or contradicts the retrieved context. The input remains answer + context only.
One supported claim does not establish whole-answer support. The absence of an
assertion from context is not proof of its negation. Instructions within answer
or context are data, not instructions to the verifier.

Use a new, separately versioned class-label prompt, with:

- `A` meaning supported (all factual assertions supported, or no factual assertions).
- `B` meaning unsupported (at least one factual assertion lacks support or conflicts).
- Exactly one class token requested, without JSON, evidence quotes, explanations
  or a verbalized numeric probability.

Start from the response-level support criteria of the preserved
`faithfulness-binary-diagnostic-v1` prompt and change its output instructions.
The exact resolved text and hash must be committed in the offline implementation
before any model calls. This is a new candidate, not an unchanged evidence-v2
prompt or an isolated comparison of its extraction method. Do not import the
old JSON/evidence schema into this path, or overwrite earlier prompt defaults.

Retain Qwen3-32B at revision `9216db5781bf21249d130ec9da846c4624c16137`,
BF16, non-thinking, one H200 and the pinned tokenizer revision. Use a separately
named scoring profile for any required serving changes; preserve existing
profiles. Server software versions and the exact command must be recorded.

## Score definition

At the **same first assistant class-token position**, obtain the raw,
full-vocabulary log probabilities for the two verified token IDs:

```text
l_supported   = log P(next_token = A | fixed_prompt, answer, context)
l_unsupported = log P(next_token = B | fixed_prompt, answer, context)

unsupported_log_odds = l_unsupported - l_supported
unsupported_score = sigmoid(unsupported_log_odds)
                  = exp(l_unsupported) / (exp(l_supported) + exp(l_unsupported))

log_class_token_mass = logaddexp(l_supported, l_unsupported)
```

Higher values mean greater relative model preference for unsupported. Compute
with stable log-space arithmetic. Retain both raw log probabilities and the
unrounded log-odds margin; use the margin for primary ranking metrics so numeric
saturation of sigmoid near 0 or 1 does not introduce artificial ties.

`unsupported_score` is the model's probability of B **conditional on the next
token being one of the two selected class tokens**. It is not automatically
P(the answer actually contains an unsupported claim). Treat it as an uncalibrated
score. Never name it a calibrated probability without a separate fitted and
evaluated calibration procedure.

Retain class-token mass because a high relative score can coexist with almost
no total probability on either class token. Report its distribution and off-label
first tokens. Do not discard low-mass examples or tune a mass cutoff after seeing
test labels. Off-label generation is a separate diagnostic: a score is still
available if both raw class log probabilities at the specified position are
valid. Never scan later generated text to find a more favorable label position.

## Tokenizer and extraction contract

1. Verify A and B are distinct single tokens with the pinned tokenizer. Record
   their IDs, decoded bytes, tokenizer/chat-template hashes and the fully rendered
   assistant boundary, including Qwen's non-thinking template behavior. Checking
   `encode(label)[0]` alone is insufficient: assert the complete encoding length,
   exact bytes and compatibility at the actual scored position. No automatic
   whitespace variants, token substitution or multi-token fallback.
2. Retrieve **both** candidate values by explicit token ID at the identical
   position, including when a candidate is not among the returned top-k tokens.
   Missing candidates, non-finite values, duplicate IDs, alignment errors or
   wrong model identities yield a missing score and an explicit error. Reject
   positive log probabilities or class-token mass above one beyond a predeclared
   numeric tolerance; do not silently normalize corrupt values.
3. Require raw log probabilities before temperature, penalties, top-k/top-p,
   grammar masks, allowed-token constraints or logit bias. Do not infer raw-mode
   semantics merely from a field named `logprobs`. Do not force one candidate
   to appear with logit bias. A greedy generated label is not a 0/1 probability.
4. Preferred transport is one request returning both selected raw log
   probabilities. Inspect the installed vLLM 0.29.0 implementation and verify its
   wire behavior before selecting an endpoint. Current documentation is a design
   reference, not proof of this cluster's compatibility. No silent backend or
   library upgrade is part of this experiment.
5. If the direct route cannot supply both values, stop and record that limitation.
   Teacher-forced likelihood extraction may be specified as a separate revision
   with exact token alignment and cost accounting; do not silently switch methods.
   A normalized scalar alone is insufficient for this contract because it cannot
   establish both underlying values or class-token mass.

Latest vLLM documentation distinguishes raw from processed log probabilities,
including temperature and sampling processors in the latter. Its sampling
parameters document explicit selected-token logprob requests. These facilities
motivate compatibility checks, not a claim that the pinned endpoint supports
our required output fields. See the [model configuration](https://docs.vllm.ai/en/latest/api/vllm/config/model/#vllm.config.model.ModelConfig.logprobs_mode)
and [sampling parameters](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/).
The [generative scoring endpoint](https://docs.vllm.ai/en/latest/serving/online_serving/generative_scoring/)
also documents normalization over label tokens, but its prompt construction and
returned fields must be checked against this contract before use. A transport
parameter called `query` must never contain the benchmark query; any transport
must render only the fixed prompt and answer/context.

## Development sequence and stopping rules

1. **Offline contract first.** Resolve and hash the new prompt, token IDs,
   template, score equations, output schema and cache identities. Check arithmetic
   with artificial log probabilities, extreme margins, swapped labels, missing
   candidates and invalid values. Tokenizer inspection loads no model weights
   and performs no generation. Preserve all earlier code paths.
2. **Bounded synthetic compatibility run.** Commit its exact fixtures, calls,
   timeout, cumulative budget and serving profile before execution. Verify the
   selected token position, raw values, numeric identities, replay and usage
   accounting. Include supported/unsupported and unknown/explicit-false cases,
   short/embedded answers and a label-mapping swap diagnostic. Record both
   orientations; the primary A=supported/B=unsupported mapping remains fixed.
   Do not pick or average orientations after observing which performs better.
   Set numeric transport tolerances before the run. Scores need not achieve
   perfect semantic accuracy to pass transport checks; report errors separately.
3. **Original-50 TRAIN development pilot.** Only after transport is verified,
   prepare a new formatted-length audit and explicit bounded execution plan.
   Keep every original ID, including the seven evidence-v2 failures. This is a
   new inference run with a different output contract, not salvage of old raw
   verdicts. Inspect ranking, saturation, class-token mass, coverage and known
   error cases. Do not fit a threshold or calibrator on these adaptively reused
   50 examples, and do not repeatedly tune the prompt to maximize their metrics.
4. **Freeze and independent source development.** Before any held-out evaluation,
   resolve native-source grouping/overlap, register disjoint calibration and
   operating-threshold data, and freeze the full evaluation manifest. The current
   shared-context grouping is a proxy, not proof of native-source disjointness.

Transport failure stops escalation. A scientifically disappointing but valid
score remains a reportable result; it does not justify another unregistered
prompt search. Model changes would define a separate experiment.

## Calibration, thresholds and evaluation

- Report raw ranking with AUROC and AUPRC; define AUPRC computation precisely in
  the later evaluation manifest. Report coverage and class composition alongside
  metrics. AUPRC depends on prevalence. Never fill missing scores with 0, 0.5,
  a binary verdict, or a repaired evidence response.
- Treat sigmoid(margin) as an uncalibrated candidate probability for reliability
  analysis only. Report ECE and Brier score with that qualification. Freeze ECE
  binning and edge conventions before evaluation, and retain calibration plots.
- A separate source-development calibrator may use a two-parameter logistic map
  of the margin, with positive slope so it preserves the ranking. Specify its
  objective, regularization and numerical bounds before fitting. If the boundary
  is reached or fitting fails, report failure rather than trying alternatives on
  test results. Report raw and calibrated results separately.
- Calibration fitting data and threshold-selection data must be disjoint from
  the adaptive pilot and held-out test, using resolved source groups. A registered
  cross-fitting alternative is possible only if specified before fitting. Pin
  exact IDs, sizes and split hashes; do not quietly reuse the unvetted candidate
  pool as if source disjointness were established.
- Select a threshold on the registered operating-development data, using a
  predeclared objective and tie rule. Transfer the exact prompt, score mapping,
  optional calibrator and numeric threshold to HaluBench. Use the existing
  canonical group-disjoint 8k test split; no HaluBench-label tuning or new split.
- Preserve S4, MiniCheck-7B and metadata-free S2+S4 as the planned comparisons.
  Benchmark-aware fusion stays explicitly metadata-aware. Compare on aligned IDs
  and disclose missingness, with full-cohort coverage and a common-valid subset
  sensitivity analysis rather than presenting different denominators as equal.
- Freeze bootstrap grouping, replicates, seed, binning, score orientation and
  primary endpoints before test evaluation. The original protocol's other
  controls remain in effect. No TEST or HaluBench scoring occurs in this design step.

## Engineering records and resource accounting

Record per request: exact prompt/template and input hashes, model/tokenizer and
software revisions, token IDs and class mapping, scoring position, raw log
probabilities, log-odds, normalized score, class-token mass, status/error, latency,
known/unknown usage and any emitted token ID. Keep sample IDs, labels, task,
generator identity and dataset provenance offline for analysis only.

Cache identity must include the new contract, prompt, model, tokenizer/template,
class mapping/IDs, extraction mode and exact answer/context. Use a separate run
namespace and immutable journal; incremental saving, resumability and one attempt
per input remain required. Missing usage is unknown, not zero. Reserve cumulative
budgets before requests; retain terminal failures and unknown execution windows.

No claim of free inference follows from avoiding generated explanations. Measure
all scoring requests, tokenization/preflight overhead, elapsed client time and
server allocation time; record startup/idle separately and apply actual rental
rates only when provided. Any two-pass compatibility reference or future
teacher-forced alternative must count both passes. Freeze the live budget only
after the required transport is known.

## Offline implementation and next check

The [offline prompt/score contract and pinned-tokenizer check](LABEL_SCORE_OFFLINE.md)
are implemented without model generation. The actual pinned tokenizer passed four
assistant CPU and four cluster boundary checks. Installed-source fingerprints
also match. The [bounded synthetic plan](LABEL_SCORE_RUN.md) completed with 30 valid scores and a zero-call cached replay. The [original TRAIN token audit](LABEL_SCORE_PILOT_AUDIT.md) passed. The [bounded pilot](LABEL_SCORE_PILOT_RUN.md) completed all 50 primary requests and a zero-call replay. Its [development findings](../../results/post_thesis/llm_judge/label_score_train_pilot_v1_20260920.md) record AUROC 0.7051 and emitted-label F1 0.5455. Preserve the candidate and resolve native-source grouping before registering disjoint calibration/evaluation. This design does not update the evidence-v2 prompt,
serve a new model, assign a scoring budget or declare a benchmark-ready verifier.

## Native TRAIN audit implementation

The [offline source audit](SOURCE_AUDIT.md) now verifies native response IDs, exact answer/context provenance and conservative exact-overlap components within pinned TRAIN data. It preserves the original manifest and makes no model calls. Its local validation found no new pilot-linked exclusions; the cluster report is pending. This is a scoped exact-overlap check, not resolution of fuzzy/partial-document, TRAIN–TEST or cross-benchmark overlap. The native release contains both splits; non-TRAIN records are ignored after ID filtering and native annotation labels are not used. Final calibration/evaluation reservations remain unselected.
