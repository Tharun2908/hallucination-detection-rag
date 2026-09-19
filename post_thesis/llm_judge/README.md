# Post-thesis experiment: LLM judge for RAG faithfulness

> The experiments below were conducted after thesis submission and are not part
> of the submitted thesis results.

**Status: H200 synthetic checks completed; pinned TRAIN pilot preparation
implemented. No benchmark judge scores have been collected.** The
[synthetic observations](../../results/post_thesis/llm_judge/synthetic_smoke_20260919.md)
record a repeatable absence-claim failure as well as successful cache reuse.
Next: [prepare the 50-example TRAIN pilot](PILOT.md), with labels kept offline
and shared-context group exclusions recorded.

The initial development candidate is **Qwen3-32B, BF16, one H200, non-thinking**.
See [SERVING.md](SERVING.md) for exact pins, installation, synthetic checks and
the distinction between measured server time and total rental cost.

The thesis defence is still pending. Preserve the thesis results and their
provenance throughout this extension.

## Research question

Under a fixed evaluation protocol, when does a prompted LLM judge improve
cross-domain faithfulness detection, and which failures remain shared with
trained verifiers?

Read [PROTOCOL.md](PROTOCOL.md) before implementation. The agreed study design is
recorded there; the model, prompt, budget, and run configuration are not yet
frozen for final evaluation.

## Boundaries and locations

| Content | Location |
| --- | --- |
| Protocol and judge implementation | This directory, `post_thesis/llm_judge/` |
| New summaries and run manifests | [results/post_thesis/llm_judge/](../../results/post_thesis/llm_judge/) |
| New figures | [figures/post_thesis/llm_judge/](../../figures/post_thesis/llm_judge/) |
| Local request cache and raw responses | `.artifacts/post_thesis/llm_judge/<run_id>/` by default; the runner preserves `post_thesis/llm_judge/` under a configured artifact root |
| Existing comparison-artifact identities | [baseline_manifest.json](baseline_manifest.json) |

The manifest records the cleaned-up repository baseline and hashes of existing
comparison artifacts. It is **not** a thesis-submission snapshot. The exact
submission commit remains unverified; do not label the current commit as that
snapshot or silently replace historical result files.

Raw benchmark content and API credentials do not belong in committed summaries.
Use IDs, hashes, configuration, and aggregate metrics for public artifacts.

## Sequence

1. Record the protocol and post-thesis boundary — complete.
2. Implement the judge interface, strict parser, offline tests, and resumable runner
   — complete, with an offline-tested vLLM adapter.
3. Select one model/backend and bounded API or GPU budget; implement the adapter
   and resource accounting, then pilot on 50–100 source TRAIN examples.
4. Freeze the prompt and use a separate source development subset for thresholding
   and any declared calibration.
5. Evaluate the frozen system on RAGTruth test and the canonical HaluBench test set.
6. Analyze disagreements and evidence sensitivity, including negative results.
7. Consider a cascade only if the standalone findings justify it.
8. Publish a separately labeled post-thesis report and release.

Model comparison and paid execution follow the protocol stages. The only real
model command currently supplied is the bounded synthetic smoke test.
`prepare_pilot.py` creates a TRAIN manifest without model calls; a bounded pilot
scoring command, formatted-length audit and final resource budget remain pending.

## Offline foundation

Requires Python 3.10+ and only the standard library. From the repository root:

```bash
python -S -m unittest discover -s tests -p "test_llm_judge*.py" -v
```

The existing CPU reproduction workflow also discovers these tests. All backends
in the tests are local fakes; no credentials, downloads, or API calls are needed.
Passing these tests validates engineering contracts, not judge quality.

| Module | Responsibility |
| --- | --- |
| `prompts.py` | Versioned `faithfulness-development-v1` prompt and content hashing |
| `parse.py` | Strict single-object JSON parsing, without extracting or repairing text |
| `judge.py` | Immutable requests, injected asynchronous backend, one attempt and per-response metadata |
| `runner.py` | Exact input manifest, sequential retries, cache reuse, coverage and attempt accounting |
| `storage.py` | Durable SQLite journal, atomic derived reports, cross-process run lock |
| `vllm_backend.py` | HTTP adapter, token counting, schema-constrained requests and response validation |
| `serve.py` / `smoke.py` | Explicit server launcher, resource windows, six synthetic examples |
| `prepare_pilot.py` | Pinned TRAIN file check, label-blind selection, group exclusions and private manifest |
| `../../tests/test_llm_judge.py` | Parser failures, input boundary, request identity, failure accounting, concurrent metadata isolation |
| `../../tests/test_llm_judge_runner.py` | Resume, retry budgets, real process death, locks, cache corruption, alignment and privacy boundaries |

The prompt is a **development candidate**, not the frozen evaluation prompt. It
defines the response-level target in the protocol and treats embedded instructions
as data. This is a prompt boundary, not a guarantee against prompt injection;
semantic behavior still requires the pilot and evidence-sensitivity checks.

`JudgeInput` accepts only `answer` and `context`. Their exact text is preserved
inside the user JSON message. Whitespace-only answers and non-string fields are
invalid; an empty context is allowed and supplies no evidence. Dataset adapters
must later record invalid rows explicitly rather than dropping them. IDs, labels,
gold answers, task/source metadata, and generator identity stay outside this input.

`JudgeConfig` requires explicit provider, requested model, and `api_config_id`.
The latter is a non-secret versioned identity for the adapter, endpoint,
API revision, and any provider-specific settings. Change it whenever those
settings or adapter behavior change. Never put an API key in it. Temperature
defaults to zero, but reproducibility still depends on the provider. The vLLM
profile now selects the development model; final evaluation settings remain unfrozen.

`build_request` includes a JSON response schema and hashes the exact messages,
prompt version/hash, configuration, schema, protocol ID, and request-contract
version. `run_judge` uses this identity for caching inside a run. Repeating the
lower-level `judge_once` directly still invokes the supplied backend again.
Mutable model aliases require a new run identity/cache policy when resolved
versions change; a request hash alone cannot detect a provider changing an alias.

### Backend contract and results

`VLLMBackend` implements `JudgeBackend.complete(request)` for the development
profile. An adapter must honor the request's messages, configuration and response schema;
normalize provider refusals/truncation into `BackendResponse.outcome`; and return
the reported model, response ID and token usage where available. Unknown model
and usage stay `None`. Reject unsupported settings explicitly. Do not silently
truncate evidence, add prompts, retry, or switch models. Synthetic GPU integration has been exercised; real-data
formatted lengths and pilot resource limits still need validation.

`await judge_once(item, config=config, backend=backend)` returns an immutable
`JudgeResult`. Its request carries requested-model and prompt provenance; its
response carries returned-model provenance. Nothing reads shared last-model
state. The full request/response may contain benchmark text: keep raw records in
the ignored post-thesis artifact directory, not public result summaries.

| Status | Score | Meaning |
| --- | --- | --- |
| `ok` | Numeric `[0, 1]` | Strictly valid output; not assumed calibrated |
| `invalid_output` | `None` | Malformed JSON, duplicate/extra keys, or invalid value |
| `refused` | `None` | Provider refusal, even if response text contains valid JSON |
| `incomplete` | `None` | Truncated/incomplete generation, even with parsable text |
| `backend_error` | `None` | Adapter-normalized operational failure, with sanitized error code |

Invalid input raises before the backend is called. Programming errors and async
cancellation propagate. `BackendError` preserves known token usage; unknown
usage remains unknown. A valid `0.5` is distinct from all failure states. There
are no automatic retries or replacement scores. Latency measures one backend
attempt's wall time, excluding request construction and parsing; it is not a
cache-read or full-run latency measure.

## Resumable runner

`run_judge` accepts an ordered collection of `Example(sample_id, JudgeInput(...))`,
an injected backend, explicit code/dataset revisions, and a safe `run_id`. The
sample ID and revision metadata never enter the model messages. It validates the
whole input collection before any call: empty collections, duplicate IDs, invalid
items, and missing revisions fail explicitly. Dataset adapters must surface and
resolve invalid rows; this runner does not silently filter them.

Example integration (the backend object must be supplied; no model is loaded):

```python
from post_thesis.llm_judge.runner import Example, RetryPolicy, run_judge
from post_thesis.llm_judge.judge import JudgeInput

# Inside an async function, with backend/config and pinned revisions supplied:
report = await run_judge(
    [Example("sample-001", JudgeInput(answer="Example answer", context="Example evidence"))],
    run_id="development-run-001",
    config=config,
    backend=backend,
    code_revision=code_revision,
    dataset_revision=dataset_revision,
    policy=RetryPolicy(max_attempts=2, delay_seconds=1.0),
    max_new_attempts=10,
)
```

This example's text is illustrative, not pilot data. For offline verification,
run the test command above; its backends use synthetic responses only.

The default artifact root follows `RAG_WORKSPACE` in `research_paths.py`; an
explicit `artifact_root` is also supported. Both routes append
`post_thesis/llm_judge/<run_id>/`. Store raw artifacts on a private **local disk**
with reliable OS locking and SQLite support; network/shared filesystems are not
supported by this implementation.

| Local file | Purpose |
| --- | --- |
| `journal.sqlite3` | Authoritative manifest and all attempt records, committed before/after each call |
| `manifest.json` | Exact configuration, code/data revisions, ordered IDs/request hashes, and raw requests |
| `summary.json` | Coverage, per-ID scores/status, attempt/token/latency accounting; regenerated on resume |
| `run.lock` | Persistent file holding an OS lock while a runner is active; do not delete it |

All these files remain private under the artifact root. The manifest contains raw
benchmark text and is **not** a public run summary. Do not commit or copy raw
records into thesis result directories. Export of sanitized publication artifacts
will be added with evaluation tooling.

### Resume and retry semantics

- Resume with the **same run ID and exact manifest**. Changed input text/order,
  IDs, model/prompt/settings, code/data revisions, or retry policy are rejected
  before another call. Use a new run ID for a genuinely new experiment.
- A valid saved score is reused. Identical answer/context pairs with distinct IDs
  are scored once within a run and aligned back to both IDs. New run IDs use
  independent caches, allowing intentional repeated scoring for stability tests.
- The runner validates result checksums, request identity, attempt order and
  score parsing before any new call. It stops on corruption rather than silently
  deleting a result and spending more compute.
- `max_attempts` includes the initial call and interrupted attempts, across all
  resumes. Its default is **1**. When increased, default retryable outcomes are
  `backend_error` and `interrupted`. Invalid output/incomplete generation require
  explicit opt-in. Refusals and successes are never retried within the same run.
- `max_new_attempts` caps calls in one invocation. Reaching it leaves explicit
  pending rows, which a later invocation may finish. It is **not a monetary or
  GPU-hour budget**. The runner is sequential (`concurrency=1`).
- A started call is committed before invoking the backend. Cancellation and
  programming errors propagate, preserving that record. On restart it becomes
  `interrupted`, with no score and unknown usage/latency, and consumes one attempt.
  If the server finished just before process death, a permitted retry may repeat
  that work. Exactly-once remote execution is not guaranteed.
- Concurrent runners for the same run fail promptly. OS locks release on normal
  exit or process death; a leftover lock file does not require manual deletion.

### Accounting and remaining work

The report retains every sample ID, including terminal failures and pending rows.
It distinguishes total/scored/attempted examples from unique requests and actual
attempts. Token totals include retries once per request attempt; unknown-token
counts stay explicit. Cached results retain historical model-call latency; cache
reads do not create fresh latency observations. `reused_success` identifies a
prediction reused from a previous invocation or an earlier identical input in
the current invocation. A failed attempt never becomes a score of `0.5`.

Only finished calls contribute measured latency. Interrupted calls have unknown
latency. Costs currently have `amount: null` and `status: not_configured`; this
does not mean execution is free. The serving wrapper now writes separate measured
GPU-time windows, with an optional declared rental-rate estimate. These windows must not be double-counted
or mistaken for a full rental invoice; see [SERVING.md](SERVING.md). The benchmark
preparation is implemented; the actual cluster manifest and pilot resource budget
remain pending until the preparation/preflight steps complete. Either a hosted API or
self-hosted open-weight backend can implement the existing interface. GPU
availability does not change the evaluation protocol or thesis boundary.

The test suite includes abrupt process death and runs in the existing Linux CPU
workflow plus a Windows job with separate stdlib and HTTP test steps. The
operator has exercised synthetic GPU integration; benchmark judge performance
has not been evaluated. HTTP contract tests use offline responses. Existing
thesis artifacts are unchanged.
