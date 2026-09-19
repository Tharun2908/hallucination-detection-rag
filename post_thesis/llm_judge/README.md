# Post-thesis experiment: LLM judge for RAG faithfulness

> The experiments below were conducted after thesis submission and are not part
> of the submitted thesis results.

**Status: offline interface, development prompt, strict parser, and contract tests
implemented. No provider adapter, pilot, paid API call, or new evaluation result
is included.** The notice above identifies the scope of this section; it does not
claim completed experiments.

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
| Local request cache and raw responses | `.artifacts/post_thesis/llm_judge/` by default; future runner must preserve `post_thesis/llm_judge/` under any configured artifact root |
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
   — interface/parser/tests complete; runner next.
3. Select one model and a bounded API budget; pilot on 50–100 source TRAIN examples.
4. Freeze the prompt and use a separate source development subset for thresholding
   and any declared calibration.
5. Evaluate the frozen system on RAGTruth test and the canonical HaluBench test set.
6. Analyze disagreements and evidence sensitivity, including negative results.
7. Consider a cascade only if the standalone findings justify it.
8. Publish a separately labeled post-thesis report and release.

Model comparison and paid execution follow the protocol stages; there is no
API-backed judge command yet.

## Offline foundation

Requires Python 3.10+ and only the standard library. From the repository root:

```bash
python -S -m unittest discover -s tests -p test_llm_judge.py -v
```

The existing CPU reproduction workflow also discovers these tests. All backends
in the tests are local fakes; no credentials, downloads, or API calls are needed.
Passing these tests validates engineering contracts, not judge quality.

| Module | Responsibility |
| --- | --- |
| `prompts.py` | Versioned `faithfulness-development-v1` prompt and content hashing |
| `parse.py` | Strict single-object JSON parsing, without extracting or repairing text |
| `judge.py` | Immutable requests, injected asynchronous backend, one attempt and per-response metadata |
| `../../tests/test_llm_judge.py` | Parser failures, input boundary, request identity, failure accounting, concurrent metadata isolation |

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
The latter is a non-secret versioned identity for the future adapter, endpoint,
API revision, and any provider-specific settings. Change it whenever those
settings or adapter behavior change. Never put an API key in it. Temperature
defaults to zero, but reproducibility still depends on the provider. No real
provider or model is selected here.

`build_request` includes a JSON response schema and hashes the exact messages,
prompt version/hash, configuration, schema, protocol ID, and request-contract
version. This is a request-identity primitive; **caching and persistence are not
implemented yet**. Repeating `judge_once` invokes the supplied backend again.
Mutable model aliases require a new run identity/cache policy when resolved
versions change; a request hash alone cannot detect a provider changing an alias.

### Backend contract and results

Implement `JudgeBackend.complete(request)` only when the provider is selected.
An adapter must honor the request's messages, configuration and response schema;
normalize provider refusals/truncation into `BackendResponse.outcome`; and return
the reported model, response ID and token usage where available. Unknown model
and usage stay `None`. Reject unsupported settings explicitly. Do not silently
truncate evidence, add prompts, retry, or switch models. The adapter and runner
need their own integration tests before any paid pilot.

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

The next engineering step is the resumable runner: incremental attempt records,
cache verification, bounded retries, run manifests, budget enforcement, and
price-versioned cost accounting. Those features, provider integration, and pilot
selection are still pending. Existing thesis artifacts are unchanged.
