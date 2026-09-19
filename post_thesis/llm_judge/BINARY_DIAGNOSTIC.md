# Post-thesis binary-only synthetic diagnostic

**Development only. Excluded from the submitted thesis.** This follow-up was
selected after inspecting the completed
[probability diagnostic](../../results/post_thesis/llm_judge/synthetic_diagnostic_v1_20260919.md).
It is not held-out validation. No binary GPU results are available yet.

## Question and controlled inputs

Does asking for a binary answer identify the embedded unsupported claims that
received low response-level probabilities from v2?

Reuse all ten inputs from `diagnostic_cases.py` in the same order, with the same
case hash `92e8287dfd4559a8722c47d0047f122fb65036d319efd4d669651973dab9aff2`.
No answer, context, synthetic expectation or case selection changes. Model,
checkpoint/tokenizer revisions, BF16, non-thinking mode, temperature zero, seed,
concurrency and the 128-token output allowance remain fixed.

The intervention changes the output-elicitation wording and JSON schema together:

```json
{"verdict": "supported"}
```

or

```json
{"verdict": "unsupported"}
```

`unsupported` means at least one assertion lacks support or contradicts context.
`supported` means all factual assertions are supported, or there are none.
No probability, rationale, extra key, boolean, numeric label or third category is
accepted. Missing/refused/truncated/malformed responses stay failures with a null
verdict; they are never converted to supported or a probability.

## Prompt and output identity

The separate `faithfulness-binary-diagnostic-v1` prompt is in
`binary_contract.py`, with SHA256
`e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683`.
It retains v2's grounding, missing-value, numerical, claim-by-claim and
any-unsupported criteria. It replaces the opening probability-task wording and
final probability instructions with binary-verdict instructions. The resolved
full prompt is saved in every request record and protected by the plan hash.
It does not add examples, labels or a rationale requirement.

The request contract is `binary-diagnostic-request-v1`, with a separate enum JSON
schema and schema hash. The vLLM schema name is `faithfulness_verdict`; existing
probability requests retain their original name and unchanged schema. Requests
contain only the system prompt and answer/context input. Expected labels are
used only for offline reporting. Verdicts remain strings throughout storage and
reporting; **they are not probabilities 0 and 1**.

The probability prompts, parsers, pilot plans and existing results are unchanged.
This does not create a v3 continuous-score verifier or replace the study's
continuous-output formulation.

## Interpretation recorded before execution

Report all ten verdicts, missing outcomes and expected-verdict matches, with
supported controls alongside each unsupported case. Compare them descriptively
with the preserved probability diagnostic; do not invent a probability threshold.

- Correct unsupported verdicts on the three embedded errors, together with
  correct supported controls, would strengthen the hypothesis that the previous
  low scores depend on probability elicitation on these cases.
- Correct short verdicts but missed embedded errors would show that the binary
  formulation also has difficulty with surrounding supported content.
- Unsupported verdicts on supported controls would indicate a possible broad
  response bias, not successful error discrimination.
- Mixed outcomes remain case-specific evidence. Preserve them without rerolling,
  relabeling or modifying this prompt during the run.

The wording, schema and response space change together. A difference does not
prove that the probability judge internally detected an error and merely encoded
it incorrectly, or isolate a pure parser/numeric-format effect. This tiny,
adaptively selected diagnostic cannot establish generalization, calibration,
stability or superiority over trained verifiers. No AUROC or ECE is computed
from these binary outputs.

## Frozen run and budget

[configs/binary_diagnostic_v1.json](configs/binary_diagnostic_v1.json) binds the
case, prompt, schema and profile identities. It uses a separate fixed run ID:
`qwen3-binary-synthetic-diagnostic-v1`.

- Exactly ten synthetic inputs; one attempt per input; sequential execution.
- One generation invocation, with 300 client seconds including preflight,
  tokenization, generation and bookkeeping; 60 seconds per attempt.
- At most 4,096 formatted input tokens and 128 output tokens per example.
- Tokenization checks the input cap, then the adapter checks the count again
  before generation: at most 20 tokenization and ten completion requests.
- No truncation, automatic retries, model fallback, thinking-mode change or
  transfer of unused time from earlier experiments.

A durable SQLite journal reserves each attempt before a network call, stores the
exact binary request and raw response, and validates cached verdicts against
those responses. A separate checksum-protected execution record reserves the
single invocation before preflight. Repeating the command after that reservation
only recovers/inspects the cache, even after failures or interruption. It cannot
start another invocation. Unknown interrupted duration charges the entire
300-second reservation; deleted/corrupt artifacts or changed identities fail
closed. Preserve incomplete runs for review.

Private records live under the existing ignored post-thesis namespace. They
include `journal.sqlite3`, `manifest.json`, `execution/budget.json` and
`binary_summary.json`. The summary reports coverage, failures, known/unknown
usage, attempt latencies, charged client time and string verdicts. Cancellation
is cooperative; measured cleanup overruns are retained, and cancellation does
not prove server work stopped instantly. Stop the server to finalize its
resource record. Server startup/idle time is separate; do not sum overlapping
client/server windows. Rental cost remains unknown.

## H200 execution after committing this patch

Pull the new commit and start the existing pinned launcher from that same clean
checkout with the established CUDA/cache environment. No new packages are needed.
After `Application startup complete`, in the client terminal run:

```bash
python -m post_thesis.llm_judge.binary_diagnose \
  --server-session-id server-REPLACE_WITH_SESSION_ID
```

Use the directory printed by the new launcher. This generates up to ten binary
synthetic verdicts. It does not load RAGTruth or HaluBench, rerun earlier pilots,
fit thresholds/calibration or change v2. Paste the full console output, then stop
the server in its launcher terminal.

For zero-network inspection from the same code revision:

```bash
python -m post_thesis.llm_judge.binary_diagnose --inspect-only
```

Inspection before the first generation invocation does not consume it. Exit code
zero means all ten valid verdicts were obtained, not that they match expectations.
