# Post-thesis: controlled synthetic diagnostic for prompt v2

**Development only. Excluded from the submitted thesis.** Motivated by the
[paired TRAIN pilot](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_v2_20260919.md).
This run is complete; see the [recorded findings](../../results/post_thesis/llm_judge/synthetic_diagnostic_v1_20260919.md).
The [binary-output follow-up](BINARY_DIAGNOSTIC.md) also completed. The subsequent
[binary TRAIN pilot](BINARY_PILOT.md) and focused error review are also complete.
Preserve this historical run. Its v2 prompt, pinned model,
non-thinking mode, temperature, JSON schema and 128-token output limit were unchanged.
This experiment does not introduce a v3 prompt or read any benchmark dataset.

## Question and controls

Does v2 distinguish unknown values from support or contradiction, and does an
unsupported claim receive a lower score when surrounded by supported statements?

Ten manually authored cases are versioned in `diagnostic_cases.py`:

| Family | Evidence / target assertion | Short answer | Embedded answer |
| --- | --- | --- | --- |
| Attribute: supported | OutdoorSeating=True / offers seating | 1 | 1 |
| Attribute: contradicted | OutdoorSeating=False / offers seating | 1 | 1 |
| Attribute: unknown | OutdoorSeating=None / offers seating | 1 | 1 |
| Numeric: supported | lowest 60% / lowest 60% | 1 | 1 |
| Numeric: contradicted | lowest 60% / highest 60% | 1 | 1 |

Each short/embedded pair uses **identical context**. The embedded answer adds
12 explicitly supported sentences, with the target after sentence six. The short
answer contains just the target assertion. The extra facts are present in both
contexts. Attribute evidence variants keep the answer identical; numeric variants
keep the context identical. This studies added answer content, not increased
context length. Length, surrounding content and target position change together;
these cases cannot separate those mechanisms or prove an attention failure.

The setting and quantities are synthetic, without benchmark IDs, copied answers,
labels, generator identities or task metadata. Expected outcomes are stored for
offline comparisons only. Requests contain the unchanged v2 prompt and only
`answer` and `context`. No rationale or chain-of-thought output is requested.

## Interpretation fixed before execution

- In both answer lengths, contradicted and unknown attribute claims should score
  above the supported attribute claim; the contradicted numeric claim should
  score above its supported control. Ties do not count as the expected direction.
- Report every raw score and unsupported-minus-supported difference; no operating
  threshold or synthetic benchmark accuracy is selected.
- Report embedded-minus-short score differences for all five matched pairs.
  For an unsupported target, a negative difference is consistent with sensitivity
  to supported surrounding text on this case. Check supported controls too; a
  general shift is not evidence specific to error dilution.
- If unknown cases fail even in isolation while explicit contradictions separate,
  that is consistent with a missing-value support-judgment problem on these cases.
- Missing/failed scores remain missing, with comparisons marked null. Do not
  replace failures, reroll examples, or tune the prompt while this run is underway.

There is only one small constructed setting and one execution per input. Results
are qualitative diagnostics, not estimates of population performance, stability,
calibration or causal explanations of the model's internal computation.

## Frozen resource limits and preservation

[configs/synthetic_diagnostic_v1.json](configs/synthetic_diagnostic_v1.json) pins
the case content hash, prompt hash and model profile. It fixes:

- Run ID: `qwen3-v2-synthetic-diagnostic-v1`; no CLI override.
- Ten inputs, one attempt each, sequential execution, no automatic retries.
- At most 4,096 formatted input tokens and 128 output tokens per example.
  The maximum requested generation-token budget is 40,960 input + 1,280 output.
- One generation invocation, at most 300 client seconds including preflight,
  tokenization, generation and bookkeeping. Each attempt has a 60-second limit.

Before each completion, a tokenization call enforces the input cap. The adapter
then checks that count again before generation. Thus at most 20 tokenization
requests and ten completion requests are made; tokenization calls are not
completion usage. Oversized inputs and token/model/output-limit discrepancies
halt the invocation. Other bounded failures retain their terminal records.
Actual known usage is retained even if the server reports exceeding a limit.

This is intentionally a single diagnostic invocation. **Repeating the command
only inspects/reconstructs cached results after the first invocation**, including
if it failed or left rows pending. It never grants another budget or retries.
A crash with unknown duration conservatively charges the entire 300-second
reservation. Missing/corrupt records or changed identities fail closed. No
unused pilot allowance is transferred. Preserve any failed run for review.

Private records use the existing ignored post-thesis namespace. They include the
runner journal and manifest, `execution/budget.json`, and
`diagnostic_summary.json`. The latter has scores, comparisons, new-attempt count,
coverage, usage and charged client seconds. v1/v2 pilot artifacts stay intact.
Client time excludes server startup/idle time and can slightly exceed the
cooperative deadline during cleanup. Cancellation does not attest that server
work stopped instantly. Keep server resource records; do not sum overlapping
client and server windows. Rental cost remains unknown.

## Historical H200 execution — do not rerun from a newer commit

After applying, committing and pushing the patch, pull it on the cluster and
start the pinned launcher from that same clean commit using the established
CUDA and cache environment. No new package installation is needed. Wait for
`Application startup complete`, then run in the client terminal:

```bash
python -m post_thesis.llm_judge.diagnose \
  --server-session-id server-REPLACE_WITH_SESSION_ID
```

Use the session directory printed by the new launcher. This command **generates
up to ten synthetic scores**. It does not load RAGTruth or HaluBench. Paste the
printed scores, comparisons and execution summary, then stop the server to
finalize its resource record. Do not rerun either pilot or token audit.

For zero-call inspection from the same code revision, including before execution:

```bash
python -m post_thesis.llm_judge.diagnose --inspect-only
```

Inspection before the first invocation does not consume it. Inspection after
execution does not reset it. The exit code is zero only when all ten scores are
present; it does not indicate that the expected score directions were observed.
