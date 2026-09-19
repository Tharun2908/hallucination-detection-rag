# Post-thesis: binary follow-up on the existing TRAIN pilot

**Development only. Excluded from the submitted thesis.** The binary synthetic
diagnostic matched all ten constructed expectations; see the
[recorded findings](../../results/post_thesis/llm_judge/binary_diagnostic_v1_20260919.md).
This follow-up asked whether the unchanged binary formulation identifies errors
in realistic answers without excessively flagging supported answers.

**Current status: binary TRAIN execution and focused error review complete.**
All 50 verdicts were valid; the judge detected 13/24 labeled positives with three
false positives. All four previously reviewed misses persist. See the
[recorded result](../../results/post_thesis/llm_judge/ragtruth_binary_pilot_20260919.md)
and [focused review](../../results/post_thesis/llm_judge/ragtruth_binary_error_review_20260919.md).

The procedures below document the completed run at scoring commit
`4b87517a7687df5d8684f9a28db7dbffe4278304`. Preserve the preparation, audits and
run records. **Do not rerun them from this newer reporting commit.** The
execution identity is tied to the original scoring revision. The reporting update
introduced no changes to the binary prompt, generation calls or execution budget.
The next [evidence diagnostic](EVIDENCE_DIAGNOSTIC.md) is a separate offline
contract/parser step; the completed binary contract is unchanged.

## Fixed inputs and contract

- Reuse the exact 50 TRAIN examples and ordering in the original preparation
  manifest; SHA256
  `ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
- Preserve its preparation revision, `initial_prompt` v1, group reservations,
  labels, IDs and raw answer/context text. Do not rerun preparation or select
  another sample. Historical `initial_prompt` records provenance; it does not
  describe the new request prompt.
- Use unchanged `faithfulness-binary-diagnostic-v1`, SHA256
  `e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683`.
- Build the same enum schema and `binary-diagnostic-request-v1` request contract
  used in the synthetic binary run. The schema hash and contract version are
  explicitly recorded in this audit identity. Per-input request keys bind the
  exact schema, prompt, messages and configuration.
- Keep the pinned Qwen3-32B H200 profile, non-thinking mode, BF16, seed,
  temperature zero, concurrency one and 128-token output allowance unchanged.
- The server receives the system prompt plus answer/context only. Labels,
  generator identities, tasks, separate questions and original sample IDs are
  excluded from messages; their offline records stay private.

This remains the same adaptively inspected development sample. It cannot supply
held-out evidence for choosing the final prompt, and source-group proxy
limitations from [PILOT.md](PILOT.md) still apply. No source-disjoint threshold
subset has been established, and neither test benchmark is loaded.

## Completed binary token audit — historical procedure

The operator completed the following command on commit
`32222be78c64274dd50fe8a0e6e63b51b6d94457`. Preserve this audit unchanged;
**do not rerun it or the preparation step from the scoring commit**:

```bash
python -m post_thesis.llm_judge.audit_pilot \
  --prompt-version faithfulness-binary-diagnostic-v1 \
  --expected-manifest-sha256 ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25
```

The default directory is
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-token-audit-binary-v1/`.
It is distinct from v1/v2 probability audits and both synthetic diagnostics.
Passing another prompt's reserved default audit name is rejected; a changed
existing audit identity fails closed before HTTP or overwriting its results.

The audit counts formatted input tokens via `/tokenize`; it never invokes a
completion endpoint. It permits one tokenization attempt per example, a maximum
of 50 requests, up to 60 seconds per operation and a 300-second invocation
window including preflight. Counts are saved incrementally; a completed cache
replay makes zero network requests. A failed/interrupted attempt is terminal
for that audit ID and blocks further counting. Do not delete or rename records
to retry. The deadline is cooperative; preserve resource records and unknown
states rather than treating interruptions as free compute.

### Operator-reported audit evidence

These values come from the supplied cluster console output. The private audit
was not downloaded for independent inspection; the scorer validates its full
checksum, identities and per-request counts locally before any generation.

| Field | Value |
|---|---|
| Audit SHA256 | `55c8b27a10cf0f26c6dfcb35d6882d2c9b400a8fa90bb20689ab7746348f181b` |
| Audit code revision | `32222be78c64274dd50fe8a0e6e63b51b6d94457` |
| Counted / total | 50 / 50 |
| Failed, pending, overlength | 0, 0, 0 |
| Minimum / maximum input tokens | 719 / 2,984 |
| Total input tokens | 60,474 |
| Output allowance per example | 128 |
| Model context limit | 32,768 |
| All inputs fit | true |
| Generation calls | 0 |

Tokenization timing is not inference latency or a price estimate.

## Completed binary scoring — historical procedure

The separate [execution plan](configs/ragtruth_pilot_50_binary_v1.json) fixes run ID
`qwen3-ragtruth-train-pilot-50-binary-v1`. It allows exactly the existing 50 inputs,
at most one generation attempt per input across resumes, sequential requests,
60 seconds per request, and **600 cumulative client seconds**. Output allowance
is 128 per example (6,400 total); audited input plus maximum requested output is
**66,874 tokens**. No unused allowance transfers from previous runs.

Apply, commit and push the scoring patch; pull it on H200 and restart the pinned
launcher from that same clean checkout, using the established CUDA/cache
environment. No new packages are needed. Keep `HF_HOME` on the larger root
filesystem. Once the server reports `Application startup complete`, run in the
client terminal, replacing the session name with the launcher's printed ID:

```bash
python -m post_thesis.llm_judge.run_binary_pilot \
  --server-session-id server-REPLACE_WITH_CURRENT_SESSION
```

This command uses the original preparation manifest and the completed binary
audit. It checks the frozen prompt, schema, request contract, model profile,
server revision and audited request identities, then re-tokenizes each exact
request before generation and requires its count to match. Input or returned
model/token-limit mismatches halt further generation and block new calls on
resume. There are no prompt, model, run-ID or budget overrides.

Results remain string verdicts, `supported` or `unsupported`; failures have no
verdict. No labels or benchmark metadata enter requests. The probability-only
`run_pilot.py` and synthetic-only `binary_diagnose.py` remain unchanged.

For a cache-only inspection from the same scoring revision, with no server calls:

```bash
python -m post_thesis.llm_judge.run_binary_pilot --max-new-attempts 0
```

A smaller `--max-new-attempts` can split execution into invocations without
increasing the lifetime attempt or time allowance. Completed and failed attempts
are not retried. Missing/corrupt journals, ledgers or changed identities block
execution; do not delete records to reset a run. A window left unfinished after
an unrecorded interruption is charged its full reserved remaining budget.
Timeouts are cooperative and do not prove that server work stopped immediately.

Private records live under
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-train-pilot-50-binary-v1/`,
including `binary_pilot_summary.json`, the attempt journal and execution ledger.
The summary reports coverage, failures, pending inputs, attempts in the current
invocation, cumulative known token usage and charged/remaining client seconds.
Preserve all records and stop the server afterward to finalize its resource
window. Client budget windows include preflight and runner work; startup and
idle time are excluded. Server records and actual rental duration are needed
for a cost analysis; this plan does not imply zero GPU cost.

## Analysis plan, fixed before collecting binary TRAIN verdicts

Use `unsupported` as the positive class and preserve original labels. Report all
50 IDs with verdict/status, then confusion counts, precision, recall and F1 on
valid verdicts, with overall and class-specific failure coverage explicit. An
undefined metric stays undefined; no failed response becomes a supported verdict.
If coverage is incomplete, do not claim a full-sample quality estimate from the
successful subset alone.

Compare case-level decisions with the preserved probability pilots, including
9350, 6040, 6486 and 12052, and inspect false positives as well as recovered errors.
Do not search a probability threshold on these examples to optimize comparisons.
This binary formulation has no tunable score threshold. Do not recode its
verdicts as probabilities and report calibration or continuous-score performance.
The final study's continuous-score design remains unresolved; retain that
limitation before any final prompt freeze or benchmark expansion.
