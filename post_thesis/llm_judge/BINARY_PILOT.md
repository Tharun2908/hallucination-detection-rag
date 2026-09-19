# Post-thesis: binary follow-up on the existing TRAIN pilot

**Development only. Excluded from the submitted thesis.** The binary synthetic
diagnostic matched all ten constructed expectations; see the
[recorded findings](../../results/post_thesis/llm_judge/binary_diagnostic_v1_20260919.md).
The next question is whether the unchanged binary formulation identifies errors
in realistic answers without excessively flagging supported answers.

**Current step: token audit only.** No binary TRAIN verdicts have been generated,
and no binary TRAIN scoring command is enabled by this patch. The cluster audit
checksum and counts are required before committing its execution plan.

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

## Separate binary token audit

Apply, commit and push the patch; pull it on H200 and restart the pinned launcher
from the same clean checkout using the established CUDA/cache environment. No
new packages are needed. After `Application startup complete`, in the client
terminal run:

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

Check the console for the binary prompt/version, 50 counted inputs, zero
failures/pending/overlength cases and zero generation calls. Paste the full output,
including **audit SHA256, audit code revision, minimum/maximum/total input tokens**.
Stop the server afterward to finalize its resource window. Tokenization timing
is not inference latency or a price estimate.

## Intended scoring plan — not enabled at this step

After reviewing and pinning the audit, prepare a separate plan with the fixed
run ID `qwen3-ragtruth-train-pilot-50-binary-v1`. Intended limits are 50 examples,
one attempt each across resumes, sequential requests, 60 seconds per request,
and 600 cumulative client seconds. Output allowance remains 128 per example
(6,400 total). Input token totals must come from the real audit; they are not
inferred from earlier prompts. No unused allowance transfers from previous runs.

The current `run_pilot.py` selects probability v1/v2 only, while
`binary_diagnose.py` runs the ten synthetic cases only. Neither is a binary TRAIN
scorer. Do not use either to attempt this pilot, edit old plans, or rerun old
probability/tokenization jobs from the newer commit.

## Planned analysis, fixed before collecting binary TRAIN verdicts

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
