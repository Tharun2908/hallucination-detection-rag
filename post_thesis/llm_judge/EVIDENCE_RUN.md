# Post-thesis: bounded synthetic evidence run

**Synthetic development only. Excluded from the submitted thesis.** The
[frozen evidence contract](EVIDENCE_DIAGNOSTIC.md) now has a durable runner and
[committed execution plan](configs/evidence_diagnostic_v1.json). No GPU results
for this contract have been collected yet. No benchmark examples are loaded.

## Fixed scope and budget

| Setting | Frozen value |
| --- | --- |
| Run ID | `qwen3-evidence-synthetic-diagnostic-v1` |
| Inputs | 14 fixed synthetic examples; six expected supported, eight unsupported |
| Prompt | `faithfulness-evidence-diagnostic-v1` (unchanged) |
| Model | Same pinned Qwen3-32B weights/tokenizer, BF16, one H200 |
| Profile selection | `evidence-v1` |
| Server alias | `qwen3-32b-evidence-9216db5781bf` |
| Temperature / thinking / concurrency | 0 / disabled / 1 |
| Maximum generation attempts | 14 total, one per input, one invocation |
| Input allowance | 4,096 tokens per input, at most 57,344 across 14 |
| Output allowance | 512 tokens per input, at most 7,168 requested across 14 |
| Maximum input + requested output | 64,512 tokens (bound, not measured usage) |
| Tokenization requests | At most 28: 14 initial checks plus 14 pre-generation rechecks |
| Request timeout | 60 seconds |
| Client window | 300 seconds including preflight, token checks and runner work |
| Replays | Cache inspection only; no remaining invocation allowance |

The separate [profile](configs/qwen3_32b_h200_evidence_v1.json) changes only profile
version, server alias and output allowance relative to the original profile.
Model/tokenizer revisions and decoding settings remain fixed. Defaults still
select the original 128-token profile; its configuration and completed runs are
unchanged. The distinct alias and recorded profile help reject an old server.
HTTP metadata alone is not cryptographic attestation of loaded weights.

512 tokens is a maximum, not a guarantee every valid evidence record will fit.
A truncated response remains incomplete even if it contains parseable JSON.
Do not increase allowances or change the prompt within this run to fix failures.
Character limits in the evidence schema remain unchanged.

## Execution order and persistence

1. Validate the plan, prompt/schema/case hashes, inference profile and clean code
   revision. New model calls also require a matching live launcher record.
2. Create/validate the separate attempt journal and execution ledger. Reserve the
   single 300-second invocation before any HTTP request, including preflight.
3. Preflight server metadata. Tokenize all 14 exact requests, saving the start and
   outcome of each count incrementally in the checksummed execution ledger.
   If any count fails or exceeds the cap, generate **nothing**. Do not truncate.
4. Only after all counts pass, generate sequentially. Re-tokenize each request and
   require the count to match its saved audit; check returned model/token usage.
   Alignment or token-limit violations halt further generation.
5. Preserve received response text, known usage, attempt latency, valid evidence
   objects or sanitized errors in the private journal. Reconstruct summaries from
   that journal. Cache replay reparses stored responses against the original
   inputs, rejecting inconsistent or corrupted evidence records.

`evidence_runner.py` handles durable one-attempt execution; `evidence_diagnose.py`
adds the frozen synthetic plan, full-set token checks and invocation budget.
The generic runner is not an alternative CLI for bypassing that plan.

A normal completed invocation, preflight/audit failure, deadline or interruption
cannot open another generation window. A process lost before final bookkeeping
leaves an unknown window charged the full 300 seconds; only cache inspection is
allowed. Started generation attempts are recovered as interrupted with unknown
usage, not retried. Missing ledgers/journals or changed identities fail closed.
Do not delete, rename or edit artifacts to reset a run.

Timeouts are cooperative and do not prove remote computation stopped immediately.
Measured client time excludes server startup/idle time; the separate launcher
resource record and actual rental duration are needed for cost analysis. Token
usage counts generation input/output, not tokenization as extra billed generation.
No zero-cost or per-example latency claim follows from the configured budget.

## H200 procedure after committing and CI

Apply, commit and push the patch, verify Linux/Windows CI, then pull on H200.
Stop the previous launcher in its terminal before starting this new profile.
No dependency reinstall is needed. Preserve all earlier run and audit directories.

Terminal A:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0 --profile evidence-v1
```

Terminal B, after the launcher prints its new server directory:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
judge_session=$(basename "$(ls -td .artifacts/post_thesis/llm_judge/server-*/ | head -n 1)") &&
tail -n 40 -f ".artifacts/post_thesis/llm_judge/$judge_session/server.log"
```

Once `Application startup complete` appears, press Ctrl+C in terminal B only,
then execute the fixed synthetic run:

```bash
python -m post_thesis.llm_judge.evidence_diagnose \
  --server-session-id "$judge_session"
```

Confirm that the selected session is the evidence-profile launcher from terminal
A. The client checks its recorded profile/revision and advertised model alias.
After the invocation, preserve the console output and stop the server in terminal
A with Ctrl+C to finalize its resource record.

The private directory is
`.artifacts/post_thesis/llm_judge/qwen3-evidence-synthetic-diagnostic-v1/`.
It contains `evidence_summary.json`, a manifest, SQLite attempt journal, and
`execution/budget.json` with the incremental input audit. There is no separate
TRAIN audit or TRAIN scoring command for the evidence contract yet.

From the same clean scoring revision, inspect without server calls:

```bash
python -m post_thesis.llm_judge.evidence_diagnose --inspect-only
```

## Interpret the output

The console distinguishes valid evidence records, terminal failures, pending
inputs, verdict matches and issue-type matches. Invalid records have missing
verdicts and missing match indicators, never an automatic supported verdict.
All comparisons are against manually authored synthetic expectations.

**Valid record count is not accuracy.** Real quotations can be irrelevant or
misleading, and an explanation can be wrong even when both verdict and issue type
match expectations. Each result therefore retains `semantic_evidence_review`
as pending. Review the saved evidence objects for claim scope, quote relevance
and explanation faithfulness after execution; do not treat those flags as an
independent annotation or automatically mark them passed.

No continuous scores, calibration metrics, threshold fitting, prompt changes,
TRAIN pilot generation or final benchmark claims are introduced here. Decide
whether to prepare the separate original-50 TRAIN audit only after reviewing
both execution validity and synthetic evidence quality.
