# Post-thesis: bounded live evidence schema-v2 diagnostic

**Development only; excluded from the submitted thesis.** The operator-reported
[native check passed 38/38](../../results/post_thesis/llm_judge/evidence_schema_v2_native_20260919.md).
This plan checks whether schema v2 works through the actual Qwen tokenizer and
vLLM generation endpoint, and whether the three previous supported-output
contract failures recur. No v2 model results are available yet.

## Fixed comparison and serving policy

The [new execution plan](configs/evidence_diagnostic_v2.json) uses run ID
`qwen3-evidence-synthetic-diagnostic-v2` and request contract
`evidence-diagnostic-request-v2`. The schema hash is
`2c933735d69532381a4f45fbe8ad7868a34b03e78478ab8ba0b0e6486bd45807`.

Prompt `faithfulness-evidence-diagnostic-v1`, its exact messages, all 14 inputs
and their order, parser, character limits, model/tokenizer revisions and
`evidence-v1` inference profile remain fixed. Metadata and synthetic expected
labels stay outside the judge messages. New schema/request identities produce
new cache keys; v1 artifacts are never reused as v2 results or modified.

**Serving choice: retain the previous automatic backend-selection policy.**
The server command is unchanged; no backend override is added. vLLM's
[structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/)
describes automatic backend selection. The installed XGrammar check does not
prove which backend vLLM selects for either schema. Any observed difference is
therefore a schema change under the existing serving policy, not a controlled
comparison within a proven identical compiler. Preserve the new server log.
Do not silently force a backend if the live request is rejected.

The launcher now records Python, torch, vLLM and XGrammar versions. The v2 client
requires those recorded versions to match the native observation, the same
recorded default-port command, matching profile and clean scoring revision,
plus the existing HTTP model/version preflight. These local records support
reproducibility; they do not remotely attest weights or the selected compiler.

## Frozen budget

| Setting | Value |
| --- | --- |
| Synthetic inputs | 14: six expected supported, eight unsupported |
| Attempts | One per input; at most 14; one invocation |
| Client window | 300 seconds including preflight and token checks |
| Input cap | 4,096 tokens each; at most 57,344 total |
| Output allowance | 512 tokens each; at most 7,168 requested total |
| Tokenization requests | At most 28; all 14 audited before generation |
| Request timeout / concurrency | 60 seconds / 1 |
| Truncation / retries | None / none |
| Temperature / thinking | 0 / disabled |

The invocation is reserved before any HTTP call. All inputs must fit before
any generation. Each is re-tokenized immediately before its request. Alignment
errors stop execution; v2 additionally stops on HTTP 400, 422 or 500 to avoid
repeating a rejected schema or server error across the remaining inputs.
There is no fallback schema or new allowance after an error or interruption.
Completed or interrupted invocations can only be inspected from the same
scoring revision. Preserve journals and execution ledgers; do not reset them.

Report measured client time, actual token usage, missing usage and server lifetime
separately. The budget does not include startup/idle rental cost. Timeouts are
cooperative; they cannot prove remote computation stopped immediately.

## H200 procedure after commit, push and CI

Stop the earlier launcher with Ctrl+C in its own terminal, preserving its records.
Pull the new commit and restart from that clean revision. No dependency changes
are needed. Keep the cache outside the nearly full workspace.

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

Terminal B: use the exact new server directory printed in terminal A. Replace
`server-REPLACE_WITH_NEW_ID` below before running these commands.

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
judge_session=server-REPLACE_WITH_NEW_ID &&
tail -n 40 -f ".artifacts/post_thesis/llm_judge/$judge_session/server.log"
```

After `Application startup complete`, stop only the log tail with Ctrl+C, then:

```bash
python -m post_thesis.llm_judge.evidence_diagnose \
  --schema-version v2 --server-session-id "$judge_session"
```

The private result directory is
`.artifacts/post_thesis/llm_judge/qwen3-evidence-synthetic-diagnostic-v2/`.
Retain `evidence_summary.json`, `manifest.json`, `journal.sqlite3` and
`execution/budget.json`. Stop terminal A's launcher after the run to finalize
its resource record. Cache-only inspection from the same clean revision:

```bash
python -m post_thesis.llm_judge.evidence_diagnose --schema-version v2 --inspect-only
```

## Review before any TRAIN work

Report all 14 statuses, contract-valid coverage, valid verdict/issue matches,
known tokens, missing usage and measured time. Invalid/refused/truncated outputs
remain missing; a matching raw verdict never salvages an invalid record.
Review the three supported short cases and manually review every accepted
unsupported explanation, including the known unknown-versus-absence confusion.
Quote membership and successful structured decoding do not prove semantic
correctness. A clean run also does not establish generalization or calibration.
No TRAIN/test scoring, new labels, threshold fitting or final prompt freeze is
part of this run. Decide the next experiment only after inspecting its output.
