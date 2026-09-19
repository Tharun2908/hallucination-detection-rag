# Post-thesis: bounded evidence-v3 TRAIN pilot

**Adaptive TRAIN development only; excluded from the submitted thesis.**
The run below completed at revision `dbf1bfc0e03dbf0b88d8b0914bbdbb0d224f1808` with 37/50 valid records. See the [findings](../../results/post_thesis/llm_judge/ragtruth_evidence_pilot_v3_20260919.md). Preserve its artifacts; these historical instructions do not create a fresh allowance at newer revisions.

The run used the unchanged evidence prompt v1 and verdict-first schema v3 on the original
50 examples. The synthetic result was 13/14 expected verdicts, with an embedded
unknown-absence claim missed. Keep that failure visible; this is not a final
benchmark freeze or independent validation.

## Pinned inputs and budget

The reviewed configuration is [`configs/ragtruth_pilot_50_evidence_v3.json`](configs/ragtruth_pilot_50_evidence_v3.json).
It pins the original manifest, preparation revision and
[completed token audit](../../results/post_thesis/llm_judge/ragtruth_evidence_v3_audit_20260919.md).
Preserve the existing files; do not rerun preparation or the audit after updating
code. Their older recorded revisions are intentional.

| Limit | Value |
| --- | --- |
| Inputs | Original 50 TRAIN examples, unchanged order/text |
| Generation attempts | At most 50 total, one per input |
| Concurrency | 1 |
| HTTP operation timeout | 60 seconds |
| Cumulative client budget | 600 seconds across resumes |
| Formatted input tokens | 77,574 total; maximum 3,326 |
| Output allowance | 512 per example; 25,600 total |
| Input plus requested output bound | 103,174 tokens |
| Truncation / replacement | None |

Use the unchanged `evidence-v1` serving profile: pinned Qwen3-32B revision,
BF16, non-thinking, temperature 0, seed 0 and vLLM default auto structured-output
backend selection. The server must record the same Python, torch, vLLM and
XGrammar versions as the native v3 check and run the current clean code revision.
The runner checks the original audit digest, exact serialized schema and request
keys, then retokenizes each input to check alignment before generation.

Only the fixed system prompt plus answer/context reach the model. Labels, task,
query, generator identity and sample IDs remain offline. Outputs are categorical
evidence records, not probabilities. No metrics, threshold fitting, prompt
changes, TEST or HaluBench reads are performed by this command.

## Historical H200 execution at revision dbf1bfc

Stop the existing launcher with Ctrl+C to finalize its resource record. In
terminal A, pull and restart the same profile; dependencies stay unchanged:

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

Wait for `Application startup complete` in that session's log. In terminal B,
enter the exact `server-...` directory name printed by this launch:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Current server session directory name (server-...): " judge_session &&
python -m post_thesis.llm_judge.run_evidence_pilot --server-session-id "$judge_session"
```

The fixed output directory is
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-train-pilot-50-evidence-v3/`.
Keep its journal, manifest, execution budget and `evidence_pilot_summary.json`.
To inspect cached results without any HTTP request or generation:

```bash
python -m post_thesis.llm_judge.run_evidence_pilot --max-new-attempts 0
```

A normal resume uses the same run, code and configuration, skips all attempted
inputs and spends only the remaining budget. Failed/interrupted attempts are
terminal. An unfinished client window charges its full reserved remaining
budget. Alignment or selected serving errors halt the run and block further
calls. Missing/corrupt state requires inspection; do not delete, rename or reset
artifacts to recover attempts. Startup and idle server time are excluded from the
client budget, retained separately in launcher records; rental cost is unknown.

## Review the output before further experiments

The console summary and per-example evidence records were returned. Targeted review is recorded in the findings; review of every claim remains incomplete.
Schema validity, exact quote membership and raw response field order are reported
separately from semantic correctness. Invalid/refused/truncated outputs remain
missing; raw verdicts are not salvaged. Valid outputs in an unexpected order
remain valid but are flagged separately. CLI success checks valid coverage and
order, not accuracy. Manual evidence review remains required even if all 50
records are structurally valid.

Preserve historical probability and binary predictions for a later paired
**development** comparison. Native-source overlap auditing, continuous-score
and calibration design, threshold selection and final benchmark evaluation
remain unresolved separate steps.
