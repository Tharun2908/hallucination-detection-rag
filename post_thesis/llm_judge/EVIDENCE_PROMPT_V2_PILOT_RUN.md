# Post-thesis: bounded evidence prompt-v2 TRAIN pilot

**Adaptive development only; excluded from submitted thesis results.**
The [token audit](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_audit_20260920.md)
is complete. This separate [execution plan](configs/ragtruth_pilot_50_evidence_prompt_v2.json)
pins the actual audit checksum, revision and lengths. The run is complete at
`2265ac33683274383a42ec0be2e8df3a91fde86b`: see the [findings](../../results/post_thesis/llm_judge/ragtruth_evidence_prompt_v2_pilot_20260920.md).
Preserve the existing artifacts; the commands below document that completed run.
Do not rerun it on the findings commit or rename it to obtain more attempts.

## Fixed experiment and limits

Reuse the original 50 TRAIN examples and manifest, with the unchanged
`faithfulness-evidence-diagnostic-v2` prompt, schema-v3 serialized field order,
exact-quote parser and `evidence-v1` Qwen3-32B serving profile. Keep all prior runs.
The runner defaults to the historical evidence prompt v1; v2 requires the explicit
selector below and owns a separate audit and generation journal.

- Maximum 50 generation attempts, one per original input, concurrency one.
- Maximum 512 output tokens per attempt, 25,600 total. No truncation.
- 600 cumulative client seconds across resumes; 60-second request timeout.
- Audit totals: 77,724 input tokens; minimum 1,064; maximum 3,329.
- Each generation is preceded by a count check against its pinned audit entry.
- Only the fixed prompt and answer/context reach the model. Labels, task,
  generator, sample IDs and benchmark metadata remain offline.
- Interrupted/failed attempts stay terminal. Unknown execution windows charge
  their full remaining reservation. Cache replay makes no HTTP calls.
- Budget covers client preflight and runner windows. Server startup/idle and
  rental cost are recorded separately; no zero-cost claim is implied.

The existing audit remains pinned to revision
`6137fb47b5aa65ace9eb5de78a8bea52202850bc`. Do not rerun or overwrite it under the
scoring revision. The server and scoring runner must instead share the current
clean committed revision. Do not rename/delete runs or reset their budgets.

## H200 procedure after committing, pushing and passing CI

Stop the previous foreground launcher with Ctrl+C to finalize its resource
record. Start the same profile on the new commit in terminal A:

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

After `Application startup complete`, in terminal B enter the complete directory
name printed by that launcher, including the `server-` prefix:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Current server session directory name (server-...): " judge_session &&
python -m post_thesis.llm_judge.run_evidence_pilot   --prompt-version faithfulness-evidence-diagnostic-v2   --server-session-id "$judge_session"
```

Private records are written to
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-train-pilot-50-evidence-prompt-v2/`.
Return the full console summary. Invalid outputs yield a nonzero process exit
status; this does not authorize another attempt on failed inputs.
To inspect the retained cache without making requests:

```bash
python -m post_thesis.llm_judge.run_evidence_pilot   --prompt-version faithfulness-evidence-diagnostic-v2 --max-new-attempts 0
```

## Review after the run

Report structural coverage, quote-membership failures, raw field order, known
and unknown token usage, client time and manual semantic findings. Invalid
outputs remain missing, including their raw verdicts. Exact quotes do not prove
that explanations are correct. Retain all 50 cases in coverage reporting;
compare prompt versions on aligned IDs with missingness made explicit.

Carry forward the synthetic v2 unknown-as-absence rationale regression and the
false context-absence assertion missed by both prompts. Inspect hierarchical
context, faithful paraphrases, omitted information and explanation/issue agreement.
Do not tune v2 further from this run by default. This repeated pilot is adaptive
TRAIN development, not held-out evidence or a final benchmark freeze. Verdicts
are not probabilities; no threshold fitting or calibration claim is introduced.
The continuous-score design and native-source overlap audit remain unresolved.
