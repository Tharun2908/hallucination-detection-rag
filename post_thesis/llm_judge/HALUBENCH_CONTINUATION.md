# Post-thesis HaluBench budget continuation

The original four-hour run stopped with 7,652 valid scores, one interrupted
request and 347 pending requests. Its offline replay reproduced report
`2be55b823bb5b22e5bd5814ed578837b43e961ca2fe1a96bc0cf3f1bd0e95a3b`.
This is an execution-budget amendment, excluded from submitted thesis results.
It is based on completion status, before HaluBench performance evaluation.

## Recorded amendment

Preserve the entire original run, including its exhausted budget, raw responses,
interrupted attempt, unknown token usage and original summary. The new plan
`configs/halubench_label_continuation_v1.json` permits at most 348 additional
requests and 7,200 additional client seconds. It reissues the interrupted request
once and attempts the 347 unattempted requests once each, in canonical order.
The selected payloads contain 1,589,707 input tokens and allow 348 output tokens.
The per-input length bounds inherited from the original plan remain conservative
bounds for this subset. The new runner has no automatic retries.

The interrupted request may have finished on the old server. Its input/output
usage therefore remains unknown; the subsequent successful request cannot erase
that uncertainty. A completed continuation would yield 8,000 unique scores from
8,001 total client attempts, including the original interruption. Known token
usage, unknown-usage counts and charged time remain separately recorded.

No prompt, tokenizer, model, serving profile, label mapping, score definition,
calibration, operating threshold, canonical split or baseline changes. No examples
are selected by labels, predicted scores or observed performance. No incomplete
7,652-example benchmark evaluation is substituted for the registered population.

## First: read-only CPU check

Use the CPU pod and its system Python. No serving virtual environment or package
installation is needed for this check:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
python3 -S -m post_thesis.llm_judge.continue_halubench_scores --prepare-only
```

The command validates the canonical inputs and pinned baseline reports; checks
the exact original report, plan, revision and full prepared requests; opens the
original SQLite journal in read-only mode; and recomputes scores and usage from
its raw responses. It must reproduce the pinned parent hash. It then verifies the
exact status pattern and selects only the 348 unchanged saved request payloads.
It neither recovers nor rewrites the parent journal, locks, budget or summary.
It prints the selected-request hash, the repeated sample ID and amendment hash.
If any check fails, retain the files and investigate; do not reset the old run.

## Then: finish on the H200

A CPU pod cannot finish Qwen inference using the frozen H200 profile. After the
CPU check passes, restore the pinned serving environment, CUDA 13 build tools
and model cache as needed on the GPU pod. Pull this commit there too. Launch a
new server from that checkout using the same profile:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0 --profile label-score-v1
```

After server readiness, use another terminal with the serving environment active:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "New serving directory (server-...): " judge_session &&
python -m post_thesis.llm_judge.continue_halubench_scores \
  --server-session-id "$judge_session"
```

The new server revision and pinned runtime are checked before transport. The
existing global judge lock prevents concurrent registered judge runners. Child
results checkpoint every 25 requests in a new directory:
`.artifacts/post_thesis/llm_judge/qwen3-halubench-test-label-score-continuation-v1/`.
Only its 348 requests and compact references to the unchanged parent preparation
are stored in the child budget identity. Parent artifact validation is repeated
before execution. The two-hour allowance is a cap, not a completion-time promise.

Do not reuse the old overnight launcher: it invokes the exhausted parent runner.
Do not edit the original budget or clear its halt reason. An interrupted child
also halts without automatic retry; preserve its records if that happens.

## Replay and next step

After a successful invocation, offline verification works on either pod:

```bash
python3 -S -m post_thesis.llm_judge.continue_halubench_scores --inspect
```

Expect 348/348 valid child scores, 8,000/8,000 combined valid scores, zero new
attempts, the same child report hash and one cumulative unknown-usage attempt.
The original parent report retains one interruption and 347 pending rows.
The child summary carries original canonical indices and separate cumulative
accounting; it does not overwrite or masquerade as the original 8k summary.

The subsequent CPU evaluation must verify both reports and join the 7,652 valid
parent rows with the 348 valid child rows exactly once per canonical ID. This
patch does not compute TEST metrics, merge predictions or refit anything.
