# Post-thesis frozen Qwen HaluBench run

**Run update, 2026-09-29:** the original budget is exhausted with 7,652 valid scores. Preserve that run and follow the [separate continuation](HALUBENCH_CONTINUATION.md); the original launch instructions below document its registered execution.

MiniCheck inference and replay completed on all 8,000 canonical TEST examples,
with no failures, no truncation and identical report hash
`18a157b4caf96fba23b35b8adb9f961dc5df45fd847a3e026305df7289e181b0`.
S4, S2 and the frozen metadata-free fusion are also complete. This step verifies
those artifacts together and runs the already frozen Qwen judge. It is excluded
from submitted thesis results. No metric evaluation, refitting, new threshold,
new split, prompt revision or HaluBench adaptation occurs.

## Fixed scope

The canonical input manifest remains
`0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded`.
The completed CPU token audit remains
`7173c30e3a197aabb61c8ffd8c0e145cc556affa9aad20d7dcd43bc9cb24782a`:
8,000 inputs, 7,794,485 input tokens, 506–7,298 per example, all fitting unchanged.

Use the same Qwen3-32B revision, BF16 H200 runtime, non-thinking prompt,
primary mapping A=supported/B=unsupported, one-token unmasked raw-logprob
request and `label-score-v1` serving profile as the RAGTruth experiment.
The saved scores remain raw class-token margins and normalized two-label scores.
No probability calibration is applied during inference. The frozen RAGTruth
calibration and raw-margin operating rule remain available for later evaluation.

`configs/halubench_test_label_scoring_v1.json` fixes 8,000 requests, one attempt
each, concurrency one, no retries, a 60-second request timeout, and 14,400 client
seconds across invocation windows. Each request permits one output token.
There is no prompt truncation. Requests contain answer/context-derived token IDs
only; offline labels, source names, generators and benchmark metadata do not
enter the model payload.

## Baseline verification in the same command

`halubench_comparison.py` checks the pinned full report hashes, all 8,000 IDs,
indices and input hashes for S4, S2, fusion and MiniCheck. It checks the historical
TRAIN operating rules and frozen fusion lineage, then saves a deterministic
comparison descriptor at
`.artifacts/post_thesis/llm_judge/halubench-fresh-comparison-manifest-v1/manifest.json`.
The descriptor binds the full source artifacts and input manifest; it is not a
second copy of the per-example predictions. Its readiness is scoped to these
fresh runtime baselines. Judge predictions and paired evaluation remain pending.

Every scoring or inspection invocation repeats these integrity checks. They read
the existing baselines but do not compute label metrics or select among runtime
variants. The canonical 7,838 TEST groups merge to 7,198 exact-passage bootstrap
components. Exact passages link 994 TEST rows to adaptation components; this
run performs no adaptation training and does not claim strict document disjointness.
Historical TRAIN/checkpoint provenance and evidence-window differences remain
explicit. MiniCheck's documented fresh/legacy drift on RAGTruth is not assumed
to disappear on HaluBench.

## Prepare and launch

In the pod, update the repository and use the existing serving environment:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.run_halubench_scores --prepare-only
```

This prepares all requests and checks the installed vLLM source, tokenizer and
baseline lineage without HTTP calls or model loading. It prints plan and request
hashes. The four-hour budget covers HTTP preflight, inference and associated
client work; CPU preparation and server loading/idle time are recorded separately.
An unknown interrupted budget window conservatively consumes its reserved remainder.

Start a new serving session from this updated checkout in a separate terminal.
If an older Qwen launcher is running, stop it with Ctrl+C in its original terminal
and let it clean up first. Preserve all session records. Do not stop unrelated jobs.

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

Keep that terminal open. Once the server log shows application startup complete,
run in another terminal, entering the full newly printed `server-...` directory
name including its prefix:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Current serving directory (server-...): " judge_session &&
timeout --signal=TERM --kill-after=60s 300m \
  python -m post_thesis.llm_judge.run_halubench_scores \
  --server-session-id "$judge_session"
```

The server session must match the new checkout revision and unchanged pinned
profile. Endpoint checks run before inference. No automatic fallback to another
model, prompt, mapping or server is permitted.

## Durable execution and replay

Each request is journaled before transport and committed after completion.
Summaries checkpoint every 25 responses and progress prints every 100. A clean
partial invocation can continue under the same identity; completed requests are
not repeated. Malformed responses, transport failures or interrupted requests
halt the run without retries. Their usage remains unknown when unavailable.
Do not delete records or change the run ID to silently retry failures.

After successful completion:

```bash
python -m post_thesis.llm_judge.run_halubench_scores --inspect
```

Expect 8,000 valid scores, no failures or pending examples, zero new attempts,
and the same report hash. Inspection needs no live server, tokenizer execution
or HTTP. It validates the recorded preparation and replays saved raw responses.
The invocation-local `new_attempts` field is excluded from the report hash.

Private inference artifacts live in
`.artifacts/post_thesis/llm_judge/qwen3-halubench-test-label-score-v1/`.
The external five-hour timeout bounds one invocation, including preparation;
the internal client budget is four hours. Server loading/idle time is not included
in client timing. No hourly price is invented: serving-session resources preserve
the lifetime record, with monetary cost unknown unless a rate is supplied.

Local checks exercise mock HTTP, durable replay, malformed output, interruption,
budget exhaustion and baseline alignment. Live scoring remains on the pod.
After scoring and replay, the next step is frozen cross-domain metric evaluation.
