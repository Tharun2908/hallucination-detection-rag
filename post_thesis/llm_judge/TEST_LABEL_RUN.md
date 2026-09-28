# Post-thesis frozen Qwen RAGTruth TEST run

This extension is outside the submitted thesis. The frozen prompt, model,
class mapping, calibration and operating threshold remain unchanged. Fresh
baseline comparison verification completed with manifest SHA256
`3e8c25eaaeaf03aa93377f8a7b6c516e6a9f4e8c811305fb88683e78e0b615bb`.
MiniCheck remains a fresh runtime variant with a transferred historical TRAIN
threshold and unresolved legacy drift; see [COMPARISON_MANIFEST.md](COMPARISON_MANIFEST.md).

## Fixed execution allowance

The committed [plan](configs/ragtruth_test_label_scoring_v1.json) permits:

- Exactly the canonical 2,700 TEST inputs in original order; 3,334,607 audited
  input tokens, minimum 649 and maximum 2,849 per input. No truncation.
- Primary mapping only: A/token 32 = supported, B/token 33 = unsupported.
  The request contains token IDs encoding only the fixed prompt, answer and
  context. No labels, generator/task metadata or benchmark sample IDs.
- The existing `label-score-v1` Qwen3-32B BF16 H200 profile, raw log probabilities,
  temperature zero, no prefix caching, one generated token, one request at a time.
- At most 2,700 total attempts and 2,700 output tokens. One attempt per input;
  no automatic retries. Unknown token usage is retained explicitly.
- At most 7,200 cumulative client seconds across invocations, including HTTP
  preflight and scoring. Each HTTP request retains the existing 60-second limit.
  Unknown interrupted windows consume their full reserved remaining budget.
- Any request error halts the run. A deliberate partial invocation can resume
  unattempted inputs, but failed or interrupted attempts are not repeated.

This is a two-hour **client execution ceiling**, not an estimate of completion
or a cap on rental charges. CPU preparation and model startup are outside that
ceiling; serving lifetime includes startup and idle time. Monetary cost remains
unknown unless an actual rental rate and currency are recorded.

## Prepare on the current pod

Commit/push the patch and pull it on the pod before running. The tokenizer
files used for the earlier audit must still exist under the private artifact
root. This command does not download files, load weights or contact the server:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export HF_HOME=/root/llm-judge-hf-cache &&
python -m post_thesis.llm_judge.run_test_scores --prepare-only
```

It verifies the canonical input manifest, completed token audit, frozen judge
and numerical contract, completed fresh baseline reports, comparison manifest,
installed vLLM source fingerprints and tokenizer environment. It then re-renders
every request and requires exact agreement with the audited token fingerprints.
The existing canonical manifest includes offline labels for integrity checks;
those fields are never projected into a model request or used for selection.
The command prints build-tool availability for the new pod. This is not a live
Qwen compatibility test. Missing CUDA build tools previously prevented this
serving profile from starting; restore the known CUDA 13 toolchain if absent.
Do not change model, dtype, attention backend or sampling settings to work
around a failure without recording a separate implementation review.

## Start the same serving profile

Use the existing serving environment and the restored CUDA toolchain. Keep
model files under the larger root filesystem cache. Start from the same clean
commit as the client; free the GPU from previous model processes first.
Do not terminate unrelated processes automatically.

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

The launcher can download the pinned model if its cache was lost when the pod
was deleted. Use the printed session directory and server log to confirm
startup. The default serving profile has a different identity; it must not be
used for this run. If recording rental cost, add the actual `--hourly-rate` and
`--currency` values to the launcher command, not a guessed API token price.

## Score and replay

After server startup, open another shell:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Current server session directory name (server-...): " judge_session &&
python -m post_thesis.llm_judge.run_test_scores --server-session-id "$judge_session"
```

Enter the full directory name beginning `server-`, not just the suffix. The
client checks the exact launcher command, source, package version, model API
and registered profile before scoring. Progress counts appear every 100 new
attempts. Individual responses are journaled before the next request, with a
summary checkpoint every 25 attempts and on exit.

`--max-new-attempts N` limits only the current invocation; it cannot increase
the cumulative plan. Do not interpret partial TEST performance to change the
prompt or operating rules. To verify the completed run without HTTP or
retokenization:

```bash
python -m post_thesis.llm_judge.run_test_scores --inspect
```

Outputs are private under
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-test-label-score-v1/`:
`prepared.json`, `budget.json`, `journal.sqlite3`, and `summary.json`.
New-attempt counts are outside the report hash, so normal completed replay has
zero calls and an identical report hash. Preserve these files together and
back them up before deleting a pod. A changed checkout/source/plan or partial
artifact deletion fails closed; never reset a run to hide an attempt.

The result contains raw class-token scores, usage, timing, failures and serving
session snapshots. It computes no TEST metrics, performs no fitting, applies
no new threshold, reads no HaluBench data and rewrites no legacy caches.
Registered evaluation and frozen calibration application follow separately
once the complete judge output is verified. Local tests use fake tokenizers
and mock HTTP; they establish software behavior, not live model quality.
