# Post-thesis fresh MiniCheck RAGTruth TEST run

The [v2 synthetic check passed on the H200](../../results/post_thesis/llm_judge/minicheck_synthetic_v2_20260928.md).
This runner applies that configuration to the canonical 2,700-example TEST
manifest. It produces fresh per-example baseline scores and input provenance.
It never fits a model, changes a threshold, calls the Qwen judge or computes
label-based TEST metrics. Submitted thesis files and legacy caches are preserved.

## Fixed model and semantics

The pinned MiniCheck weights, saved-tokenizer adapter, NLTK resources, engine,
package versions, prompt and stop-token construction must match the preserved
successful v2 report. The initialization uses native InternLM2, BF16 H200,
FlashAttention, raw logprobs and `VLLM_USE_FLASHINFER_SAMPLER=0`.

Only answer and context enter the prompt. IDs and input hashes align results
outside the model. Manifest integrity verification includes its offline-label
section, but the inference projection does not pass labels or metadata to the
model or calculate performance metrics.

Context newline/sentence chunking and answer sentence splitting follow the
pinned upstream implementation. Each example makes one `generate` call with
all chunk/sentence prompts in chunk-major order. The score is the minimum over
answer sentences of their maximum support across context chunks. Each sentence
support is the sum of probabilities for returned first-position tokens whose
lowercased decoded text is exactly `yes`. No yes/no renormalization, leading-space
stripping or prompted numeric probability is substituted.

The historical TRAIN-selected decision remains support `< 0.20000000000000004`
(exact hex `0x1.999999999999bp-3`). The separate upstream `> 0.5` supported label
is retained for traceability, not substituted for the frozen decision.

## One preparation and scoring command

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export NLTK_DATA=/root/nltk_data &&
export HF_HOME=/root/llm-judge-hf-cache &&
timeout --signal=TERM --kill-after=60s 150m \
  python -m post_thesis.llm_judge.run_minicheck_test
```

The command first hashes the cached files, verifies the successful synthetic
configuration, and prepares all TEST prompt lengths on CPU. It saves the exact
workload and input fingerprints before loading any model weights onto the GPU.
No benchmark model call occurs if an input is overlength, the preparation has
changed, or a count limit is exceeded. To perform only this preparation, append
`--prepare-only`; no separate code patch is required to proceed to scoring.

The fixed limits in `PLAN` are at most 2,700 examples, 100,000 sentence requests,
100 million input tokens and 512 requests in any example. Each request allows
one output token and at most 32,767 input tokens. Actual counts are computed
from all inputs and printed before inference. No prompt truncation is permitted.
Empty answer sentence lists stop preparation rather than receiving a made-up
score. Context chunking may normalize whitespace exactly as recorded upstream;
the full canonical input hash and the actual chunks/sentences are both retained.

The run allows one attempt per example, at most four model initializations for
partial-run continuation, and 7,200 known client seconds across inference and
initialization. The time check runs before starting new work and does not
preempt an active example. CPU preparation and report I/O are outside those
measured stages. The external 150-minute timeout bounds an invocation. An
interrupted attempt has unknown usage and stops further automatic inference.
At least 64 GiB GPU memory must be free before model initialization.

## Persistence and replay

Progress prints every 25 newly attempted examples. SQLite commits after every
example. Initialization is separately journaled before loading, so a startup
failure is retained and not silently retried. Valid results retain returned
first-position logprobs, input-token hashes and support matrices. Summaries are
recomputed from those records, including validation of score aggregation.

After the run completes, verify replay:

```bash
python -m post_thesis.llm_judge.run_minicheck_test --max-new-examples 0
```

Expected: `Valid scores: 2700 / 2700`, no terminal failures or pending examples,
`New attempted examples: 0`, and the same summary hash. This command still does
CPU preparation and file hashing, but never loads the model. A partial run that
stopped cleanly can continue with the original command; completed examples are
not regenerated. Do not rerun the old synthetic check after a code update:
its original run identity is preserved and the new runner verifies it read-only.

Timing is synchronized request completion as exposed by `LLM.generate`, with
CPU prompt preparation/token checks included in each example timer. Model
initialization/warm-up time is reported separately. It is not a CUDA-kernel-only
benchmark or an apples-to-apples timing comparison with S2/S4. Failed/interrupted
example usage is explicitly unknown; zero tokens are never assigned as an
estimate for that missing usage.

Private records live in
`.artifacts/post_thesis/llm_judge/minicheck-ragtruth-test-fresh-v1/`:
`preparation.json`, `journal.sqlite3`, `initializations.sqlite3`, `summary.json`.
Keep the successful synthetic v2 directory and tokenizer adapter. No new model
or resource downloads are enabled.

The [cluster run and replay are complete](../../results/post_thesis/llm_judge/minicheck_fresh_test_cluster_20260928.md).
Run `python -S -m post_thesis.llm_judge.compare_minicheck_legacy` for the next
CPU-only numerical comparison, then assemble the baseline
comparison manifest. Fresh inference resolves current input/weight visibility;
it cannot prove which runtime generated the historical caches or independently
prove the original S4 training exclusions. Those limitations remain explicit.
