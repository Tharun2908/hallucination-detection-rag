# Post-thesis fresh MiniCheck on canonical HaluBench

The fresh S2, S4 and frozen metadata-free fusion HaluBench runs have completed.
This step scores the unchanged canonical 8,000 TEST rows with the MiniCheck
runtime already validated on RAGTruth. It is excluded from submitted thesis
results. No target-domain training, threshold tuning, judge calls or TEST
performance metrics occur.

## Frozen inference

Use `bespokelabs/Bespoke-MiniCheck-7B` revision
`1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8`, the preserved successful synthetic v2
report and its saved tokenizer adapter. The runner rechecks model-file bytes,
tokenizer behavior, package versions, English Punkt resources and runtime
settings against that report. It also verifies the completed fusion report
`2dbd25f11f167cc7e0ec74c1557450dcf2407cb9ea89db53e876d84f2acea415`
and its canonical input alignment. No downloads or dependency changes occur.

Byte-pinned shared helpers preserve BF16 native InternLM2 inference on H200,
FlashAttention, eager execution, seed 2024, no prefix caching, raw logprobs and
the disabled FlashInfer sampler. Inputs contain only answer and context. Offline
labels participate in manifest integrity verification, never in model prompts.

Document chunks and answer sentences retain the existing upstream preprocessing.
One example submits its chunk-major sentence prompts together. Support is the
sum of returned token probabilities whose lowercased decoded text equals `yes`,
then the maximum across chunks for each answer sentence, then the minimum across
answer sentences. No yes/no renormalization or whitespace stripping is added.

Preserve the historical RAGTruth TRAIN decision: support
`< 0.20000000000000004` (hex `0x1.999999999999bp-3`). The upstream supported label
at `> 0.5` is recorded separately. No HaluBench-derived threshold is substituted.
MiniCheck's sentence splitting differs from S2's length-filtered splitting:
S2's 1,886 empty-pair raw-zero fallbacks are not copied into MiniCheck.

## Pod command

If Qwen is running, stop its `post_thesis.llm_judge.serve` launcher with Ctrl+C
in its original terminal and let it finish shutting down. Preserve its session
records. Do not terminate unrelated processes. Check `nvidia-smi`; the MiniCheck
backend requires at least 64 GiB free on GPU 0 and fails closed otherwise.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export NLTK_DATA=/root/nltk_data &&
export HF_HOME=/root/llm-judge-hf-cache &&
timeout --signal=TERM --kill-after=60s 150m \
  python -m post_thesis.llm_judge.run_minicheck_halubench
```

Before loading the model, the command checks every input and prompt on CPU,
prints the full workload and saves `preparation.json`. To run just this portion,
append `--prepare-only`; the same command can subsequently proceed to inference.
Every input must fit without truncation. Empty answer sentence lists stop
preparation instead of receiving a fabricated score.

Local CPU preparation found exactly 8,000 examples, 9,158 sentence requests and
4,905,370 input tokens (102–6,756 per prompt), all fitting without truncation.
The plan caps the run at these totals, pins every prepared input reference, and
also limits each example to 512 requests. The pod must reproduce these counts
and hashes before any model loading. Each sentence request generates one
token, with at most 32,767 input tokens. It permits one attempt per example,
no automatic retries, at most four model initializations, and 7,200 known client
seconds across initialization and inference. Time is checked before new work;
an active example is not preempted by the internal budget. The external
150-minute timeout bounds an invocation. Preparation and report I/O are outside
the recorded model-stage timing.

## Persistence and replay

SQLite commits each example and records initialization separately. Terminal
errors or interrupted attempts are retained with unknown usage and block further
automatic inference. A clean partial run can continue with the same command and
unchanged code; completed examples are not repeated. Progress prints every 25
newly attempted examples.

After completion:

```bash
python -m post_thesis.llm_judge.run_minicheck_halubench --max-new-examples 0
```

Expect 8,000 valid scores, no failures/pending examples, zero new attempts on
replay and the same report hash. Replay still checks files and prepares inputs
on CPU, but does not load MiniCheck or rewrite an identical summary.

Private records are under
`.artifacts/post_thesis/llm_judge/minicheck-halubench-test-fresh-v1/`:
`preparation.json`, `journal.sqlite3`, `initializations.sqlite3`, `summary.json`.
Raw responses and actual prepared chunks/sentences are retained. No legacy file
or prior run is overwritten.

## Limits and subsequent step

This transfers the fresh-runtime RAGTruth MiniCheck baseline, whose scores drifted
from historical caches. It does not establish which runtime generated those
caches, and it does not recalibrate the transferred threshold. The canonical
8k membership and previously documented exact-passage overlap remain unchanged;
no adaptation training is used. Timings include CPU work within each completed
example, with model loading reported separately; they are not directly comparable
to S2/S4 CUDA-only times.

Local execution tests use artificial model responses. CPU input preparation is
recorded separately from GPU compatibility: live HaluBench scoring remains for
the pod. Frozen Qwen scoring and cross-domain evaluation follow this step.
