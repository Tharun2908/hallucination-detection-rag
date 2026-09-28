# Post-thesis fresh S2 HaluBench features

Fresh S4 inference and replay completed on all 8,000 canonical HaluBench TEST
examples, with report SHA256
`2d68d0657d983db0df84b146264652eea00034734e82d09a42126c56e4ba0f2c`.
S4 truncated context in 850 examples and answers in none. The original input
hashes and truncation records are preserved. No new TEST performance metrics
have been calculated.

This step produces fresh raw S2 relevance features on those identical inputs.
It reuses the RAGTruth S2 runtime and preserves the frozen TRAIN normalization
and fusion model for later application. No HaluBench fitting is performed.

## Fixed plan

- Same canonical 8k input manifest, SHA256
  `0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded`.
- The completed S4 report is checked read-only for its hash, input order and
  all 8,000 exact answer/context hashes before S2 model loading.
- S2 checkpoint: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Existing model-file hashes and
  reused implementation-file hashes are checked; no newer revision is substituted.
- Same original regex sentence split, newline replacement and minimum
  10-character retained-sentence rule for both answer and context.
- Fixed workload: **8,000 examples, 81,107 sentence pairs, 8,327 forward batches**.
  Batches contain up to 32 pairs, stay within an answer sentence, and retain
  original context-sentence order. References SHA256:
  `92ce62a176b1963a87e49c260f081c6fe573e915debd8070a6594f018134959d`.
- Same H200 GPU 0, FP32, TF32 off, eager attention; longest-first 512-token
  truncation and padding to the longest input in each batch. All three BERT
  input fields are preserved. Original/effective input fingerprints, raw pair
  logits and truncation coverage are recorded.
- Relevance uses raw logits with Identity activation: maximum over context
  sentences for each answer sentence, then minimum of those maxima for fusion.
- At most one attempted execution per example over all resumes. The same
  7,200-second per-invocation deadline is checked before examples; an active
  example is not preempted. Resume cannot repeat attempted examples.

## Empty-pair behavior is preserved

Local CPU preprocessing found **1,886/8,000 examples (23.575%)** with no retained
answer sentences. Every context retains at least one sentence. Counts by source:
DROP 255, FinanceBench 450, halueval 1,104, covidQA 77, pubmedQA 0.
These counts use input preprocessing, not model predictions or label metrics.

The original S2 policy assigns raw minimum and mean relevance **0.0** and an
empty per-answer maxima list when no pair exists. It performs no pair forward
for that example. This is a predefined feature fallback, not a hallucination
probability or a fabricated replacement for a failed model call. Failed or
interrupted attempts still have missing (`null`) features. Do not shorten the
sentence filter after inspecting target-domain data; that would be a different
system. Record the limitation in later fusion interpretation.

## Run on the pod

After applying and committing the patch on Windows:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.run_s2_halubench
```

The default model cache is `/root/llm-judge-hf-cache`, already used for fresh
RAGTruth S2. No download is enabled by default. If that cache was lost during a
pod replacement, the same command supports `--download-model` to fetch only the
pinned S2 files. Do not change package versions or model revision to resolve a
missing-cache error. The profile checks for at least 3 GiB free GPU memory;
no automatic precision or batch-size fallback is used.

On completion, verify replay with no model loading or new examples:

```bash
python -m post_thesis.llm_judge.run_s2_halubench --max-new-examples 0
```

Keep the same code revision and environment while running/replaying. Successful
examples are durable. A failed or interrupted example is terminal and is never
automatically retried; errors stop the invocation. Manual resumption may process
only previously unattempted examples. Do not delete journals to force a rerun.

Private artifacts:

```
.artifacts/post_thesis/llm_judge/s2-halubench-test-fresh-v1/journal.sqlite3
.artifacts/post_thesis/llm_judge/s2-halubench-test-fresh-v1/summary.json
```

The journal contains the raw pair logits/effective-input provenance. The derived
summary contains raw features, truncation counts and operational timings. Complete
replay yields the identical summary hash. No labels, source names, questions or
group IDs enter the scorer; these remain offline.

Shared-GPU example/forward timings exclude model loading and journal overhead;
they are not a controlled speed benchmark or a monetary-cost estimate. Fresh
inference linkage does not retroactively attest historical caches or training
membership. No legacy files, RAGTruth scores or frozen judge choices are changed.

After these features are complete, apply the previously frozen metadata-free
fusion with its fixed TRAIN normalization and threshold. Fresh MiniCheck and
bounded frozen-judge HaluBench scoring follow before cross-domain evaluation.
