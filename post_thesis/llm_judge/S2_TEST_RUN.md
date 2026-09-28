# Post-thesis fresh S2 TEST inference

This run is post-thesis baseline preparation, separate from submitted results.
The fresh S4 run and score agreement are recorded in
[the operator observations](../../results/post_thesis/llm_judge/s4_fresh_test_cluster_20260920.md).
This step produces S2 features with current input/model fingerprints; it does
not change the historical caches, fit normalization/fusion, or compute TEST metrics.

## Fixed inference profile

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`. This is a fresh post-thesis pin;
  the historical inference revision has not been proven. The old alias has
  an extra hyphen in `MiniLM-L-6-v2` and redirects to this model.
- Six downloaded files are checked against byte sizes and SHA256 values in
  `s2_inference.py`. Weight identity comes from the pinned release's LFS
  metadata and is checked against downloaded bytes before inference.
- H200 GPU 0, FP32, eager attention, TF32 disabled, evaluation mode, no gradients.
  The existing judge-serving environment is required unchanged. No dependency
  installation or sentence-transformers installation is needed.
- The sentence splitter reproduces `signals/relevance_verifier_full_v2.py`:
  replace newlines with spaces, split after `.`, `!`, or `?` followed by
  whitespace, strip, and keep sentences of at least ten characters.
- Answer sentence first, context sentence second. Each answer sentence is
  compared with every context sentence in its original order, in batches of
  at most 32. BERT receives `input_ids`, `attention_mask`, and `token_type_ids`.
  Tokenization uses dynamic batch padding and longest-first truncation at 512.
- The model configuration specifies **Identity** activation. Preserve raw
  logits, including negative values. For each answer sentence, take the maximum
  context score; `raw_min_relevance` is the minimum of those maxima. This is
  the fusion feature. Also retain the mean and the per-answer maxima.
- Preserve the historical empty-sentence policy (zero min/mean and no maxima).
  There are no such cases in the canonical 2,700-row TEST workload.

The original CrossEncoder implementation's batching, pair tokenization and
Identity activation can be inspected in its
[official source](https://github.com/UKPLab/sentence-transformers/blob/v2.7.0/sentence_transformers/cross_encoder/CrossEncoder.py).
This supports the explicit fresh profile; it does not establish which library
version produced the old cache. The pinned model configuration is
[here](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/config.json).

## Bounds, records and recovery

The canonical manifest must yield exactly **2,700 examples, 362,918 sentence
pairs and 20,576 forward batches**. One attempt is permitted per example.
Each invocation has a 7,200-second soft deadline checked before starting an
example; an active example is not preempted. Model loading/download time counts
toward this invocation limit. A deadline or `--max-new-examples` limit leaves
pending examples which the same command can resume, under the same clean commit.

An example's reservation is committed before inference. Its result is committed
after all sentence pairs complete. Finished examples are never recomputed on
resume. A crash or exception leaves an explicit terminal failure, not a zero
feature or automatic retry. If failure occurs after some forward batches, those
partial calls have unknown accounting; they do not become completed-pair counts.
Stop and inspect a terminal failure before deciding on another experiment.

The SQLite journal retains raw pair logits, batch tensor hashes, full/kept token
counts for each side, timings, and runtime identity. The smaller `summary.json`
contains per-example input/sentence hashes, full-precision aggregate features,
coverage and timings. Aggregates are recomputed from journal logits on replay.
The summary updates every 25 new examples and at invocation completion; committed
progress is available after every example. A completed replay does not load or
download the model and preserves the report hash.

Timings are descriptive shared-GPU measurements: model loading and journal I/O
are excluded from per-example timings. Do not use them as a controlled latency
comparison with Qwen or S4. GPU memory availability is checked before loading.
The Qwen server can remain loaded if at least 3 GiB is free.

Only answer/context reach the model. Ground-truth labels, generator/task metadata
and queries are not scoring inputs. The canonical manifest is validated locally;
labels present in that manifest are not used to evaluate or tune this run.

## Pod commands

After committing and pushing this patch, in the existing pod:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.run_s2_test \
  --model-cache /root/llm-judge-hf-cache \
  --download-model
```

The six pinned files total about 92 MB and are cached outside `/workspace`.
No original TRAIN/TEST relevance script should be launched: it also fits
normalization and writes historical paths.

After completion, verify replay with:

```bash
python -m post_thesis.llm_judge.run_s2_test --max-new-examples 0
```

Expected: zero new attempts, identical report hash, no model loading. Reports live
in `.artifacts/post_thesis/llm_judge/s2-ragtruth-test-fresh-v1/`.

## Validation and next use

Offline tests cover raw min-of-max aggregation across batch boundaries, original
sentence-split equivalence, invalid/missing pairs, full/kept token records,
partial resume, interruptions, errors and replay without loading. Actual pinned
tokenizer files were checked locally: vocabulary 30,522, limit 512 and four
short/long pair encodings match the saved tokenizer, including token type IDs.
No local weight inference was performed; the pod run verifies that runtime path.

After the run, compare fresh raw features to the preserved legacy S2 cache
without labels. Close agreement will not manufacture historical provenance.
Keep full precision here. Any later application of the frozen fusion must
explicitly account for its historical four-decimal input representation and
fixed TRAIN normalization; do not refit the fusion from TEST observations.
