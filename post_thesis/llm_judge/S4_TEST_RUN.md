# Fresh post-thesis S4 TEST inference

This run establishes current inference-input provenance for the saved S4
checkpoint. It is a post-thesis reproduction, not a replacement for the thesis
cache and not retrospective proof of the original training or inference history.

## Fixed scope

- The existing 2,700-example canonical RAGTruth TEST manifest, SHA256
  `bc564d50ed4c0fcff17b5c8dda4feef558d896f2f0c8b4c7239a5085f9ccc3d3`.
- The four final S4 checkpoint files already fingerprinted and successfully
  loaded by `check_s4_checkpoint`. Required compatibility report SHA256:
  `504dccf973f8c9995a02128332dc21a83d97c80c594cd1cbaee8396b11653077`.
- Existing Python 3.12 serving environment: Transformers 5.17.0, PyTorch
  2.13.0+cu130, tokenizers 0.23.2, safetensors 0.8.0, huggingface-hub 1.32.0.
- H200 GPU 0, FP32, TF32 disabled, eager attention, evaluation mode, no gradients.
- Fixed batch size 16; 168 full batches and one final batch of 12. At most
  **169 attempted batches / 2,700 examples** in the run, one attempt per batch.
- Answer first, context second; longest-first pair truncation to 512 tokens,
  padding to 512; only input IDs and attention masks enter the model.
- Class 1 softmax is the unsupported score. Preserve full precision and logits.
  The historical TRAIN threshold remains `>= 0.55`; this command does not apply
  thresholds, compute benchmark metrics or tune anything.
- A 1,800-second invocation deadline is checked before each batch. An active
  batch is not preempted. A partial run can resume the remaining batches.

The CLI verifies the frozen manifest and compatibility record, checks current
checkpoint fingerprints and records the code revision and package versions.
Labels/metadata in the manifest are covered by its integrity hash but are not
passed to the scorer. The scorer receives only answer/context text.

## Run

Apply and commit the patch on Windows, then on the pod:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.run_s4_test --checkpoint /workspace/signal4_model
```

No package installation or model download is required. Qwen may remain loaded;
the S4 loader checks for at least 3 GiB free GPU memory before loading. GPU memory
pressure still can cause a failure; there is no automatic precision/batch-size
fallback. If it fails, preserve the run and paste the error before proceeding.

Each batch has a durable SQLite reservation before tokenization/forward execution
and a committed result afterwards. Successful batches are never repeated. A
failed or interrupted batch is terminal, with explicit missing scores; it is not
silently retried or replaced with zeros. Errors stop the invocation. A manually
resumed run may process the remaining, previously unattempted batches.
`--max-new-batches N` can restrict an invocation without changing batch boundaries.

Rerunning the identical completed command verifies the journal and reuses all
scores with zero new attempts and no model loading. Keep the code revision and
environment unchanged while completing or replaying this run.

Private artifacts:
`.artifacts/post_thesis/llm_judge/s4-ragtruth-test-fresh-v1/`.
The SQLite journal is authoritative; `summary.json` is a derived report. Each
prediction includes its original TEST index, sample ID, exact answer/context
hash, effective input-ID/mask hash, answer/context token counts before and after
truncation, logits and full-precision score. The frozen tokenizer/checkpoint and
canonical text allow effective inputs to be reconstructed and checked by hash.

## Interpretation and timing

Fresh inference establishes which current inputs and checkpoint produced these
new scores. It does not certify the old cache's input bytes, historical checkpoint
linkage, or training exclusions. No legacy artifact is overwritten. Comparison
readiness remains false until the remaining baseline alignment work is complete.

GPU forward timing is synchronized and excludes tokenization and host transfer.
Batch wall timing includes tokenization, transfer and score extraction. Neither
includes model loading, journal persistence, server idle time or total pod rental.
Failed/interrupted batches have explicitly unknown timing. The first batch is
included; no extra warm-up forward is performed. Qwen can be resident and other
work is not controlled, so these timings are operational observations, not a
controlled latency comparison between systems. No API cost is assigned.

Next: check fresh-versus-legacy S4 score agreement without changing thresholds,
then finish S2/MiniCheck inference provenance for the registered comparison.
