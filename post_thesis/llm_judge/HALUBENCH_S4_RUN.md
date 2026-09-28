# Post-thesis fresh S4 HaluBench zero-shot inference

The HaluBench input manifest and CPU judge token audit are complete. This plan
produces fresh S4 predictions for the same canonical 8,000 TEST examples. It uses
the existing RAGTruth-trained checkpoint; no HaluBench adaptation is performed.
Submitted thesis caches and prior post-thesis RAGTruth artifacts remain intact.

## Fixed inputs and workload

- Canonical manifest SHA256:
  `0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded`.
- Completed judge token audit SHA256:
  `7173c30e3a197aabb61c8ffd8c0e145cc556affa9aad20d7dcd43bc9cb24782a`,
  at revision `cbc9cca82ebeb46daa42e82722fe8c5cdff141ab`. Both operator runs
  reported the same audit hash; replay counted zero new inputs. All 8k inputs
  fit the judge window, with 7,794,485 input tokens and maximum 7,298.
- S4 reuses the four previously fingerprinted final checkpoint files and the
  completed CPU compatibility report
  `504dccf973f8c9995a02128332dc21a83d97c80c594cd1cbaee8396b11653077`.
- **500 fixed batches of 16**, at most one attempted forward per batch across
  all resumes. Canonical batch-reference SHA256:
  `eca3ad65ea7d1bbe05d588ca13d410afa39d86451e28d7974d289fb8f6d1db03`.
- Same H200 GPU 0 profile as fresh RAGTruth S4: FP32, TF32 disabled, eager
  attention, evaluation mode, no gradients, seed 42, two CPU threads.
- Same saved tokenizer: answer/context pair, longest-first truncation to 512,
  padding to 512; only `input_ids` and `attention_mask` enter the model.
- Save full-precision class-1 softmax and logits, exact original input hashes,
  effective forward-tensor hashes and answer/context token coverage.
- The historical TRAIN threshold remains **unsupported score >= 0.55**.
  This command does not apply it or compute TEST metrics.
- Same 1,800-second per-invocation deadline checked before batches; an active
  batch is not preempted. Total attempts remain bounded by 500 regardless of
  how many invocations resume the run.

The code pins the previously exercised S4 runtime/helper source hashes. It
verifies checkpoint, compatibility, environment, manifest, batch references and
completed token audit before creating the GPU backend. Source names, questions,
labels and group metadata stay outside the model input.

S4's 512-token limit is preserved even though the judge has a larger window.
The report records actual S4 truncation; no equal-evidence claim is made.
The canonical source/question/passage groups remain unchanged. Exact-passage
links to the unused adaptation pool are disclosed in the input manifest.

## Run on the pod

After applying and committing the patch on Windows:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.run_s4_halubench \
  --checkpoint /workspace/signal4_model
```

No packages or model files need downloading. The loader requires the existing
Python 3.12 environment with torch 2.13.0+cu130, Transformers 5.17.0,
tokenizers 0.23.2, safetensors 0.8.0 and huggingface-hub 1.32.0. It checks for
at least 3 GiB of free GPU memory; no automatic precision or batch-size fallback
is allowed. Qwen may remain resident, but sharing can still cause memory pressure.
If an error occurs, preserve the run and inspect it before continuing.

After completion, verify the stored scores without loading the model:

```bash
python -m post_thesis.llm_judge.run_s4_halubench \
  --checkpoint /workspace/signal4_model --max-new-batches 0
```

Successful batches are never repeated. Failed or interrupted batches remain
terminal with explicit missing scores, not zeros. Errors stop an invocation;
manual resumption can process only previously unattempted batches. Partial
invocations retain original batch boundaries. Keep the code revision and package
versions unchanged while completing and replaying this run.

Private records:

```
.artifacts/post_thesis/llm_judge/s4-halubench-test-fresh-v1/journal.sqlite3
.artifacts/post_thesis/llm_judge/s4-halubench-test-fresh-v1/summary.json
```

The journal is authoritative. The derived summary can be reconstructed from it
without forward calls. On an unchanged completed run, its report hash and bytes
remain identical on replay.

## Interpretation and remaining work

This establishes current S4 inference linkage, not historical training membership
or the exact inputs behind the old caches. Comparison readiness remains false
until all planned baseline inputs and provenance are resolved. No model selection
uses HaluBench labels or scores.

Timing reports synchronized GPU forward time and batch wall time including
preparation/transfer. Model loading, journal overhead and total rental duration
are excluded; shared-GPU timings are not controlled cross-system speed comparisons.
There is no invented monetary cost.

Next are fresh S2 features, application of the frozen metadata-free fusion, and
fresh MiniCheck predictions; then bounded frozen judge inference and the registered
cross-domain evaluation. The judge, calibration and operating thresholds remain
unchanged throughout.
