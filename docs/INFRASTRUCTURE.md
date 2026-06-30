# Infrastructure Notes

This document records the environment, storage layout, and operational details needed to reproduce the experiments in this repository. It captures the specific traps encountered during the project so that someone reproducing this work has the actual fixes, not just a list of pinned versions.

## Hardware

Two GPU environments were used:

- **Tesla V100S-PCIE-32GB** — original signal training (S1–S5, S8), MiniCheck baselines, fusion, cascade, robustness experiments, efficiency benchmark, RAGTruth++ retraining, HaluBench adaptation curve, and the bidirectional cross-domain study.
- **NVIDIA H200** — out-of-fold S4 retraining for the final fusion protocol. Used purely to reduce turnaround time on the 5-fold retraining sweep; nothing in the experimental design depends on H200 specifically.

The V100 has compute capability 7.0, which constrains a few dependencies — most notably `xformers`. Newer GPUs can use newer versions; the V100 cannot.

Both environments were Kubernetes pods in a shared namespace, with a persistent volume mounted at `/workspace`. The container overlay is separate from the persistent volume and behaves very differently — see "Storage layout" below.

## Storage layout

```
/workspace                    # persistent volume (PVC, ~50 GB)
  ├── *.py                    # working scripts (not all are in the repo)
  ├── *_results_*.json        # per-example score files (large, not in Git)
  ├── *_metrics_*.json        # aggregate metrics (in Git, under results/)
  ├── signal4_model/          # final S4 checkpoint (not in Git; HuggingFace release planned)
  ├── signal4_oof_models/     # OOF fold checkpoints (not in Git)
  ├── models--bespokelabs--Bespoke-MiniCheck-7B/  # ~15 GB MiniCheck cache
  └── merlin/                 # MERLIN-DDx clinical extension (data not in Git)

/root/.cache                  # container overlay (~200 GB, but wiped on pod recreation)
  └── huggingface/hub/        # HuggingFace's default model cache
```

Three things about this layout matter operationally:

**The container overlay is much larger than the persistent volume, but it disappears.** Any HuggingFace model that downloads into the default cache (`/root/.cache/huggingface/hub`) is lost when the pod is recreated. Recreating a pod is something the cluster does for you, sometimes unexpectedly.

**Large model caches must live on `/workspace` and be symlinked into the default cache location:**

```bash
# After the first MiniCheck-7B download (~30 minutes, ~15 GB):
mv /root/.cache/huggingface/hub/models--bespokelabs--Bespoke-MiniCheck-7B /workspace/
ln -s /workspace/models--bespokelabs--Bespoke-MiniCheck-7B \
      /root/.cache/huggingface/hub/models--bespokelabs--Bespoke-MiniCheck-7B
```

The symlink itself does not persist across pod recreation (it lives on the overlay), so it has to be recreated each time. The model weights, however, are safe on `/workspace`.

**Per-example score files are intermediate artifacts.** Files like `signal4_results_train_oof.json`, `relevance_results_test_v2.json`, and the HaluBench per-example scores stay on `/workspace` and are excluded from Git. Aggregate metric JSONs (`signal4_metrics.json`, `complete_metrics_results.json`, etc.) are committed under `results/`.

## Pod operation

Common operations:

```bash
# Get a shell on the GPU pod
kubectl exec -it ml-training -n <namespace> -- bash

# Check disk usage when things get tight
df -h /
du -sh /workspace/* | sort -rh | head -20

# Run long jobs unbuffered so logs are useful
nohup python -u /workspace/script.py > /workspace/script.log 2>&1 &
tail -f /workspace/script.log
```

Two operational gotchas:

**`nohup` buffers output.** Without `-u` (or `PYTHONUNBUFFERED=1`), nothing shows up in the log until the script exits or the buffer fills. The `python -u` flag is the simplest fix.

**`ml-pvc` can only mount on one pod at a time.** If a CPU pod and a GPU pod are both trying to mount it, the second one will fail with a multi-attach error:

```bash
kubectl delete pod ml-cpu -n <namespace>
# then start the GPU pod
```

## Dependency stack

The combination below is the one that has been verified end-to-end. Several of the pins are tighter than they look — substituting "close" versions has produced subtle failures during this project.

```
torch==2.3.0
transformers==4.44.0
vllm==0.4.3
xformers==0.0.26.post1
```

Why each pin matters:

- **`transformers==4.44.0`** — the saved S4 checkpoint was trained with this version and uses internal APIs that change in newer releases. Loading it with `transformers>=4.45` fails with a `register_fake` error. Some newer libraries silently upgrade transformers as a transitive dependency, so this version can re-break itself after an apparently-unrelated `pip install`. Re-pin if loading suddenly stops working.
- **`vllm==0.4.3`** — required by the MiniCheck-7B baseline (which uses vLLM under the hood). vLLM 0.4.3 pins `torch` back to `2.3.0+cu121`. This is intentional and should not be "fixed" by upgrading torch.
- **`xformers==0.0.26.post1`** — required for V100 compute capability 7.0. Newer xformers releases raise `NotImplementedError` on the V100. On newer GPUs (H200) this pin is unnecessary, but keeping it does no harm.

Other dependencies (`scikit-learn`, `sentence-transformers`, `bert-score`, `nltk`, `fsspec`, `sentencepiece`, `datasets`) are not version-sensitive.

Required one-off setup after the first install:

```bash
# MiniCheck-7B's sentence splitter
python -c "import nltk; nltk.download('punkt_tab')"

# Required for the `hf://` URLs used to load RAGTruth++ via pandas
pip install -U "fsspec>=2024.2.0"
```

## Cluster recovery (fresh pod)

When a pod is recreated, this is the recovery sequence:

```bash
# 1. Restore the MiniCheck cache symlink
mkdir -p /root/.cache/huggingface/hub
ln -s /workspace/models--bespokelabs--Bespoke-MiniCheck-7B \
      /root/.cache/huggingface/hub/models--bespokelabs--Bespoke-MiniCheck-7B

# 2. Reinstall the version-pinned stack
pip install -r requirements.txt --break-system-packages
python -c "import nltk; nltk.download('punkt_tab')"

# 3. Sanity-check CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If CUDA reports not-available after a clean install, the usual fix is `unset CUDA_VISIBLE_DEVICES` followed by reconnecting to the pod.

## Out-of-fold fusion protocol

The final fusion protocol uses out-of-fold S4 predictions on the RAGTruth training split:

1. The training split is partitioned into 5 stratified folds.
2. For each fold, an S4 model is fine-tuned on the other 4 folds, and the held-out fold is scored with it.
3. The concatenated out-of-fold predictions become the S4 feature seen by the logistic-regression fusion at fusion-training time.
4. At test time, S4 predictions are produced by a single S4 model trained on the entire RAGTruth training split.

This is the standard stacking protocol. It avoids training the fusion model on S4 features the base model had already memorized, which would produce optimistic feature distributions during fusion training. The S4 model used in production (test-time and HuggingFace release) is unchanged — the OOF protocol affects only the inputs the fusion meta-classifier sees during its own training.

The OOF script is `signals/signal4_oof_train_scores.py`. It writes:

- `/workspace/signal4_results_train_oof.json` — concatenated OOF predictions for the training split
- `/workspace/signal4_oof_metrics.json` — per-fold and aggregate metrics
- `/workspace/signal4_oof_models/fold_*/` — fold checkpoints (kept for diagnostic purposes; not in Git)

All fusion, cascade, and downstream evaluation scripts in the repository read `signal4_results_train_oof.json`. The one script that intentionally reads the original (non-OOF) `signal4_results_train.json` is `signals/signal4_score_train.py`, whose purpose is to *produce* that file for standalone-S4 diagnostics. This is not a holdout that needs fixing.