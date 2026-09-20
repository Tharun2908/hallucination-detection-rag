# Post-thesis legacy baseline provenance audit

The completed cluster TEST manifest and replay match the locally reconstructed
SHA256 `bc564d50ed4c0fcff17b5c8dda4feef558d896f2f0c8b4c7239a5085f9ccc3d3`.
The operator located all six historical caches under `/workspace`, each with
15,090 TRAIN or 2,700 TEST rows. The TEST inventory found no missing scores or
indices and matching labels/metadata, but no input hashes. The final S4 and five
OOF model configuration paths were found. The bounded search found no joblib or
fusion-pickle artifact; this does not establish global absence. See the
[operator inventory record](../../results/post_thesis/llm_judge/legacy_baseline_inventory_20260920.json).

## Purpose and limits

This CPU-only audit checks what the available artifacts can establish:

1. Pin all six cache file hashes to the operator's originals. Verify exact unique
   index coverage and label/task/generator alignment against the pinned processed
   TRAIN data and completed TEST manifest. Preserve missing scores; no imputation.
2. Reconstruct historical S4 OOF held-out assignments using five shuffled
   stratified folds and seed 42, and compare every stored fold and score-type tag.
   Matching tags do not independently prove which examples trained a checkpoint.
3. Reproduce the S4 and MiniCheck-7B TRAIN threshold audit using only cached TRAIN
   scores and labels. Preserve the historical NumPy grids, first-maximum F1 tie
   rule, score orientation and equality handling. Compare against the committed
   historical reference; mismatches are reported and never replace the policy.
4. Fingerprint current final/fold checkpoint configuration, tokenizer and weight
   files. Hash bytes without loading/deserializing model, pickle or training-args
   objects. Current fingerprints do not establish a historical checkpoint/cache
   relationship by themselves.

The historical S4 policy is unsupported score >= 0.55. MiniCheck's recorded policy
is support score < 0.20000000000000004. Preserve that exact float and comparator;
do not silently round it to 0.2 or substitute an unsupported-score >= policy.
The audit uses existing cached precision, not invented unrounded model outputs.
It does not compute TEST predictive metrics or use TEST outcomes to choose a
baseline, threshold, judge prompt or model.

The pinned baseline reference is
`results/evaluation/table41_threshold_audit_results.json`, SHA256
`1a7af7f7dc7c98dca5672fb9467dc0c9402880043282ebf2a12f16503b8b1dc6`.
The existing code `evaluation/table41_threshold_audit.py` defines the comparison
and grid semantics reproduced here. Metadata-free fusion recovery remains a
separate CPU operation using the established training procedure and cached
features; this audit does not fit or select a fusion model.

This is post-thesis artifact verification, not a claim that legacy predictions
have retrospectively gained input hashes. All comparison-ready flags remain
false until content provenance, checkpoint linkage, evidence visibility and
fusion predictions are resolved or explicitly handled in the comparison design.
Do not append current input hashes to old score rows as evidence of what an
unrecorded historical inference actually consumed.

## Run on the pod after applying, committing and pulling the patch

No GPU server is required. Keep the original caches, snapshots, fit and TEST
manifest intact. The CPU environment retains the frozen NumPy/SciPy versions:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m pip install -r post_thesis/llm_judge/requirements-baseline-audit.txt &&
python -m post_thesis.llm_judge.audit_baseline_provenance \
  --baseline-dir /workspace \
  --train-parquet .artifacts/post_thesis/llm_judge/source-audit-inputs-v1/train.parquet \
  --checkpoint-root /workspace
```

The program prints progress for each checkpoint directory because hashing model
files can take time. This is disk reading, not inference. It creates a private
report at:

```text
.artifacts/post_thesis/llm_judge/ragtruth-legacy-baseline-provenance-v1/report.json
```

Send the cache checks, OOF summary, threshold reproduction, checkpoint summaries,
report hash and code revision. The full checkpoint file fingerprints remain in
the private report. An identical same-checkout invocation verifies and reuses
that report; changes fail rather than overwrite. Do not rerun older preparation
commands after this checkout update: the audit consumes their pinned artifacts.

No new model calls, judge refitting, TEST metrics, split creation or scoring
allowance is introduced. The next decision follows the actual audit result;
threshold agreement alone cannot settle input or checkpoint provenance.
