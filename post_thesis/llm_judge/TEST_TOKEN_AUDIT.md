# Post-thesis RAGTruth TEST token audit

This CPU-only step follows the completed [metadata-free fusion reconstruction](../../results/post_thesis/llm_judge/fusion_recovery_cluster_20260920.md).
It belongs to the post-thesis extension, not the submitted thesis.

## Fixed inputs and boundaries

`audit_test_tokens.py` verifies the unchanged frozen judge fit, the completed TEST
manifest and the completed fusion report. It consumes those historical artifacts
read-only; it does not rerun their preparers or fitters after checkout updates.

- TEST manifest SHA256:
  `bc564d50ed4c0fcff17b5c8dda4feef558d896f2f0c8b4c7239a5085f9ccc3d3`.
- Fusion report SHA256:
  `1c90a003abbda9044767bb9167b1ecc171f7a8a746b87093a143d833ba4ec0d6`.
- Frozen judge SHA256:
  `90a70a63c550e87ad0f3fa41f11e1de99d9a08fc85c47de23f6ffafd3314fde8`.

Use all 2,700 original processed TEST rows in original order. Only the frozen
instruction plus exact answer and context enter tokenization. Sample IDs remain
outside model messages. Manifest checks include offline metadata/labels in the
integrity hash, but never send those fields to the tokenizer or a model.

Use the pinned local Qwen tokenizer/template and primary A=supported/B=unsupported
mapping. The prompt is `faithfulness-label-score-v1`; model/tokenizer revision,
non-thinking assistant boundary and rendering versions remain fixed. No download,
model weights, server request, generation or GPU is needed. Check the full rendered
input plus **one output token** against the 32,768-token model limit. No truncation,
reformatting, automatic omission or new sampling is permitted.

The private audit retains each input/messages/rendered-prompt/token-ID hash,
position and token count, but does not duplicate rendered source text. It records
complete coverage, overlength IDs, totals and pinned identities. A complete replay
performs zero new TEST tokenizations and does not rewrite the audit. The CLI still
checks four synthetic tokenizer-reference fixtures on startup; those checks are
not counted as TEST tokenizations.

Checkpoint every 25 successful rows and on a handled interruption. A hard process
kill may repeat the last incomplete checkpoint (up to 25 tokenizations); no model
calls can result. Corrupt rows or changed identities fail instead of overwriting
previous records. Any overlength inputs remain present and give a nonzero exit
status; resolve them explicitly before creating a scoring plan.

## Pod commands

Use the existing serving environment for its pinned Transformers/tokenizers
packages. This command performs CPU tokenization only; the vLLM server can remain
stopped. Do not install tokenizer packages into the separate numerical fitting
environment just for this step.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_test_tokens
```

Repeat the Python command once and send both summaries/hashes. The completed
frozen fit, TEST manifest, fusion report and small `label-tokenizer-cache-v1`
directory must remain in the existing artifact tree.

Private output:
`.artifacts/post_thesis/llm_judge/ragtruth-test-token-audit-label-score-v1/audit.json`.

## Local validation before the pod run

The [local CPU record](../../results/post_thesis/llm_judge/test_token_audit_local_20260920.json)
uses the pinned actual tokenizer and independently reconstructed TEST manifest
whose hash matches the completed pod manifest. All 2,700 inputs fit: **3,334,607
input tokens**, minimum 649, maximum 2,849. Replay performed zero new TEST
tokenizations and preserved the audit exactly.

This exercises tokenization and input projection, not the private fit/fusion
report checks: those full pod reports were not supplied locally. The pod CLI must
verify them. Its audit hash is expected to differ from local validation because
it records the committed code revision and actual package versions. Return the
pod hash rather than substituting the local one.

## What this audit does not establish

Fusion verification checks the supplied report hash, run identity, complete
prediction coverage and target-manifest alignment without fitting. It does not
promote legacy baseline inference provenance. `comparison_ready` stays false.
The [evaluation protocol](EVALUATION_PROTOCOL.md) still requires verified
content-level alignment and explicit evidence/model provenance for main paired
comparisons. Historical caches must not quietly become fully verified baselines.

This step computes no metrics or probabilities, selects no thresholds, makes no
judge calls, and reads no HaluBench data. A successful token audit supplies lengths
for a later bounded execution manifest; it is not scoring authorization. Model,
prompt, calibration, thresholds and benchmark membership remain unchanged.

Cluster update: [the completed audit and replay](../../results/post_thesis/llm_judge/test_token_audit_cluster_20260920.md) match these lengths and the independently reconstructed full audit hash. Preserve this historical audit after checkout updates. Proceed with the [pre-inference evaluation numerical checks](EVALUATION_MATH.md); no inference allowance is implied.
