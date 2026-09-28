# Post-thesis HaluBench CPU token audit

The [canonical 8k manifest](HALUBENCH_INPUTS.md) was created on the pod at
`aac2d9e0ce9ef4ac98ff9a60e26238daa08a46f3`. Its exact manifest SHA256 is:

`0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded`.

The same hash was independently reconstructed locally from the pinned parquet,
split and legacy cache files. The new command requires this manifest, the accepted
private RAGTruth fit and the unchanged frozen judge configuration.

## What is counted

Each original answer/context pair is rendered using the frozen
`faithfulness-label-score-v1` prompt and pinned Qwen3-32B tokenizer/template.
The A/B class-token mapping, non-thinking assistant boundary and model limit
remain unchanged. The check reserves one output token within 32,768 total tokens.
It does not truncate inputs, shorten the prompt or remove overlength examples.
Labels, questions and source names remain offline; sample IDs identify audit
records but are not included in the prompt.

Tokenization uses only cached tokenizer/configuration files. The loader disables
Torch/TF/Flax backends; no model weights, vLLM client or network inference client
is created. No tokenizer download is enabled by this command.

Results contain hashes and lengths, not rendered prompts. They are checkpointed
every 25 inputs and on handled interruption. A completed cached replay performs
zero new input tokenizations and leaves the audit file unchanged. It still checks
the tokenizer's four synthetic reference cases before accepting cached results.

All overlength IDs are reported; the command exits nonzero if any input exceeds
the limit. Such a result requires an explicit protocol decision before inference,
not silent truncation or row removal. This audit grants no scoring allowance and
does not promote legacy baseline comparison readiness.

## Local CPU reference

The [local pinned-tokenizer run](../../results/post_thesis/llm_judge/halubench_token_audit_local_20260928.json)
counted all 8,000 inputs: **7,794,485 input tokens**, minimum **506**, maximum
**7,298**, with no overlength inputs. Cached replay returned identical results
and zero new input tokenizations. These are reference counts, not an attestation
of the pending pod audit hash or runtime environment.

## Run on the pod

Use the serving environment for the already pinned tokenizer packages; no running
GPU server is needed. The earlier manifest command used the fitting environment
only for parquet preparation.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_halubench_tokens
```

Repeat the final command at the same clean revision to verify cached replay:

```bash
python -m post_thesis.llm_judge.audit_halubench_tokens
```

The audit is saved under:

```
.artifacts/post_thesis/llm_judge/halubench-test-token-audit-label-score-v1/audit.json
```

The canonical 8,000-example membership and registered group map stay fixed.
The known exact-passage overlap with the unused adaptation pool remains disclosed.
Fresh baseline input/checkpoint linkage and a bounded judge execution plan are
subsequent work. No fitting, thresholds, benchmark performance metrics or model
calls are added by this step.
