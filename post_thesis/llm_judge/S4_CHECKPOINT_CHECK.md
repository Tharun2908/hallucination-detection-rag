# Post-thesis S4 checkpoint compatibility

This check supports the post-thesis comparison and is not a submitted thesis
result. It uses the current final S4 checkpoint, not an OOF fold checkpoint.

The operator's saved final checkpoint has four files and records Transformers
5.5.4 in its config. Fold 1 has seven files, records Transformers 4.44.0 and has
a different tokenizer fingerprint. A config's version field is a serialization
observation, not proof of the complete original training environment. Do not
copy fold tokenizer files into the final checkpoint or modify either config.

## Run on the existing pod

Use the existing serving environment (Python 3.12, Transformers 5.17.0,
tokenizers 0.23.2, PyTorch 2.13.0+cu130). No package installation is needed. This
is a CPU compatibility probe of that runtime, not a claim of historical runtime
equivalence. It does not call the running vLLM server.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.check_s4_checkpoint --checkpoint /workspace/signal4_model
```

The command verifies all four recorded file sizes and SHA256 hashes before
loading anything. It uses local files only, disables Hub access, hides CUDA in
its own process, and loads safetensors in FP32 on CPU. Missing, unexpected or
mismatched model keys are errors; no replacement classification head is allowed.
It needs enough host RAM for the approximately 738 MB checkpoint and activations.

Four artificial pairs exercise short supported/contradicted text, a long answer
and a long context. Verdict accuracy is not a compatibility acceptance criterion.
The check requires finite two-class logits, confirms class 1 means HALLUCINATED,
and verifies that the loaded tokenizer matches the saved tokenizer's untruncated
pair encodings. The embedded vocabulary may be smaller than the model embedding
table; equality with the saved tokenizer and valid model token indices are checked.

The forward pass uses answer first, context second, `max_length=512`,
`truncation=True`, `padding="max_length"`, and only `input_ids` and
`attention_mask`, matching the original S4 script's actual arguments. Here
`truncation=True` means **longest-first pair truncation**, which can remove answer
tokens as well as context tokens. The old script's comment about keeping the
full answer does not describe that behavior. The report records token counts
before and after truncation for each side.

The private report is
`.artifacts/post_thesis/llm_judge/s4-checkpoint-compatibility-v1/report.json`.
Replaying the identical successful run checks fingerprints and reuses that
report without loading the model or running further forward passes. A failed
probe does not write a success report. Preserve its console error for diagnosis.

## What success establishes

Success establishes that these current saved files load and produce finite
synthetic CPU scores in the recorded runtime. It does not establish GPU runtime
compatibility, historical cache provenance, training exclusions, or benchmark
quality. `comparison_ready` stays false. Next, use the identified compatible
checkpoint/tokenizer for fresh post-thesis inference with per-example input and
effective-token provenance, preserving the legacy cache and frozen thresholds.

Reference: [Transformers local loading and loading diagnostics](https://huggingface.co/docs/transformers/main_classes/model).
