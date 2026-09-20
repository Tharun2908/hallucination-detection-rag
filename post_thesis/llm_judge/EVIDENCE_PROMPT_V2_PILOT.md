# Post-thesis: original TRAIN pilot token audit for evidence prompt v2

**Tokenization only; adaptive development, excluded from the submitted thesis.**
The [paired synthetic findings](../../results/post_thesis/llm_judge/evidence_prompt_pair_v1_20260919.md)
show one corrected verdict, one v2 rationale regression and a shared false-absence
miss. Hold v2 fixed now, rather than tuning further on the 26 synthetic cases.
The next question is whether its extraction and reasoning behavior transfers to
the longer original TRAIN inputs. This step measures input length only.

## Unchanged inputs and fixed candidate

Reuse the existing private manifest:
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-v1/manifest.json`.
Its required SHA256 is
`ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
Do not prepare a new sample or rewrite the manifest. The same 50 answers,
contexts, order and offline labels remain intact. No parquet downloads, TRAIN
resampling, TEST or HaluBench reads are needed.

| Component | Fixed value |
| --- | --- |
| Prompt | `faithfulness-evidence-diagnostic-v2` |
| Prompt SHA256 | `7b43e40be9758a9eb2e210a3b4636c043eca88578b9fa62c2774ac7df5414e82` |
| Schema | v3, verdict first |
| Serialized schema SHA256 | `f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33` |
| Request contract | `evidence-diagnostic-request-v3` |
| Serving profile | `evidence-v1`, pinned Qwen3-32B and tokenizer |
| Output allowance for length check | 512 tokens per input |
| Model context limit | 32,768 tokens |

The new fixed audit directory is
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-token-audit-evidence-prompt-v2/`.
The earlier prompt-v1 evidence audit stays in its original `...-evidence-v3/`
directory. Prompt version and schema version identify different components.
The CLI rejects reuse of old or custom audit names for the new evidence prompt.
Its audit identity retains the exact schema string, order, prompt, profile,
code revision and per-input request keys for later scoring alignment.

## Limits and interpretation

- At most 50 tokenization attempts, one per original input, sequentially.
- 60-second HTTP operation timeout; 300-second client window per invocation.
- Only preflight and `/tokenize` are called; no generation endpoint is used.
- Only the fixed prompt and exact answer/context reach tokenization. Labels,
  queries, IDs and task/generator metadata remain offline.
- Fit requires formatted input tokens + 512 <= 32,768. No truncation, replacement
  or filtering. Report overlength inputs unchanged.
- Completed token counts are reused. Fully cached replay makes no HTTP calls.
  Failed/interrupted attempts block further tokenization for that audit; do not
  delete or rename artifacts to retry them.
- Keep `audit.json` and all `client-window-*.json` records. Missing checkpoints
  with existing evidence-audit windows fail closed. Each invocation records its
  own client resource window; server startup/idle and rental cost are separate.

The output always states `generation_calls: 0` and
`scoring_authorized_by_this_audit: false`. It establishes input fit, not faithfulness,
explanation quality, generation usage or a spending estimate. Once actual lengths,
audit SHA256 and code revision are reported, a separate scoring configuration can
be recorded. The existing v1 evidence scoring command remains pinned to v1; it
cannot score this v2 candidate or reuse the new audit.

## H200 procedure after commit, push and CI

Stop the previous launcher with Ctrl+C, then start the same evidence profile from
the newly committed code in terminal A. Dependencies and model cache stay unchanged.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0 --profile evidence-v1
```

After `Application startup complete`, run in terminal B:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_pilot \
  --prompt-version faithfulness-evidence-diagnostic-v2 \
  --expected-manifest-sha256 ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25
```

Return the full console summary, including prompt/schema hashes, token counts,
audit checksum and code revision. Finalize terminal A's server resource record
when finished. No new benchmark result, threshold or final prompt freeze is
introduced. The known v2 rationale regression remains part of the next semantic
review, and the original 50 remain adaptive TRAIN development data.
