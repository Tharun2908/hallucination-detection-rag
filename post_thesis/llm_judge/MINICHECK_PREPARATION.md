# Post-thesis MiniCheck-7B preparation

The [fresh fusion application and replay](../../results/post_thesis/llm_judge/fresh_fusion_cluster_20260928.md)
completed with 2,700 scores and zero changes at the frozen operating threshold.
MiniCheck is the remaining fresh comparison baseline. The judge, calibrator,
fusion and all TRAIN-selected operating thresholds remain fixed.

## Source-derived contract

The repository pins MiniCheck to commit
`b58b9fa69acbd1015ec970fa65dd752413a053d2`. The relevant official sources are
[the wrapper](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/minicheck/minicheck.py),
[LLMCheck](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/minicheck/inference.py),
and [prompts](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/minicheck/utils.py).
Reviewing these files does not prove which upstream/library version generated
the historical cache.

The fresh model candidate is `bespokelabs/Bespoke-MiniCheck-7B`, revision
`1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8`. Its official
[configuration](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B/blob/1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8/config.json)
declares InternLM2ForCausalLM. Four BF16 safetensors shards total 15,475,541,232
bytes (about 14.41 GiB); tokenizer/configuration and KV-cache memory are additional.
No model weights were downloaded during this preparation.

Preserve these upstream behaviors when building the fresh runner:

1. Input is document=context and claim=answer; no query, benchmark label, task
   or generator metadata enters the model.
2. Use NLTK sentence splitting on the claim. Document splitting preserves newline
   blocks, then groups sentences into chunks using tokenizer counts. The default
   limit is 32,768 and document chunk budget is 32,468. The implementation joins
   sentences and strips chunk edges, so exact effective chunk text must be saved
   or hashed. Original-document hashes alone are not effective-input provenance.
3. Render the pinned system/user prompt with the model's chat template for every
   document-chunk/answer-sentence pair. The 300-token reserve is not a guarantee
   that the final rendered prompt fits: audit exact lengths including the claim.
4. Generate one token, temperature zero, seed 2024, prefix caching disabled.
   The upstream requests top-five logprobs, plus any sampled-token behavior of
   its vLLM runtime. Inspect the first generated position.
5. Sum exp(logprob) for returned tokens whose decoded text lowercases to `yes`,
   without whitespace stripping. This is returned vocabulary mass, **not** the
   judge's normalized two-class score. In a valid response with no matching token
   the upstream returns zero. Missing transport output is a failure, not zero.
6. For each answer sentence take its maximum support over document chunks; the
   whole-answer score is the minimum of those maxima. An empty answer has no
   defined minimum and must not silently receive a fabricated support score.
7. The upstream convenience label uses support > 0.5. Benchmark decisions must
   instead retain the established TRAIN threshold: support strictly less than
   `0.20000000000000004` indicates unsupported. Preserve the comparator and exact
   float; do not retune from fresh TEST scores.

`minicheck_contract.py` encodes and tests the score reduction and strict threshold.
It is not yet a live inference adapter. NLTK version/resources, tokenizer/template
bytes, model shard hashes and current vLLM logprob behavior still need to be bound
to the fresh execution profile. Historical rounding to four decimals must be
handled explicitly when comparing to legacy output; preserve raw new scores.

## Check the existing pod before loading

The old repository requirements include vLLM 0.4.3 and an older torch stack.
Do not install them over the working Qwen environment. The current vLLM may
support InternLM2 natively, but a registry name alone does not verify live
compatibility. Qwen and MiniCheck also share GPU memory if both are loaded.

After committing/pushing this patch:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.check_minicheck_environment
```

This reads installed package versions, the vLLM registry source, nvidia-smi
memory totals, disk space and the filenames of any already-cached pinned model.
It imports no torch/vLLM model code, allocates no GPU memory, downloads nothing,
installs nothing and stops no server. It writes a private environment record
under the post-thesis artifact root. Changes in available memory naturally
produce a different inventory hash.

Send the printed report. Use it to choose the exact tokenizer/resource setup
and whether the Qwen server must be stopped for MiniCheck. Then verify the
fixed scoring path on synthetic inputs before the bounded TEST run. No additional
judge/fusion development or benchmark threshold selection is intended.
