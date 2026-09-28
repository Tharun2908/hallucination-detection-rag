# Post-thesis MiniCheck synthetic compatibility check

The operator recreated the H200 pod. The [new environment record](../../results/post_thesis/llm_judge/minicheck_environment_20260928.json)
shows an idle GPU, the pinned model snapshot present under `/root`, and ample
root-disk space. Snapshot filenames alone did not verify the files' contents.
This step checks those files and makes six synthetic generation requests.
It does not read RAGTruth, HaluBench, labels or benchmark prediction caches.

## Tokenizer compatibility finding and explicit adapter

Local checks with Transformers 5.17.0, tokenizers 0.23.2 and sentencepiece 0.2.2
found a concrete mismatch in the model's older custom tokenizer loading path:

| Text | Custom AutoTokenizer path | Saved tokenizer.json |
| --- | --- | --- |
| `Hello world.` | `[1,318,261,270,270,268,293,268,267,270,273,281]` | `[1,9843,2028,281]` |

The current slow-loading request also produced the first sequence locally.
Direct SentencePiece 0.2.2 loading of the model failed on a null-character
vocabulary validation. In a separate local reference installation,
SentencePiece 0.2.0 matched the saved fast tokenizer on four ordinary Unicode,
numeric and newline probes, with BOS prepended. This is evidence of a runtime
compatibility problem; it does not identify the exact historical cache runtime.

The fresh adapter loads the pinned serialized fast tokenizer directly with
`PreTrainedTokenizerFast`, preserving its normalizer, vocabulary, merges, added
tokens and postprocessor. It sets the saved special tokens and chat template,
then saves a separate standard tokenizer directory. It does not modify the
original snapshot or install/downgrade packages on the pod. The serialized file
also retains an incidental 869-token truncation setting. The adapter explicitly
disables truncation/padding, matching ordinary untruncated HF encoding, and
checks the complete resulting backend for equality after reload. Seven probe
strings (including a rendered chat and a long input beyond 869 tokens) are
checked with and without special tokens: fourteen exact encoding comparisons.

The template itself includes BOS and ordinary string encoding adds another BOS.
Preserve this upstream string-generation path; do not silently remove a token.
Before generation, the live vLLM tokenizer must reproduce every prepared prompt
token ID. Each returned response must also expose that exact prompt token list.
The pinned config's custom Python is used for configuration loading only;
`model_impl='vllm'` requires native model execution, and the explicit tokenizer
adapter avoids the old custom tokenizer code. All files needed by this path
are checked before loading. No network model/code downloads are enabled.

## Fixed synthetic workload

Four artificial examples cover single-sentence support/contradiction and
two-sentence complete/partial support. With the upstream answer sentence split,
they produce exactly six document-chunk/answer-sentence prompts. The fixed
NLTK 3.10.3 runtime must find its English `punkt_tab` resources locally; the four
resource files are fingerprinted into the private run identity. Resource
availability and actual NLTK splitting remain to be checked on the pod.

The workload uses the pinned upstream prompt, chunk/newline transformation,
first-position top-five logprob support mass and minimum-of-per-sentence-maxima
aggregation. Semantic expectation matches are reported separately from transport
validity. A synthetic semantic miss must be preserved, not repaired using labels
or automatic prompt tuning.

Engine: vLLM 0.29.0, torch 2.13.0+cu130, H200, BF16, native InternLM2, seed 2024,
32,768 model limit, prefix caching disabled, eager execution, FlashAttention,
16 maximum sequences, 4,096 batched tokens, raw logprobs and 35% GPU-memory
utilization. At least 64 GiB must be free before loading. Sampling uses one
generated token, temperature zero and the upstream stop-token construction.
These are explicit fresh-run settings; historical kernel/runtime parity is not
asserted. Model initialization includes vLLM's own profiling calls, separate from
the six requested generations.

## Run on the new pod

After committing/pushing and pulling the patch:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export NLTK_DATA=/root/nltk_data &&
export HF_HOME=/root/llm-judge-hf-cache &&
timeout --signal=TERM --kill-after=60s 20m \
  python -m post_thesis.llm_judge.check_minicheck_live
```

The command reads and hashes roughly 15.5 GB of cached weight data before model
loading; it does not redownload those weights. Keep the 20-minute external wall
limit. Output directories are created automatically, including the earlier
environment-check directory (fixed in this patch).

On success, run the same bounded command again. It revalidates local inputs and
file identities but returns the preserved result with `New model attempt: False`,
no new model loading/generation and the same report hash. Reading the weight
hashes again still takes disk I/O. Send both printed summaries.

Missing dependencies/resources or file mismatches stop before inference. The
single initialization/generation attempt is durably reserved before entering the
runtime. A recorded failure or interruption is never automatically retried.
Do not delete the journal or change the run identity to force another attempt;
retain the error for a reviewed compatibility fix.

Private records: `.artifacts/post_thesis/llm_judge/minicheck-synthetic-compatibility-v1/`.
The report retains synthetic prompts, token IDs, returned logprobs, support
matrices, tokenizer/resource identities and engine settings. No fresh TEST
MiniCheck execution is added by this check. Next, bind the verified input path
to the canonical TEST manifest and run the baseline with bounded accounting.
