# Post-thesis: offline label-score contract

This work is excluded from submitted thesis results. It implements the offline
portion of [the continuous-score design](CONTINUOUS_SCORE_PROTOCOL.md).
Evidence-prompt tuning remains paused. No model generation, benchmark evaluation,
threshold fitting or calibration is performed by these commands.

## Contract and scope

`label_score_v1.json` records the resolved, versioned prompt and its fingerprint,
the pinned Qwen3-32B model/tokenizer revision, the class mapping and extraction
requirements. Only answer and context enter the user message. The fixed system
prompt asks for `A` (supported) or `B` (unsupported).

`label_score_contract.py` computes the following from two supplied raw,
full-vocabulary log probabilities at the same first assistant position:

| Field | Meaning |
| --- | --- |
| `supported_logprob` | Raw log probability of A |
| `unsupported_logprob` | Raw log probability of B |
| `unsupported_log_odds` | B log probability minus A log probability |
| `unsupported_score` | Stable sigmoid of that difference |
| `log_class_token_mass` | Stable log of P(A) + P(B) |
| `calibrated` | Always false for this contract |

This is a relative class-token score, not an established calibrated probability.
Keep the log-odds for ranking when sigmoid values round to zero or one. Missing,
duplicate, wrong-token, nonfinite or impossible probability inputs fail; they
do not receive substitute scores. The arithmetic function cannot establish that
numbers actually came from the required model, position or raw-logprob mode.
A future transport adapter must verify that provenance.

There is no HTTP scoring adapter, live execution plan or generation allowance in
this step. No evidence JSON grammar or class-only token mask is applied.

## Recorded CPU reference

The proposed implementation was checked with the actual tokenizer at revision
`9216db5781bf21249d130ec9da846c4624c16137`, without loading model weights or
importing PyTorch. All four synthetic input boundary checks passed: A is token
`32`, B is token `33`. Both remain a single token after the complete non-thinking
assistant prefix, without merging across that boundary.

`label_tokenizer_reference_v1.json` records tokenizer-file fingerprints, exact
template and rendered-input fingerprints, token-sequence fingerprints, class
bytes/IDs, input lengths and dependency versions. This is an assistant CPU
reference from implementation development, not a cluster or vLLM result and not
a measurement of judge accuracy. Unit-test toy tokenizers are separate from this
real-tokenizer reference.

The cluster check requires `transformers==5.17.0` and `tokenizers==0.23.2`, already
present in the restored serving environment. Hugging Face Hub, Jinja2 and Python
versions are recorded; different rendered tokens or file fingerprints fail the
reference comparison. Do not silently upgrade dependencies to bypass a failure.

## Run on the restored pod after committing and pushing

No serving process, CUDA compiler or model-weight cache is needed. Preserve the
existing virtual environment and private experiment records.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -S -m post_thesis.llm_judge.check_label_score &&
python -m post_thesis.llm_judge.check_label_tokenizer --download-tokenizer
```

The first command uses only the Python standard library. The tokenizer command
requires a clean tracked checkout and downloads only allowlisted tokenizer and
configuration files at the exact revision. It does not download model weights.
The small cache is retained under
`.artifacts/post_thesis/llm_judge/label-tokenizer-cache-v1/`, on the persistent
workspace. Each check creates a new private `label-tokenizer-v1-.../report.json`
with the code revision, versions, prepared-input identities and report hash.
Completed experiment directories are untouched.

After that initial download, repeat entirely from the cache with:

```bash
python -m post_thesis.llm_judge.check_label_tokenizer
```

Return the console output, including the report path and SHA256. A successful
check reports four tokenization checks, A/B token IDs 32/33, zero generation
calls and unverified live-logprob compatibility. If a check fails, retain its
failure record; do not substitute another tokenizer or guess token IDs.

The prepared key describes offline input preparation only. It is not a live
request cache key or permission to score. Transport, server identity, raw-logprob
semantics and complete inference configuration remain to be incorporated in a
separate bounded compatibility experiment. Development/test separation and the
pending native-source overlap audit remain as specified in the protocol.

## Validation

The CPU unit tests cover stable score arithmetic, invalid/missing class inputs,
metadata exclusion, token-boundary failures, file/reference mismatches and
allowlisted loading. Existing Linux and Windows CI discover them automatically.
The real tokenizer check separately compares chat-template tokenization with
explicit encoding using `add_special_tokens=False`, and verifies each continued
prompt against the complete prompt token sequence plus exactly one class token.

API references: [Transformers chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
and [Hub download filtering](https://huggingface.co/docs/huggingface_hub/guides/download).

## Subsequent step

The [cluster setup observations](../../results/post_thesis/llm_judge/label_score_setup_20260920.md)
now record matching tokenizer/source checks. This original offline descriptor
remains frozen; the separate [live execution plan](LABEL_SCORE_RUN.md) owns the
new bounded synthetic calls. Its results remain pending.
