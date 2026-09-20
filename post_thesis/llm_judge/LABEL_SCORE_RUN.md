# Post-thesis: bounded synthetic label-score compatibility run

This experiment is excluded from submitted thesis results. It implements step 2
of the [continuous-score protocol](CONTINUOUS_SCORE_PROTOCOL.md). It does not
evaluate RAGTruth or HaluBench, fit thresholds or calibrators, or reopen evidence
prompt tuning. The [setup observations](../../results/post_thesis/llm_judge/label_score_setup_20260920.md)
are complete; **live score compatibility has not yet been established**.

## Fixed experiment

`configs/label_score_synthetic_v1.json` freezes all 30 request fingerprints,
prompt hashes, tokenizer-derived input counts, the serving profile and limits.
The exact requests are regenerated and checked before HTTP. They contain only
the fixed prompt plus each synthetic answer/context; case IDs and expectations
stay offline. `evidence_cases.py` supplies the existing 14 attribute, numeric and
absence controls, with short and embedded answers.

| Requests | Purpose |
| --- | --- |
| 14 primary | A=supported, B=unsupported; unchanged label-score-v1 prompt |
| 14 swapped | A=unsupported, B=supported; separately versioned diagnostic prompt |
| 1 repeat | Fresh primary supported-short request; numerical repeatability |
| 1 temperature control | Same primary supported-short input at temperature 1 instead of 0 |

The primary mapping remains fixed. Do not choose or average mappings after seeing
results. The two intentional repeated inputs are separate, predeclared request
slots, each with one attempt. A later replay uses the journal without new calls.

The actual pinned tokenizer rendered 18,786 input tokens in total, at most 682
per request. The client rechecks the tokenizer reference and every request hash.
Each request permits **one output token**, sequential execution, a 60-second
request timeout and **600 cumulative client seconds across resumptions**. There
are no automatic retries. Every input must fit, with no truncation. Preflight
time is included in the cumulative budget; local preparation time is recorded
separately. Model startup and idle time belong to the server session record.

## Transport and interpretation

Use `/v1/completions` with the complete locally rendered prompt **as token IDs**.
This avoids applying a second chat template. Request `logprobs=2`,
`logprob_token_ids=[32,33]`, `return_tokens_as_token_ids=true` and
`return_token_ids=true`. No grammar, allowed-token mask or logit bias is used.
The named serving profile explicitly sets `--logprobs-mode raw_logprobs`.
`ignore_eos=true` retains exactly one generated position even if it is EOS;
an off-label first token is recorded and does not by itself invalidate both
available class scores. Never search later output positions for A or B.

The parser requires the complete returned prompt token sequence to match, one
generated token, both selected log probabilities, consistent sampled-token
probability, and exact prompt/output usage. It checks model identity, duplicate
JSON keys, finite nonpositive values and probability mass. vLLM's reviewed
formatter clamps at -9999, so any returned value at or below that boundary is
rejected as censored. Failures retain missing scores and known or unknown usage;
they are not mapped to 0, 0.5, a greedy verdict or another extraction method.

Source references at vLLM 0.29.0:

- [Raw-mode configuration](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/model.py)
- [Sampler and explicit candidate gathering](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/sample/sampler.py)
- [Request conversion](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/entrypoints/openai/completion/protocol.py)
- [Response formatting and clamping](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/entrypoints/openai/completion/serving.py)

Both launcher and client require the installed files to match the reviewed Git
blob fingerprints. The new profile records the explicit raw-mode command in a
new serving session; original serving profiles retain their previous behavior.

Transport passes only if all 30 scores validate and both numerical controls
match. Compare each A/B raw log probability with absolute tolerance `1e-4` and
relative tolerance `1e-5`; the greedy repeat must also emit the same token.
The temperature control may emit a different token. These controls and source
checks provide transport evidence; they do not prove semantic correctness or
calibration. Semantic errors remain reportable results, not transport failures.
Numerical-control failure blocks escalation and must be reported without tuning
tolerances after the run. Raw values, margins, relative scores, class-token mass
and emitted token IDs are retained for review.

## Restart with the explicit profile

After committing, pushing and checking CI, stop the old server with Ctrl+C in
the terminal running `post_thesis.llm_judge.serve`. Ctrl+C in `tail -f` stops only
the log viewer. Let the launcher finish recording shutdown, then run:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" &&
export HF_HOME=/root/llm-judge-hf-cache &&
python -m post_thesis.llm_judge.serve --device 0 --profile label-score-v1
```

The existing model cache is reused. Wait for `Application startup complete` in
the new session's log. Its served model ID is now
`qwen3-32b-label-score-9216db5781bf`. Use the full new session directory name,
including `server-`, for the client. The launcher and client must use the same
clean committed checkout and the restored CUDA-13 PyTorch environment.

In a second terminal:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "New full session directory name (include server-): " judge_session &&
python -m post_thesis.llm_judge.label_diagnose --server-session-id "$judge_session"
```

This is the command that makes the bounded generation calls. The client uses the
existing small tokenizer cache and downloads nothing. Preserve any failure.
Do not rerun earlier smoke tests, TRAIN pilots or evidence diagnostics.

## Replay, resumptions and records

For inspection without HTTP, a tokenizer or a running server:

```bash
python -S -m post_thesis.llm_judge.label_diagnose --inspect
```

A planned partial invocation can use `--max-new-attempts 5`; rerunning the normal
command resumes only unattempted slots within the same total budget. Any request
failure stops further generation. Interrupted requests are terminal with unknown
usage. An execution window left open by process death consumes its entire
reserved remaining budget, so restarting cannot grant fresh compute. A changed
identity, corrupted journal or missing companion artifact fails closed.

Private records live under
`.artifacts/post_thesis/llm_judge/qwen3-label-score-synthetic-v1/`:
`prepared.json`, `budget.json`, `journal.sqlite3` and derived `summary.json`.
The journal saves each request reservation before sending and records complete
decoded response objects where available. HTTP failures or malformed JSON may
have no decodable response; their usage remains unknown. Cache keys include the
slot, mapping, prompt, complete input tokens, model revision and inference
profile; run identity additionally binds source and tokenizer versions/files.

Return the console summary and report SHA256. Review semantic behavior separately
from transport status. Even a passing result does not authorize a TRAIN or test
run; those need the next committed input audit and execution plan.
