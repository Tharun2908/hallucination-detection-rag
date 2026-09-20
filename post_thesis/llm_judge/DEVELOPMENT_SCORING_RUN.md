# Post-thesis bounded scoring of reserved TRAIN development arms

This run collects raw class-token scores for the frozen development reservation.
It does **not** fit calibration, choose a threshold, tune a prompt, evaluate TEST,
or score HaluBench. The [calibration/threshold rules](CALIBRATION_THRESHOLD_PROTOCOL.md)
remain unchanged. These are development examples, not thesis or held-out results.

## Frozen execution and identities

The [completed token audit](../../results/post_thesis/llm_judge/development_token_audit_cluster_20260920.md)
has SHA256 `ac101084ab0fa2f76ceaec118eef00ec52c877f5e7dbcc9ee917de1297390fbf`
at revision `9182e27bfdc52b81a1b26f05753bbff5f9a486af`. Both its fresh and replay
hashes agree and were independently reconstructed. Keep this audit, the completed
reservation and all earlier results intact after updating code.

[development_label_scoring_v1.json](configs/development_label_scoring_v1.json)
records two arm plans. Its canonical configuration SHA256 is
`4aa67f2f0df10d0e6fa7994e11cf5e01a745c06c87e90daf070033b6a36f08de`.
Each arm pins a hash of its complete ordered request-reference list (slot, request
fingerprint, input-token count). These references were derived locally from the
completed manifest, reconstructed cluster audit and actual pinned tokenizer.
Full request bodies remain private, while the public hashes bind exact membership,
order, rendered-input hashes, prompt, model/profile, mapping and wire payload.

| Bound | Calibration arm | Operating-threshold arm |
| --- | ---: | ---: |
| Reserved responses / maximum attempts | 600 / 600 | 600 / 600 |
| Total audited input tokens | 705,198 | 746,401 |
| Maximum requested output tokens | 600 | 600 |
| Output tokens per request | 1 | 1 |
| Attempts per input / automatic retries | 1 / 0 | 1 / 0 |
| Concurrent requests | 1 | 1 |
| Request timeout | 60 seconds | 60 seconds |
| Cumulative client-time limit across invocations | 1,800 seconds | 1,800 seconds |

The two arms therefore permit at most 1,200 scoring attempts and 3,600 cumulative
client seconds when executed sequentially. Each has its own ledger and namespace;
unused budget does not transfer. Thirty minutes per arm is a conservative run cap,
not a measured latency estimate or GPU rental budget. CPU request preparation is
recorded separately; server startup/idle and whole allocation time remain in the
launcher record. Currency cost is unknown until an hourly rate is supplied.
Do not run two arm clients simultaneously: execute them sequentially as below.

Use the unchanged Qwen3-32B BF16/non-thinking model revision, raw-logprob
`label-score-v1` profile, primary A=supported/B=unsupported mapping and original
source fingerprints. Before any HTTP, verify all historical hashes, rerender each
selected input and compare it to the audit, then match the arm's frozen request
list hash. The model receives only token IDs rendered from fixed prompt plus
answer/context. IDs, arm names, labels, tasks and generator metadata are local
bookkeeping, never model input. There is no label mask, schema, logit bias,
truncation, alternate prompt or probability fallback.

## Durability, failures and accounting

Use the existing journal, arm process lock and atomic writes. A shared execution
lock also prevents simultaneous clients across the two arm namespaces. Reserve budget before
preflight; journal each attempt before sending. Save raw response, raw class log
probabilities, margin, uncalibrated relative score, token mass, emitted token,
usage and per-attempt latency. The original strict parser validates model identity,
prompt-token alignment, output position and usage before accepting a score.
Observed malformed/failed response usage is still counted; missing usage remains
unknown rather than zero. No failure is converted to a negative or 0.5 score.

Any request failure or interruption stops that arm. No automatic retry or run-ID
override is provided. An unfinished execution window charges its full reserved
remaining budget on recovery. A deliberate `--max-new-attempts N` may stop after
N completed requests; a later invocation can process only unattempted inputs
within the same budget if no error halted it. Do not delete the journal to restart.
Changed code, tokenizer, requests or budgets refuse cache reuse. Preserve a halted
run for review; no fitting on a convenient subset is permitted by the protocol.

Completed-arm replay and `--inspect` need no server, tokenizer or HTTP calls.
Summary JSON is derived from immutable attempts and includes invocation-specific
`new_attempts`, so its hash can change between a fresh run and cache inspection.
This does not change scores, request membership or charged history. Save both
reported summary hashes when returning results.

## Launch after commit, push and successful CI

The server and client must record the same clean commit. Stop the old launcher
with Ctrl+C in the terminal running `post_thesis.llm_judge.serve`, and wait for it
to exit and record shutdown. Ctrl+C in `tail -f` stops only the log viewer.
Then launch once with the scoring profile and reuse it for both sequential arms:

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

Weights are reused from the existing root cache. Wait for `Application startup
complete`. In a second terminal, enter the full new `server-...` directory name:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Full current session directory name (server-...): " judge_session &&
python -m post_thesis.llm_judge.run_development_scores --arm calibration --server-session-id "$judge_session" &&
python -m post_thesis.llm_judge.run_development_scores --arm operating_threshold --server-session-id "$judge_session"
```

These commands perform the bounded inference. The second starts only if the first
finishes with all 600 valid scores. Any failure leaves its raw records available;
return the error rather than adding retries or changing settings. No packages or
model downloads are part of this step. Do not rerun token audits or reservation
commands under the new revision.

Then inspect both caches without any HTTP:

```bash
python -S -m post_thesis.llm_judge.run_development_scores --arm calibration --inspect
python -S -m post_thesis.llm_judge.run_development_scores --arm operating_threshold --inspect
```

Private run directories under `.artifacts/post_thesis/llm_judge/` are
`qwen3-ragtruth-development-calibration-label-v1` and
`qwen3-ragtruth-development-operating-threshold-label-v1`. Each stores
`prepared.json`, `budget.json`, `journal.sqlite3` and `summary.json`. They include
input token IDs and responses, so keep them private. Return the concise console
summaries and fresh/inspection hashes; no 1,200-row paste is needed yet.

Stop the launcher when finished to record server time. Score collection is the
only new execution stage here. Subsequent offline fitting must follow the frozen
statistical protocol; aggregate benchmark claims and any cascade remain later
work. Existing exact-overlap limitations and the post-thesis separation remain.

## Validation

The pinned tokenizer reproduced 600 ordered requests per arm and exactly the
cluster's audited token totals. Fourteen focused tests use offline HTTP responses
to check separate caches/budgets, zero-call replay, partial runs, interruption,
unknown usage/windows, preflight and server identity, no retries, payload drift,
missing journals, frozen request plans and label-free preparation. The Windows
workflow runs these tests with the client dependency installed; Linux's complete
CPU suite discovers them as well. No live model calls were made during validation.
