# Post-thesis: bounded original-50 TRAIN label-score pilot

This experiment is excluded from the submitted thesis. It uses the unchanged
50 TRAIN development examples already inspected during prompt development.
It is not held-out evaluation and cannot provide a final benchmark claim.
Evidence-prompt tuning remains paused; primary A=supported / B=unsupported and
`faithfulness-label-score-v1` remain fixed. Swapped labels are not run or averaged.

The [cluster audit](../../results/post_thesis/llm_judge/label_score_pilot_audit_20260920.json)
completed without generation. This plan pins its SHA256
`b48d1eca31d68032fff55c38f2f82a038138880c72dad29d6f4dec38ebcf51bc`
and code revision `c964dc8af892c3f5344c5a3b43c91873460de884`.
Preserve that audit; do not rerun it after updating the code, since it intentionally
binds the historical audit revision. The private audit and original manifest are
validated and all 50 tokenized inputs are compared again before any HTTP.

## Frozen execution

`configs/ragtruth_pilot_50_label_score_v1.json` freezes 50 request fingerprints,
the primary prompt, Qwen3-32B BF16 revision, raw-logprob serving profile and bounds.
Request hashes were prepared locally from the checksum-verified TRAIN parquet,
which reproduced the original pilot manifest exactly, and the pinned tokenizer.
The cluster's private audit has not been read by the assistant; the cluster runner
must validate its supplied checksum before generation.

| Bound | Value |
| --- | ---: |
| Original TRAIN inputs / maximum attempts | 50 / 50 |
| Attempts per input | 1 |
| Input tokens, total | 60,574 |
| Input tokens, minimum / maximum | 721 / 2,986 |
| Output tokens per input / total | 1 / 50 |
| Request timeout | 60 seconds |
| Cumulative client budget across invocations | 600 seconds |
| Concurrent requests | 1 |
| Automatic retries | 0 |

Only fixed prompt plus answer/context become model input. Sample IDs are private
bookkeeping; labels, task and generator metadata are not supplied. Complete local
prompt token IDs are sent to `/v1/completions`, without a second chat template,
truncation, label mask, schema, logit bias or score fallback. The extraction and
parser are the same ones checked in the [synthetic run](LABEL_SCORE_RUN.md).
Raw A/B log probabilities, log odds, relative score, class-token log mass, emitted
token ID, off-label status, complete decoded response, usage and per-attempt
latency are retained. Scores remain uncalibrated, including those near 0 or 1.

Each attempt is journaled before sending. Any failed response or interrupted
request halts generation; missing scores stay missing. An unfinished execution
window consumes its entire reserved remaining budget. Preflight counts toward
client time; CPU preparation is recorded separately. Startup and idle time belong
to server records, and currency cost stays unknown without an hourly rate.

## Run after commit, push and CI

The launcher and client require the same clean committed revision, so restart the
label-score server once after pulling this change. Ctrl+C must reach the terminal
running `post_thesis.llm_judge.serve`; Ctrl+C in `tail -f` only stops the log viewer.
Wait for the launcher to exit and record shutdown, then:

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

Existing weights are reused. Wait for `Application startup complete`. In another
terminal, use the new full session directory name, including `server-`:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "New full session directory name (include server-): " judge_session &&
python -m post_thesis.llm_judge.run_label_pilot --server-session-id "$judge_session"
```

This is the bounded generation command. Do not rerun earlier synthetic checks,
evidence pilots or the original token audit. The original audit already fits.
The client uses the persistent tokenizer cache and does not download assets.

## Cached inspection and partial runs

```bash
python -S -m post_thesis.llm_judge.run_label_pilot --inspect
```

Inspection needs no HTTP, tokenizer or server and makes no new attempts. An
optional `--max-new-attempts 5` caps one invocation; a later normal invocation can
continue unattempted inputs within the original total budget, unless a failure
or interruption halted the run. No run-id override or retry switch is provided.
Changes to code, tokenizer, request identity or budget fail rather than silently
resetting a run. The cached summary is derived again from the stored responses.

Private files are under
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-train-pilot-50-label-score-v1/`:
`prepared.json`, `budget.json`, `journal.sqlite3`, `summary.json`. Keep them private
because they contain benchmark text and responses. Return the console summary,
score rows and report SHA256. Review all 50, retaining failures in denominators.

No test thresholds, calibration fitting, prompt changes, TEST/HaluBench scoring,
native-source disjointness claim or benchmark release is part of this pilot.
The pending native-source overlap audit must still be resolved before final
split/threshold decisions. Subsequent evaluation must use the canonical HaluBench
split rather than creating another split.

## Local validation

All 358 judge tests passed, including 16 TRAIN runner/preparation regressions.
HTTP transport tests use fake responses and make no model calls. The actual
pinned tokenizer reproduced all 50 frozen requests (60,574 input tokens), and
the shared payload builder left all 30 historical synthetic request fingerprints
unchanged. Live TRAIN behavior remains untested until the bounded cluster run.
