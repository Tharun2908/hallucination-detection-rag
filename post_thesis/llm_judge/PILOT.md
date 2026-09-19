# Post-thesis: prepare the first RAGTruth TRAIN pilot

This is development after thesis submission, excluded from submitted thesis
results. This step prepares data only. It does not score any example, load a
model, read RAGTruth TEST, or read HaluBench. Keep development prompt v1 unchanged.

## Recorded selection rule

- Dataset: `wandb/RAGTruth-processed`, revision
  `eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b`.
- File: `data/train-00000-of-00001.parquet`; 15,090 rows. SHA256:
  `c14ae31ff459c829edc860bda034ee2dbc0a11107b7511195a32bb4ab1ee8000`.
- Seed: `20260919`; selection version: `ragtruth_train_pilot_v1`.
- Group by SHA256 of NFKC-normalized context with whitespace collapsed and case
  preserved. Hash-rank groups using the version/seed and select the first 50.
  Within each group, hash-rank original string IDs and select one response.
- Never use labels, task type, model, or quality to select examples. Preserve
  exact original answer/context strings; normalization is for grouping only.
- Reserve all responses in each selected context group from later threshold
  development. Unselected siblings are excluded, not scored or replaced.
- Record input hashes, original IDs, original TRAIN indices, all TRAIN group
  assignments, initial prompt identity, and the preparation Git commit.

**Grouping limitation:** the processed dataset does not expose native source
document IDs. This is a shared-context proxy, not proof of underlying document
disjointness or a near-duplicate audit. Before fixing a threshold-development
subset, verify native source provenance/overlapping evidence where available
and expand exclusions if required. Do not claim strict native-source independence
from this manifest alone. The remaining rows are only candidates; no threshold
subset, calibration method, or threshold has been selected.

One response per sampled group broadens pilot evidence coverage. It is not a
row-uniform benchmark sample; pilot metrics must not be presented as test results.
Do not reroll the seed after inspecting labels or judge scores.

## CPU preparation on the cluster

First apply, commit and push the code, then update the GPU checkout. The command
records its Git commit and rejects tracked uncommitted changes. The GPU server
can be stopped with Ctrl+C in its launcher terminal; this step needs no server.

From the repository root, use the working Python 3.12 interpreter to create a
separate small CPU environment. This leaves serving dependencies untouched:

```bash
source .venv-judge-serving/bin/activate &&
python -m venv .venv-judge-data &&
.venv-judge-data/bin/python -m pip install -r post_thesis/llm_judge/requirements-data.txt
```

Download only the pinned TRAIN file to the larger root filesystem. If `curl` is
missing, install it with the container's package manager first.

```bash
mkdir -p /root/llm-judge-data-cache &&
curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 \
  'https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/train-00000-of-00001.parquet' \
  --output /root/llm-judge-data-cache/ragtruth-train.parquet &&
.venv-judge-data/bin/python -m post_thesis.llm_judge.prepare_pilot \
  --train-parquet /root/llm-judge-data-cache/ragtruth-train.parquet
```

The script verifies the exact file checksum before parsing it. It stops on
invalid IDs, malformed labels/inputs, unexpected row counts or incompatible
existing manifests. No rows are silently dropped. Rerunning the Python command
with identical data/code reuses the manifest. A changed manifest is never
overwritten; preserve it and review the change rather than deleting it to reroll.

Expected selection (validated locally against the pinned parquet):

| Field | Count |
| --- | ---: |
| TRAIN rows / normalized-context groups | 15,090 / 2,514 |
| Selected pilot examples / groups | 50 / 50 |
| Unselected siblings excluded from threshold development | 250 |
| Remaining threshold candidates | 14,790 |
| Supported / unsupported pilot labels | 26 / 24 |
| Data2txt / QA / Summary pilot examples | 24 / 18 / 8 |

The counts were inspected after applying the recorded rule. No seed search or
label balancing was performed. All six generator identities appear; metadata
stays offline and is never interpolated into judge requests.

## Private output and input boundary

Default output:
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-v1/manifest.json`.
With `RAG_WORKSPACE`, use that workspace's `post_thesis/llm_judge/` namespace.
Keep this file private: it contains full pilot text and development labels.

The checksum-protected manifest has separate `pilot_inputs`,
`pilot_labels_offline`, and `train_reservation` sections. Future judging must use
`pilot_examples(bundle)`, which validates identity/alignment and returns immutable
`Example` objects whose `JudgeInput` contains **only answer and context**. Never
send an entire manifest, label record, task/model metadata, or query to the judge.

## Recorded cluster preparation

The operator prepared this manifest at commit
`c44f4cf172811a086437e69c4f45b15fcda3ed16` with the expected counts above:
`ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
Rebuilding from the pinned TRAIN file and that preparation commit reproduced
the same manifest hash locally. No judge scores were produced.

**Keep that manifest. Do not rerun preparation after updating the code:** its
recorded preparation commit is intentionally different from a later audit or
scoring commit. New stages reference its hash rather than rewriting it.

## Formatted token-length audit (no generation)

Apply, commit and push the audit code, then pull it on the cluster. Start the
pinned server in terminal A, retaining the printed server session directory:

```bash
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0
```

Wait for `Application startup complete` in the new server log. In terminal B,
activate the serving environment and run from the same committed checkout:

```bash
python -m post_thesis.llm_judge.audit_pilot \
  --expected-manifest-sha256 ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25
```

The command checks manifest integrity, input alignment and unchanged prompt
identity before contacting the server. It checks `/version` and `/v1/models`,
then sends only the exact fixed prompt and answer/context messages to `/tokenize`.
It uses the same tokenization payload as generation: non-thinking chat template,
generation prompt enabled, and no extra special tokens. Labels, IDs and metadata
remain outside these requests. It never calls `/v1/chat/completions` or any other
generation endpoint, and never creates probabilities or evaluation metrics.

Each count is saved before proceeding. Limits: one tokenization attempt per
example, 50 examples, sequential execution, at most 60 seconds per HTTP operation
and 300 seconds for each invocation's preflight/counting window. Oversized inputs
are reported unchanged; no truncation, replacement or filtering is performed.
The fit criterion is `formatted_input_tokens + 128 <= 32768`.

Default private directory:
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-token-audit-v1/`.
`audit.json` holds checksum-protected per-example counts, request identities,
prompt/configuration and code revision. A separate `client-window-*.json` records
every invocation, including status, attempted tokenizations, summary and elapsed
resource window. Same-ID reruns reuse completed counts; a fully cached replay
makes no HTTP requests. Incompatible configuration or corrupted records stop
before network activity. Failed/interrupted attempts are terminal for that audit
ID; inspect and fix the cause before deliberately using a new `--audit-id`.
Such a new ID may recount inputs and retains the old evidence.

Review `examples_counted: 50`, `overlength_ids: []`, `all_inputs_fit: true`,
`generation_calls: 0` and the maximum/total formatted input tokens.
Keep the supervised server records; advertised model aliases cannot attest
weights. Tokenization timing is not generation latency. Client windows overlap
the server lifetime and must not be added to it; rental cost remains unknown.
The audit deadline does not stop the server. Stop terminal A after the audit if
no further work is running, so its supervisor saves the final resource record.

## Recorded cluster token audit

The operator reported a completed audit at commit
`02ebaced714b0196607ed66b71796391c6d27859`, with checksum
`08dee554dbedde9c53354862af9d2f718c1f52895b33b32379fe4a9971a2b9f0`.
All 50 inputs were counted with zero failures, pending rows, overlength inputs or
generation calls. Input lengths: minimum 488, maximum 2,753, total 48,924 tokens.
Including the 128-token output allowance, the largest request needs 2,881 tokens.
These are operator-reported aggregate observations; the scoring command verifies
the complete local audit file against this checksum before any network call.
Do not recreate the preparation manifest or token audit after pulling scoring
code: their historical commit identities are intentionally frozen.

## First scoring pilot: frozen execution plan

[configs/ragtruth_pilot_50_v1.json](configs/ragtruth_pilot_50_v1.json) fixes the
manifest/audit checksums, their code revisions, development prompt, profile,
token totals and the following limits before the first real TRAIN scoring call:

| Setting | Frozen value |
| --- | --- |
| Run ID | `qwen3-ragtruth-train-pilot-50-v1` |
| Examples | Exactly the existing 50 TRAIN development inputs |
| Attempts | One per input, across resumes; no automatic retries |
| Concurrency | 1 |
| Per-attempt timeout | 60 seconds, including re-tokenization and generation |
| Cumulative client deadline | 600 seconds across invocations |
| Output allowance | 128 tokens per input; at most 6,400 requested across 50 inputs |
| Audited input tokens | 48,924 across the 50 requests |
| Truncation / fallback / prompt revision | None |

The 55,324 input-plus-maximum-output token bound describes requests under this
configuration, not a monetary price. No rental rate is known. A server reporting
tokens above the allowance causes an explicit halt; retain its actual known usage.

After committing/pushing the scoring patch, pull it on the cluster and restart
the pinned launcher from that same clean commit (terminal A, commands above).
Wait for `Application startup complete`. In terminal B, activate the serving
environment, then run from the repository root, substituting the **directory
name printed by that launcher** for `server-REPLACE_WITH_SESSION_ID`:

```bash
python -m post_thesis.llm_judge.run_pilot \
  --server-session-id server-REPLACE_WITH_SESSION_ID
```

**This command generates the first real TRAIN pilot scores.** No new packages
are needed in the serving environment. It loads only the existing private pilot
manifest and token audit; it does not load either test set, select thresholds,
compute benchmark metrics, or revise the prompt. The API sees only the fixed
prompt plus answer/context. Labels stay offline for later development analysis.

The launcher session record must describe the pinned H200 profile from this
scoring commit and still be marked `started`. Its snapshot is retained in the
execution ledger, and `/version` and `/v1/models` are checked before scoring.
This links the execution to operator-controlled launcher evidence, not remote
cryptographic attestation of weights. Keep the environment and server logs.
Before each completion, the formatted input is counted again with the same
payload and must equal its frozen audit count. A token/model alignment mismatch
or excessive reported output stops the invocation and blocks further calls on
resume. The failing result keeps known usage; subsequent rows remain pending.

### Resume, inspection and budget accounting

The command has a fixed run ID and no budget override. Repeating it reuses
successful results, preserves terminal failures and scores only pending examples
within the original remaining budget. `--max-new-attempts 10` optionally limits
one invocation to at most ten new attempts; it does not increase lifetime limits.
To inspect/recover the journal without network calls or a running server:

```bash
python -m post_thesis.llm_judge.run_pilot --max-new-attempts 0
```

Keep all files under
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-train-pilot-50-v1/` (or the same
namespace beneath `RAG_WORKSPACE`):

- `journal.sqlite3` and `manifest.json`: exact requests, per-attempt outcomes and
  raw response metadata using the existing durable runner.
- `execution/budget.json`: checksum-protected plan/code identity and every client
  time window, serialized by a process lock. Each window reserves the remaining
  budget before network activity and records measured duration on orderly exit.
- `pilot_summary.json`: current coverage, statuses, actual known token usage,
  invocation attempt count and cumulative charged/remaining client seconds.

Use `pilot_summary.json` for invocation reporting. The underlying runner's
`summary.json` is refreshed by a final zero-call recovery pass, so its own
`new_attempts_this_invocation` refers to that recovery pass, not the outer pilot
invocation. Neither file drops failures or substitutes probability 0.5.

The budget includes preflight, tokenization, inference and runner bookkeeping;
local artifact validation/cache inspection and server startup/idle time are
outside it. The timeout is cooperative: cancellation/cleanup can slightly exceed
the client deadline, and cancellation does not prove server work instantly
stopped. Measured overrun is retained rather than clipped. Stop the server in
terminal A when the run finishes; its supervisor records the full serving window.
Do not add overlapping server and client resource windows together.

If the client is forcibly killed and elapsed time is unknown, the unfinished
window is charged its full reserved remaining allowance. Cache inspection still
works, but more inference is blocked. Missing/corrupt ledgers or journals and
changed code/configuration also fail closed. Preserve failed runs for review;
do not delete files, change the run ID, or edit budgets to restart spending.
Unknown usage/cost remains unknown. Alignment failures require inspection, not
blind retries. No stability repeat is included in this 50-example budget.

Exit code zero means all 50 examples have valid scores, not that the judge is
accurate or calibrated. Inspect coverage, endpoint saturation, score distribution,
absence/numeric cases, usage and timings next. The threshold subset and
native-source overlap audit remain separate pending tasks; this pilot is not a
strict source-disjoint final evaluation.

Synthetic checks and this pilot are development evidence. Any prompt revision
must get a new version and separately identified runs. Freeze the final prompt
and settings after development and before either benchmark test evaluation.

Sources: [dataset revision](https://huggingface.co/datasets/wandb/RAGTruth-processed/commit/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b),
[TRAIN file](https://huggingface.co/datasets/wandb/RAGTruth-processed/blob/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/train-00000-of-00001.parquet).
