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
`generation_calls: 0` and the maximum/total formatted input tokens. These are
expected conditions, not claimed results: the actual cluster audit is pending.
Keep the supervised server records; advertised model aliases cannot attest
weights. Tokenization timing is not generation latency. Client windows overlap
the server lifetime and must not be added to it; rental cost remains unknown.
The audit deadline does not stop the server. Stop terminal A after the audit if
no further work is running, so its supervisor saves the final resource record.

## Gate before pilot scoring

Use the measured lengths to record the bounded inference attempt/time budget
and resource accounting, then add a resumable pilot scoring command. A successful
token audit does not itself authorize or perform scoring. Overlength inputs must
fail explicitly without silent truncation or substitution. The threshold subset
and native-source overlap audit remain separate pending tasks.

Synthetic checks and this pilot are development evidence. Any prompt revision
must get a new version and separately identified runs. Freeze the final prompt
and settings after development and before either benchmark test evaluation.

Sources: [dataset revision](https://huggingface.co/datasets/wandb/RAGTruth-processed/commit/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b),
[TRAIN file](https://huggingface.co/datasets/wandb/RAGTruth-processed/blob/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/train-00000-of-00001.parquet).
