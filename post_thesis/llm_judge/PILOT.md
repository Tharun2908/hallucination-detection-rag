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

## Next gate before any pilot inference

Review the preparation output. Then implement and record an exact chat-template
token-length audit, the pinned serving profile, bounded attempts/time budget,
resource accounting, and a resumable pilot scoring command. Overlength inputs
must fail explicitly without silent truncation or substitution. No inference
command is added by this preparation patch.

Synthetic checks and this pilot are development evidence. Any prompt revision
must get a new version and separately identified runs. Freeze the final prompt
and settings after development and before either benchmark test evaluation.

Sources: [dataset revision](https://huggingface.co/datasets/wandb/RAGTruth-processed/commit/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b),
[TRAIN file](https://huggingface.co/datasets/wandb/RAGTruth-processed/blob/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/train-00000-of-00001.parquet).
