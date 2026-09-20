# Post-thesis offline TRAIN source and exact-overlap audit

This is post-submission provenance work, not a thesis result, model experiment or
final data split. It follows the completed label-score pilot without changing its
prompt, scores or original 50-example manifest.

## Pinned inputs and honest read scope

The original [RAGTruth release documentation](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/README.md)
links response IDs to `source_id` and publishes source information separately.
A native source ID identifies a source-information bundle, not necessarily one
unique underlying document. Multiple bundles can contain overlapping evidence.

The auditor pins release revision
`c103204b9ce28d6bbad859304bf30de72b8ed8fe` and both SHA256 and Git blob hashes for
`dataset/response.jsonl` and `dataset/source_info.jsonl`. It also requires the
existing checksum-pinned processed TRAIN parquet and original pilot manifest.

**The native files contain both TRAIN and TEST records.** Whole-file checksums
read their bytes and JSON parsing sees records from both splits. Only response
IDs belonging to the pinned processed TRAIN set are projected for analysis;
each must have native split `train`. Other response records and their source-only
records are counted and ignored. No native annotation labels are accessed, and
no processed TEST parquet or HaluBench data is loaded. This is not a claim that
no TEST bytes were ever read. TEST labels, predictions and performance do not
enter the audit or any decision.

The historical manifest is rebuilt to verify all TRAIN reservations and exact
input identities. Processed TRAIN labels are used only to verify that historical
manifest; selection and overlap grouping do not depend on labels or judge scores.

## Declared matching and overlap rules

Every processed TRAIN response must match native answer text, task, generator,
quality and the exact context export formatting. For QA, numbered passage markers
are removed and outer whitespace stripped. Summary retains the original source
text plus `\noutput:`; Data2txt retains the Python dictionary representation with
its original leading newline and `\nOverview:` suffix. These are explicit export
rules, not fuzzy matching or corrections to judge inputs. A mismatch stops the
audit instead of guessing a native source.

Build conservative connected components using:

1. Native `source_id` membership.
2. The original normalized full-context grouping.
3. Exact normalized QA passages and full Summary source articles. Normalization
   is Unicode NFKC plus collapsed whitespace, preserving case. Every nonempty
   unit is included, including short units; this can conservatively overgroup.
4. Data2txt business tuples of name/address/city/state, only when all four fields
   are present and nonempty. Normalize whitespace/NFKC and casefold this tuple.
   This is an entity-overlap indicator, not a verified business identifier.

Take transitive closure, report every component touching a pilot source, and
propose exclusions for any linked candidate rows. Retain all original exclusions.
The original manifest is never rewritten, and no calibration subset is selected.
Private output includes row-to-native-source mappings, component hashes, shared
indicator hashes, missing business identities and proposed exclusion IDs.

This audit does **not** establish strict document disjointness: fuzzy duplicates,
partial passages, paraphrases, canonical URLs, TRAIN–TEST overlap, cross-benchmark
overlap and verifier training exposure remain outside scope. Missing business
identity fields remain explicit limitations, not successful identity checks.

## Cluster reproduction after commit and push

No GPU or serving restart is required. Use a separate CPU environment. From the
repository root, the restored serving environment supplies Python to create it:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m venv .venv-judge-data &&
source .venv-judge-data/bin/activate &&
python -m pip install -r post_thesis/llm_judge/requirements-data.txt
```

Download the three fixed input files once (about 59 MB combined). This is static
data retrieval, not model inference. The default paths below assume the existing
artifact-root configuration has not been overridden:

```bash
judge_inputs=/workspace/hallucination-detection-rag/.artifacts/post_thesis/llm_judge/source-audit-inputs-v1
mkdir -p "$judge_inputs" &&
curl --fail --location --retry 2 \
  https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/response.jsonl \
  -o "$judge_inputs/response.jsonl" &&
curl --fail --location --retry 2 \
  https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/source_info.jsonl \
  -o "$judge_inputs/source_info.jsonl" &&
curl --fail --location --retry 2 \
  https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/train-00000-of-00001.parquet \
  -o "$judge_inputs/train.parquet" &&
python -m post_thesis.llm_judge.source_audit --input-dir "$judge_inputs"
```

The audit itself has no network client. It checks hashes before parsing each
release file and emits a private report under `ragtruth-train-source-audit-v1`.
If files are already cached, run only the final audit command. Repeating under
the same code with identical inputs reuses the identical report; changed reports
fail rather than overwrite. An interrupted computation can be repeated because
it makes no model calls and writes only one atomic result after all checks.

Return the printed summary, report SHA256 and audit code revision. Keep the
private mapping and input files out of Git. The
[assistant CPU check](../../results/post_thesis/llm_judge/source_audit_local_20260920.json)
matched all 15,090 TRAIN responses and contexts and found no additional pilot-linked
candidate exclusions under these exact rules. It does not substitute for the
cluster report or close the limitations above.

The cluster check subsequently matched the local results and its complete report hash was independently reconstructed; see the [cluster record](../../results/post_thesis/llm_judge/source_audit_cluster_20260920.json). Preserve that completed report. The next [cross-split source check](CROSS_SPLIT_AUDIT.md) uses it as a pinned input and extends the scope to native TEST source content, without using TEST answers or labels.
