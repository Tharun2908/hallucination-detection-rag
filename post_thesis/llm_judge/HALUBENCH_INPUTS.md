# Post-thesis canonical HaluBench TEST inputs

This extension is separate from the submitted thesis. The frozen RAGTruth judge,
calibration, label mapping and raw-margin threshold transfer unchanged. This
step builds an input manifest, not a new split, scoring run or evaluation.

## Fixed dataset and split

- Dataset: `PatronusAI/HaluBench`, revision
  `5966a87929f51c204ab3cbef986b449495cc97b6` (upstream commit dated 2024-07-11).
- File: `data/test-00000-of-00001.parquet`, 7,512,526 bytes;
  SHA256 `c7e9cf966085ffae88d2947744418a05a26ea94380c35c238b9fc12ecb874cdc`.
- Remove exactly `source_ds == "RAGTruth"`: 14,900 upstream rows become 14,000.
- Preserve the order and membership of `test_filtered_indices` in the existing
  `results/cross_domain/halubench_groupfix/halubench_group_split.json`, SHA256
  `9d78af5b4623893cfc7d761dc6be4c2e0f64278faaba11b2d10e6f829e67bb4b`.
  The complement remains the unused 6,000-row adaptation pool.
- Model inputs are only original `answer` and `passage` (as `context`), unchanged.
  Original IDs identify rows. Labels PASS/FAIL map to offline 0/1; question,
  source, indices and grouping are offline only, never separate prompt fields.

The CLI checks the accepted private RAGTruth fit and the unchanged judge freeze.
It pins split/dataset/cache files, rejects malformed indices/IDs/labels and
checks the saved TEST source/label counts. Exact text hashes identify this new
input manifest; they do not establish historical cache text identity.

## Grouping and observed limits

Canonical grouping reproduces the existing case-sensitive whitespace-normalized
`source_ds + question + passage` relation. Its TRAIN/TEST overlap must be zero.
For the registered bootstrap, canonical groups are merged when a nonempty passage
matches exactly. This does not alter which rows belong to TEST.

Local CPU preparation on the pinned data found:

| Check | Count |
|---|---:|
| Canonical TEST rows | 8,000 |
| Canonical TEST groups | 7,838 |
| TEST bootstrap components after exact-passage merging | 7,198 |
| Canonical outer group overlap | 0 |
| Merged components touching both TEST and adaptation | 355 |
| TEST rows in those shared components | 994 |

Thus the canonical split is group-disjoint under its original definition, not
strictly passage-disjoint. Preserve and disclose this distinction. The new
zero-shot comparison uses no HaluBench adaptation training; do not interpret
these counts as evidence of training exposure for the frozen judge. Partial or
fuzzy document overlap and pretraining contamination are not established here.

Only source/question/passage/ID metadata from the adaptation pool is used for
the group check. Adaptation answers and labels are not used for selection,
training or evaluation and are not placed in Python row records or the manifest.
Arrow decodes answer/label columns before selecting TEST rows, so this is not a
claim that physical parquet reads never touch adaptation values.

## Legacy baseline inventory

Both committed 14k caches have valid selected scores and matching TEST indices,
labels and source names. They lack original IDs and answer/context hashes, so
`comparison_ready` remains false. No legacy scores are joined to judge metrics.
Later baseline preparation must establish fresh inference linkage or equivalent
content provenance, preserving original files and the frozen TRAIN policies.
The old HaluBench MiniCheck `mc_hall >= 0.80` rule must not silently replace the
registered strict support-space rule `< 0.20000000000000004`.

## Run on the pod (CPU only)

No vLLM server or GPU is needed. The existing fitting environment needs only the
already pinned parquet dependency for this step:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m pip install -r post_thesis/llm_judge/requirements-data.txt &&
mkdir -p .artifacts/post_thesis/llm_judge/halubench-inputs-v1 &&
curl -L --fail --retry 2 \
  https://huggingface.co/datasets/PatronusAI/HaluBench/resolve/5966a87929f51c204ab3cbef986b449495cc97b6/data/test-00000-of-00001.parquet \
  -o .artifacts/post_thesis/llm_judge/halubench-inputs-v1/test.parquet &&
python -m post_thesis.llm_judge.prepare_halubench \
  --parquet .artifacts/post_thesis/llm_judge/halubench-inputs-v1/test.parquet
```

The private manifest is saved at:

```
.artifacts/post_thesis/llm_judge/halubench-canonical-test-input-manifest-v1/manifest.json
```

Repeating the final Python command at the same revision verifies the identical
manifest without rewriting it. A changed manifest fails rather than overwriting
an earlier run. Keep private texts out of Git.

After recording the pod manifest, CPU token lengths, fresh baseline preparation
and a bounded judge execution plan remain. No generation allowance, fitting,
TEST metrics, threshold changes or new split are introduced by this command.
