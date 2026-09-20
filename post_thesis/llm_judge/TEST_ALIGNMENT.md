# Post-thesis RAGTruth TEST input and baseline audit

The frozen fit checker passed on the pod with no fitting/model calls or rewritten
files. Preserve the accepted map and raw threshold -1.25. This next step prepares
TEST inputs and reports available baseline caches; it performs no judge inference,
metric computation, tokenization, calibration or threshold fitting.

## What is fixed and verified

The processed dataset stays at `wandb/RAGTruth-processed` revision
`eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b`. The TEST parquet SHA256 is
`2fc4fb703ea47ee0d4ab6110b86312f94fdf0bda157bc6ee67c7e61fb90d3bbd`.
Read all original 2,700 rows in their original order. Keep `output`/`context`
unchanged. Labels, original index, source ID, grouping and task/generator metadata
are stored in a separate offline section; model inputs contain only ID,
answer and context, with ID reserved for alignment outside future requests.

Verify the completed fit, development reservation and native cross-split report
against their pinned hashes. Reconstruct the same native/context/exact-evidence/
business connected components over the pinned native source release, verify
all prior TRAIN component identities, and reject any TEST component touching
the pilot or either fitting arm. This does not certify fuzzy/partial overlap,
strict document disjointness, HaluBench overlap or baseline training exposure.

The local CPU check matched all 2,700 native answers and 2,694 contexts exactly.
The remaining six contexts belong to native source `14347`, response IDs
`12180`–`12185`. Native-formatted text has exactly one extra ASCII space at offset
1,522 (zero-based); deleting that space reproduces the processed text exactly.
Both texts have the same already-registered whitespace-normalized group hash.
This was inspected before judge TEST inference. The exception pins both full
context hashes, the source and all six IDs; other discrepancies still fail.
**The processed context is never rewritten.** Native annotations are not used to
replace processed labels. See the [local record](../../results/post_thesis/llm_judge/test_alignment_local_20260920.json).

The manifest is immutable. Same-checkout replay validates equality and does not
rewrite it. Future revisions consume this historical manifest instead of rerunning
its preparer with a changed code identity.

## Baseline inventory is separate from a comparison-ready manifest

Inspect only these explicitly named TEST files in the supplied baseline directory:

- `signal4_results_test.json`
- `minicheck_results_test_7b.json`
- `relevance_results_test_v2.json` (dependency for metadata-free fusion)

Report file hashes, row coverage, duplicate/invalid indices, score missingness,
label/metadata agreement and any existing input hashes. Never compute metrics
or write to these files. Do not infer missing input hashes from an index match.
An index-only legacy cache remains `content_provenance_unproven`; even matching
input hashes leave checkpoint, training-selected threshold and evidence-visibility
provenance to establish. No inventory entry is automatically comparison-ready.

Metadata-free fusion still needs its fitted-model provenance and aligned
per-example predictions. Its aggregate report cannot substitute for those.
Do not use metadata-aware fusion as the main metadata-free comparator.
Each inventory is saved by content hash so newly located caches can be inspected
without overwriting previous inventory records or changing the TEST manifest.

## Pod commands after applying, committing and pulling the patch

Use the existing CPU environment; installing the pinned parquet reader does not
alter NumPy/SciPy fitting requirements. No GPU server is needed.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m pip install -r post_thesis/llm_judge/requirements-data.txt
```

Download the fixed 3.9 MB processed TEST file. The earlier native release files
remain in `source-audit-inputs-v1`; preserve them and the original audit reports.

```bash
judge_test_dir=.artifacts/post_thesis/llm_judge/test-inputs-v1
mkdir -p "$judge_test_dir" &&
curl --fail --location \
  https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b/data/test-00000-of-00001.parquet \
  -o "$judge_test_dir/test.parquet" &&
python -m post_thesis.llm_judge.prepare_test_manifest \
  --test-parquet "$judge_test_dir/test.parquet"
```

The default baseline directory is the configured `RAG_WORKSPACE` (normally the
repository's `.artifacts`). If the original score caches are elsewhere, add
`--baseline-dir /path/to/original/caches`. This checks named files only, not a
recursive scan. Missing caches are reported, not regenerated or replaced.
An invalid baseline cache does not invalidate the independently verified input
manifest; its comparison remains blocked and visibly marked.

Repeat just the Python command to check identical manifest reuse. Return the
summary, manifest hash, code revision and baseline inventory. Expected local
counts: 2,700 rows, 450 native sources/components, 1,757 supported / 943 unsupported,
900 rows per task, 2,694 exact contexts and six reviewed whitespace-only matches.
Do not claim these counts constitute judge performance.

The command reads processed TEST answers, context and offline labels for alignment;
it does not claim zero TEST access. It reads no HaluBench data, creates no split,
and performs no new model calls. Benchmark baseline provenance, a token audit and
an exact bounded execution manifest remain necessary next steps.
