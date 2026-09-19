# Post-thesis: bounded verdict-first live diagnostic

**Synthetic development only; excluded from the submitted thesis.** The
[native order check passed 44/44](../../results/post_thesis/llm_judge/evidence_schema_v3_native_20260919.md).
This separately budgeted run tests emitted field order, contract validity and
judgments on the same 14 examples. The [completed result](../../results/post_thesis/llm_judge/evidence_diagnostic_v3_20260919.md)
contains 14/14 valid, correctly ordered outputs and 13/14 expected verdicts.
The commands below are historical instructions for scoring revision
`7c4823789ba0125e85d085baf0e2bab564ca417b`; preserve its run and do not rerun
from this reporting commit. Use the [original TRAIN token audit](EVIDENCE_PILOT.md) next.

## Frozen comparison

Run ID: `qwen3-evidence-synthetic-diagnostic-v3`. The [execution plan](configs/evidence_diagnostic_v3.json)
requires the exact serialized schema digest
`f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33`.
The canonical schema digest equals v2 because member ordering does not change
JSON Schema validation rules; it cannot identify this order experiment alone.

The desired raw response member order is `verdict`, `issue_type`, `answer_quote`,
`context_quote`, `explanation`. Schema values, title, prompt text (including its
original field listing), parser, model profile, inputs and order remain fixed.
The request contract and exact schema serialization identify this new run.

Keep the `evidence-v1` inference profile and existing automatic backend-selection
policy. Do not add a backend override. The installed native check does not attest
which compiler the live endpoint selects; observed results concern this pipeline
under that policy. Same recorded package versions, profile, launcher command and
clean code revision are required before new calls. These records and HTTP
preflight are reproducibility checks, not cryptographic weight attestation.

## Budget and persistence

| Setting | Fixed value |
| --- | --- |
| Inputs | Same 14 synthetic examples |
| Invocation / attempts | One invocation; one attempt each; at most 14 |
| Client window | 300 seconds including preflight and audit |
| Input caps | 4,096 tokens each; 57,344 total |
| Output caps | 512 tokens each; 7,168 requested total |
| Tokenization calls | At most 28; full-set audit before first generation |
| Request timeout / concurrency | 60 seconds / 1 |
| Temperature / thinking / truncation / retries | 0 / disabled / none / none |

A completion, error or interruption cannot open a new generation window. Repeats
are cache-only. Preserve all journals and budget records. Alignment errors and
HTTP 400/422/500 halt further generation without a fallback schema. Measured
client time excludes server startup/idle; unknown usage remains unknown and
cooperative timeouts do not guarantee remote computation stopped immediately.

## Raw-order reporting

The runner derives `response_field_order` from the original saved response text,
never from the canonicalized evidence object. It reports matches, mismatches and
unavailable order separately from the unchanged parser's validity decision and
from expected verdict/issue matches. Malformed, duplicate-key or non-object JSON
has unavailable order. A complete JSON object in a refused/truncated response
can have observable order but remains invalid as a judgment. An old-order object
may still be contract-valid; it is retained with an explicit order mismatch.
Nothing is silently reordered, repaired or rejudged.

The console prints all raw field-order lists. Exit code zero requires 14 valid
records and 14 order matches. It does not require correct judgments or certify
evidence semantics. A mismatch uses the same single invocation allowance; do not
rerun to obtain the desired order. Review the two prior false positives and every
accepted explanation, including the recurring unknown-versus-absence confusion.

## H200 procedure after commit, push and CI

Stop any previous launcher with Ctrl+C in its terminal. Preserve its resource
record. Pull and start the same evidence profile from the new clean revision;
no dependency reinstall is needed.

Terminal A:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0 --profile evidence-v1
```

Terminal B: replace the placeholder with the exact new server directory printed
in terminal A.

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
judge_session=server-REPLACE_WITH_NEW_ID &&
tail -n 40 -f ".artifacts/post_thesis/llm_judge/$judge_session/server.log"
```

After `Application startup complete`, stop only the log tail with Ctrl+C:

```bash
python -m post_thesis.llm_judge.evidence_diagnose \
  --schema-version v3 --server-session-id "$judge_session"
```

Preserve the console output and private records under
`.artifacts/post_thesis/llm_judge/qwen3-evidence-synthetic-diagnostic-v3/`.
After the invocation, stop terminal A's launcher to finalize its lifetime record.
From the same clean scoring revision, cache-only inspection uses:

```bash
python -m post_thesis.llm_judge.evidence_diagnose --schema-version v3 --inspect-only
```

Review live ordering and evidence quality before planning anything further.
There are no TRAIN/test calls, threshold fitting, probability conversion,
label changes or final prompt freeze in this diagnostic.
