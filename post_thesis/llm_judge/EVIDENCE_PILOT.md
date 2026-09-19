# Post-thesis: original TRAIN pilot evidence audit

**Development only; excluded from the submitted thesis.** The [v3 synthetic run](../../results/post_thesis/llm_judge/evidence_diagnostic_v3_20260919.md)
produced 14/14 valid, correctly ordered outputs and 13/14 expected verdicts/issue
types. Seven unsupported explanations were consistent with the fixtures; one
embedded absence claim was missed. Hold this candidate fixed for the next
original-50 TRAIN comparison rather than tuning it again on the synthetic set.
This is a development decision, not the final benchmark freeze.

## Exact input and request contract

Reuse the existing private manifest; do not prepare a new sample:

`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-v1/manifest.json`.

Its required SHA256 is
`ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
The original preparation, sample order, exact answer/context, labels and exclusion
records remain unchanged. No TEST, HaluBench, other TRAIN rows, downloads or
PyArrow installation are needed for this audit.

The evidence prompt remains `faithfulness-evidence-diagnostic-v1`, SHA256
`21ba57c9ec668c495a16e4635660c4504f26b5056638d4b17e6424e32ce52f4c`.
For this audit selector, that prompt explicitly uses request contract
`evidence-diagnostic-request-v3` and exact serialized schema digest
`f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33`.
The profile is `evidence-v1`, including a 512-token output allowance.

Only the fixed system prompt and answer/context messages reach `/tokenize`.
Labels, IDs, query, generator and task metadata remain offline. The identity
records the exact schema string, canonical schema digest, serialized schema
digest, expected field order, profile, request keys and code revision for future
scoring alignment. Tokenization does not exercise structured decoding or judge
faithfulness; those require a separately budgeted later run.

## Token-only limits

- Exactly 50 existing inputs; one tokenization attempt each, sequentially.
- At most 60 seconds per HTTP operation and 300 seconds per audit invocation.
- Fit criterion: formatted input tokens + **512** <= **32,768**.
- Report overlength inputs unchanged; no truncation, replacement or filtering.
- Failed/interrupted attempts remain terminal; completed counts are reused.
- Fully cached replay makes no HTTP requests. Each invocation retains its own
  resource window; this is not a cumulative scoring budget or cost estimate.
- There are **zero generation calls** and no new scores or evaluation metrics.

The fixed evidence audit directory is
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-token-audit-evidence-v3/`.
The evidence CLI rejects another audit ID. Preserve `audit.json` and all
`client-window-*.json` records. Missing/corrupt/incompatible artifacts require
inspection; do not delete them or rename a run to retry terminal inputs.

## Historical audit procedure at revision 9bb7dc3

This audit is complete. Preserve its artifacts and use the [scoring procedure](EVIDENCE_PILOT_RUN.md) next; do not rerun these historical commands at a newer revision.

Stop the prior launcher if still running, then pull the new commit and start
the same evidence-profile server. No serving dependency changes are needed.

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

Terminal B: after verifying `Application startup complete` in that session's log:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_pilot \
  --prompt-version faithfulness-evidence-diagnostic-v1 \
  --expected-manifest-sha256 ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25
```

The prompt name ends in v1; the command prints **Evidence schema version: v3**.
That distinction is intentional. Save the full console output, including token
counts, audit SHA256, serialized schema SHA256 and audit code revision. Stop
terminal A's launcher afterwards to finalize its resource window.

## Recorded outcome and next step

The [operator-reported audit](../../results/post_thesis/llm_judge/ragtruth_evidence_v3_audit_20260919.md)
counted all 50 inputs, with 77,574 total input tokens and no overlength cases.
The [separate bounded scoring configuration](EVIDENCE_PILOT_RUN.md) now pins that
audit and the unchanged candidate. Evidence TRAIN scoring results remain pending.
