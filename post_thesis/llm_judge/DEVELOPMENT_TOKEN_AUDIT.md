# Post-thesis CPU token audit of reserved development inputs

This step measures formatted prompt length for the exact 600 calibration and
600 operating-threshold inputs in the completed development reservation. It does
not collect scores, load model weights, fit calibration or select thresholds.

## Frozen inputs

- Development manifest SHA256:
  `56b77ada74b638720586f93835ed801d8f090d7a04b9d1f1272a60e2677d7362`.
- Allocator revision: `6f056d1f0018eb8d7ae0894279dd1074b70f1662`.
- [Fit protocol](CALIBRATION_THRESHOLD_PROTOCOL.md) configuration SHA256:
  `8902d4010ce17b0d3ac8b9a18fdfe70703cd72c4e62216bb41c6c7767a5703bc`.
- Prompt: `faithfulness-label-score-v1`, SHA256
  `0b3950a21fb265545d422d1f1ec4b21604d74e715a51ce9261320216ad8fcc3c`.

The CLI validates the entire reservation hash and each arm's ordered membership,
component membership and input hashes. It rejects extra input fields, excluded
or pilot/TEST-linked components, duplicates and cross-arm overlap. Whole-manifest
hashing includes offline labels, but only answer/context is projected into
`JudgeInput`. Sample ID and arm remain local routing fields outside the prompt.

Use the existing pinned tokenizer cache. The command permits no download and
checks tokenizer files, library versions, chat template and four reference cases
before auditing. It counts the full prompt including template/assistant boundary
and checks both one-token class continuations. Allow one output token within the
32,768-token limit. Never truncate, replace or silently skip an overlength row;
record all such IDs and exit with status 1 after completing the audit.

## Historical reproduction command

The cluster audit and replay completed at `9182e27`; see the [completed record](../../results/post_thesis/llm_judge/development_token_audit_cluster_20260920.md). Preserve that audit instead of rerunning it after pulling newer code. The commands below document the historical run.

Use the tokenizer packages already installed in the serving environment. This is
CPU-only even though the environment is named `judge-serving`; a running model
server and GPU allocation are not needed. Keep existing tokenizer packages pinned.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_development_tokens
```

The command reads the existing private reservation and cache under
`.artifacts/post_thesis/llm_judge/`. Do not rerun the reservation/source auditors
under this new revision. If the pinned tokenizer cache is missing, stop and report
that error; this command will not download replacement files or model weights.

Run the audit again at the same revision:

```bash
python -m post_thesis.llm_judge.audit_development_tokens
```

Expect `New CPU tokenizations: 0` and the same audit SHA256. The CLI repeats its
four fixed tokenizer compatibility checks at startup; the completed 1,200-input
audit is reused without retokenizing those inputs or rewriting its report.

The private report is:

```text
.artifacts/post_thesis/llm_judge/ragtruth-train-development-token-audit-label-score-v1/audit.json
```

The audit uses the existing process lock, atomic writes and a completed-prefix
checkpoint after every input. A crash resumes from the last committed row;
inputs lost before that checkpoint may be tokenized again locally. Changes to
manifest, fit protocol, input ordering, code revision, profile, tokenizer files
or software identity reject reuse rather than overwriting the old report.

Prepared records retain exact input/message/rendered/token-ID hashes, lengths,
score positions and class-token identities. They omit rendered prompt text to
avoid copying the same raw input into every checkpoint. This does not change
prompt rendering or score semantics. Original text remains in the private manifest.

## Expected counts and next step

The [actual-tokenizer local validation](../../results/post_thesis/llm_judge/development_token_audit_local_20260920.md)
counted all 1,200 inputs: calibration 705,198 tokens, operating threshold 746,401,
combined 1,451,599. Maximum formatted length was 2,886; no overlength rows.
Eight focused tests and completed-cache replay passed. Cluster reproduction and replay matched these counts, and the complete cluster hash was independently reconstructed. Preserve both historical records. The [bounded scoring plan](DEVELOPMENT_SCORING_RUN.md) now pins the completed cluster audit; it is separate from this zero-generation audit.

Return the printed combined/per-arm summaries, prompt, fit-protocol hash, audit
hash and code revision, plus the replay's new-tokenization count/hash. No raw
answer/context needs to be pasted or committed.

After this audit is recorded, freeze a separate bounded execution plan with the
exact audit identity, serving session/profile, call/token/time limits and failure
handling. The audit grants zero model calls. Statistical fitting rules remain as
preregistered; TEST/HaluBench evaluation remains a later stage. All work stays
post-thesis, with the existing exact-overlap limitations preserved.
