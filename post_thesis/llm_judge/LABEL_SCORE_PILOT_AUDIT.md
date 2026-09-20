# Post-thesis original-50 TRAIN label-score token audit

This step counts tokens only. It does not load model weights, contact a server,
download assets, generate scores or authorize generation. The synthetic run and
cached replay are recorded in the [findings](../../results/post_thesis/llm_judge/label_score_synthetic_v1_20260920.md).

Keep the original manifest, including examples whose earlier evidence outputs
were invalid. Its SHA256 is
`ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
The auditor verifies the cached synthetic report hash
`f239047289c56c28168927f8518e3203348a47084639c1377af53c75112f654c`
and its historical code revision; it does not rerun that experiment.

From the existing H200 checkout, after committing/pushing this change:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.audit_label_pilot
```

The serving process can stay running. No restart, compiler installation or weight
download is needed. The pinned tokenizer must already exist in the persistent
`label-tokenizer-cache-v1` directory beneath the private post-thesis artifact tree.
The command checks file hashes, package versions, chat template and four reference
boundaries before formatting the 50 inputs with the fixed primary prompt. Only
answer/context enter the prompt; IDs and labels stay outside it. The length check
reserves one output token and never truncates an input.

The private `ragtruth-train-pilot-50-token-audit-label-score-v1/audit.json` contains
input identities, rendered prompts, token fingerprints, the tokenizer environment
and aggregate lengths. Do not commit it: it includes benchmark content.
Each completed CPU row is saved atomically. Repeating the command under the same
code and environment reuses saved rows; interruption may recompute only an unsaved
row. Changed inputs/environment or corrupt records fail instead of overwriting.
No new generation allowance is created on resume.

Return the printed summary, audit SHA256 and code revision. A later change can
pin that audit in a bounded 50-call scoring plan. All 50 remain reused adaptive
TRAIN development data; no threshold or calibration is fitted here. The native
source-overlap audit remains unresolved, and the shared-context grouping proxy
must not be described as verified native-source disjointness. TEST and the
canonical HaluBench split remain untouched.

## Assistant CPU validation

The pinned TRAIN parquet reproduced the original manifest hash exactly using its
recorded preparation revision. With the pinned tokenizer, all 50 inputs fit:
60,574 total input tokens, minimum 721, maximum 2,986, plus one output token per
input. This local development check did not inspect the private cluster synthetic
report and does not substitute for the cluster audit's checksum or code revision.
The new audit regression tests and existing judge tests passed (342 tests).
