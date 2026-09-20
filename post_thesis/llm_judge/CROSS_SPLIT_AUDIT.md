# Post-thesis native TRAIN–TEST exact-overlap audit

This is offline data-provenance work after thesis submission. It extends the
completed TRAIN audit and does not generate predictions, evaluate TEST accuracy,
fit calibration, choose thresholds or revise the judge prompt.

## Completed prerequisite

The cluster TRAIN audit matches the assistant's local result and its complete
report hash was independently reproduced by deterministic local reconstruction:
`34c1943a2d32a09dabc3a856a963ae898fc43e97175e86549c4f037b6aadae14`.
Its code revision is `80d2fa1baab81a50da674625e21fdf9a825c1f19`.
The [record](../../results/post_thesis/llm_judge/source_audit_cluster_20260920.json)
confirms 15,090 exact answer/context matches, 300 proposed excluded TRAIN rows and
14,790 remaining candidates under the TRAIN-only rules. The existing private
report is a required input and remains unchanged. Do not rerun the earlier audit
under the new code revision.

## Read scope and rules

Reuse the same pinned native `response.jsonl` and `source_info.jsonl` files.
Whole-file checksum verification and JSON parsing read the mixed release bytes.
Project response records immediately to **id, source_id and split only**. Do not
access response text, annotation labels, generator identity, quality or scores.
For source records, project source_id, task_type, source and source_info only.

Unlike the TRAIN-only audit, **native TEST source content is now used for overlap
checking**. This is distinct from reading TEST answers/labels or measuring TEST
performance. No processed TEST parquet or HaluBench file is read. The check does
not establish that native TEST IDs exactly match every downstream processed-test
artifact; processed-test alignment remains a later provenance check.

Validate all native TRAIN IDs/source assignments against the completed TRAIN
report, including reconstructed context hashes. Apply the existing declared
exact-overlap rules across both native splits: native IDs, normalized full
contexts, exact normalized QA passages/full Summary articles, and complete
normalized business identity tuples. Use the same case and whitespace rules;
no similarity threshold or label-driven adaptation is introduced.

Take transitive closure across all these indicators. Propose exclusion of TRAIN
rows whose component touches any native TEST source or pilot source, preserving
all prior exclusions. Record overlap edges and row-to-component assignments
privately. If an adaptive pilot component touches native TEST, flag it explicitly;
do not silently replace pilot rows or remove TEST examples. The original pilot,
TRAIN reservations, benchmark test set and canonical HaluBench split are unchanged.
No final calibration/threshold subsets are created here.

A zero native-ID intersection does not rule out overlapping evidence. Conversely,
these conservative indicators are not proof of semantic duplication: generic exact
passages can overgroup data. Fuzzy/partial overlaps, paraphrases, canonical document
IDs/URLs, incomplete business identity fields, HaluBench overlap and baseline
training exposure remain outside scope. Do not claim strict document disjointness.

## Assistant CPU validation

The [local result](../../results/post_thesis/llm_judge/cross_split_audit_local_20260920.json)
found:

| Check | Result |
| --- | ---: |
| Native TRAIN / TEST responses (IDs and split only) | 15,090 / 2,700 |
| Native TRAIN / TEST source bundles | 2,515 / 450 |
| Shared native source IDs across splits | 0 |
| Shared normalized full-context groups across splits | 0 |
| Exact evidence units shared across splits | 2 |
| TRAIN rows linked to native TEST | 12 |
| Original pilot rows linked to native TEST | 0 |
| Additional candidate exclusions proposed | 12 |
| Proposed total exclusions / remaining candidates | 312 / 14,778 |

Thus the additional check catches overlap that source IDs and full-context hashes
miss. These are proposed conservative exclusions, not evidence of inflated model
performance. All 12 come from the previous candidate pool; none changes the
50-example pilot or its reported metrics. There are 20 source records across both
splits with incomplete business identity tuples; those gaps remain visible.

Eight focused regression tests cover metadata isolation, split/membership errors,
shared native IDs, exact passage overlap, transitive closure, historical report
identity and preservation of prior exclusions. The actual pinned files were also
checked locally. Cluster reproduction is the next step.

## Run after commit and push

Use the existing CPU environment and cached input files; no packages, downloads,
model weights or serving restart are needed:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-data/bin/activate &&
python -S -m post_thesis.llm_judge.cross_split_audit
```

If the inputs were saved outside the default artifact directory, pass the same
`--input-dir` used by the previous audit. The new private report is under
`ragtruth-native-cross-split-audit-v1/report.json`. The command checks pinned
hashes before analysis, writes atomically, and refuses to overwrite a differing
report. Repeating the same code/inputs reproduces the same report. No network
client, GPU library or model call is involved.

Return the printed summary, report SHA256 and audit code revision. The result is
a provenance input for registering disjoint development/calibration data; it does
not grant a new generation allowance or finalize the evaluation split.
