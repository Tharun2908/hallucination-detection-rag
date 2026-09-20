# Post-thesis offline development reservation

The selector implements the [committed design](DEVELOPMENT_SPLIT_PROTOCOL.md)
without changing its config, seed, counts, ranking or eligibility rules. This
creates calibration/threshold membership only. It does not tokenize prompts,
contact the judge, fit calibration, select a threshold, or read processed TEST.

## Inputs and guarantees

`prepare_development` requires the pinned processed TRAIN parquet and the existing
private original pilot, TRAIN source-audit and native cross-split-audit reports.
It verifies the frozen design hash and both completed audit hashes, rebuilds the
original pilot to check historical TRAIN identity, and requires every TRAIN row
exactly once. No native release download or source-content reread is needed.

The component allocation consumes only audit identifiers/flags. Labels and
metadata do not influence ranking. Original TRAIN labels enter the historical
manifest identity check and offline summaries, after allocation. All 312 exclusions
are preserved in a separate new manifest. All responses from each of the first
100 eligible components go to calibration; the next 100 go to operating threshold.
The remaining components stay unallocated. A missing class is recorded as a
blocked fitting status without replacing any examples or rerolling the seed.

The private manifest separates model inputs (routing ID, answer, context) from
labels/task/generator metadata. The routing ID is never judge input. It records
all partition memberships, component rankings, source IDs, exclusion reasons,
input/membership hashes and known limitations. It uses atomic writes and the
existing cross-platform process lock. Identical replay performs no rewrite;
changed input, design, revision or output under the same run ID is rejected.

The [local validation](../../results/post_thesis/llm_judge/development_reservation_local_20260920.md)
found 600 rows in each selected arm and 13,578 unallocated rows. Ten targeted
regression tests passed, and the actual pinned inputs produced an identical replay.
The existing Windows/Linux CPU workflow discovers the new tests automatically.
The [cluster run and identical replay](../../results/post_thesis/llm_judge/development_reservation_cluster_20260920.md) now match, including independent reconstruction of the full manifest hash.

## Historical reproduction command

The reservation was completed at `6f056d1`. Preserve its private manifest and do
not rerun the selector under a later revision, which changes its identity.
The commands below document that completed run.

Use the existing CPU data environment; no serving process or H200 is needed:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-data/bin/activate &&
python -m post_thesis.llm_judge.prepare_development
```

The environment already used for the TRAIN source audit should contain the pinned
`pyarrow` data dependency. If it was rebuilt, install
`python -m pip install -r post_thesis/llm_judge/requirements-data.txt` first.
Do not use `python -S` for this command: it needs the installed parquet reader.
If the cached parquet is elsewhere, pass `--input-dir /path/to/source-audit-inputs-v1`
(the directory must contain `train.parquet`). Do not rerun the earlier source
audits under the newer code revision; preserve and reuse their existing reports.

The output is:

```text
.artifacts/post_thesis/llm_judge/ragtruth-train-development-reservation-v1/manifest.json
```

Run the same selector a second time, at the same code revision, to confirm
`Verified identical development reservation; no rewrite.` Return the partition
summary, checks, status, design hash, manifest hash and allocator code revision.
No raw answer/context content needs to be pasted or committed.

Expected design SHA256:
`4b8e9722370ef50b4b193a4109e3d890144e66c2cf002b5ff0834dbe42871574`.
The completed cluster manifest SHA256 is
`56b77ada74b638720586f93835ed801d8f090d7a04b9d1f1272a60e2677d7362`.
It includes the allocator commit and differs from the local uncommitted check.

## Next boundary

After verifying the cluster reservation, preregister the exact calibration
objective/optimizer/bounds/failure policy and threshold objective/candidate/tie
rules, then perform token-length audits and record a separate bounded scoring
plan. The reservation itself authorizes zero inference calls. No prompt tuning,
calibrator search, threshold fitting or new HaluBench split is introduced.
Strict document disjointness remains unproven: the guarantee covers the declared
native/exact-overlap components only. All artifacts are post-thesis and do not
replace submitted thesis results.

The [calibration/threshold design](CALIBRATION_THRESHOLD_PROTOCOL.md) now specifies the fitting choices before any new scoring. The [CPU token-length auditor](DEVELOPMENT_TOKEN_AUDIT.md) is now implemented; run it on the existing reservation. A separate bounded inference plan follows; no generation allowance is implied.
