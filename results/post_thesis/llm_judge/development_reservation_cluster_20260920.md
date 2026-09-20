# Post-thesis TRAIN development reservation: completed cluster run

The cluster reservation at `6f056d1f0018eb8d7ae0894279dd1074b70f1662` and its
identical replay both report manifest SHA256
`56b77ada74b638720586f93835ed801d8f090d7a04b9d1f1272a60e2677d7362`.
Independent reconstruction from the pinned local inputs and allocator revision
reproduces the full hash. The replay performed no rewrite. The
[structured record](development_reservation_cluster_20260920.json) preserves
counts, membership hashes and completed checks.

| Partition | Responses | Components | Label 0 / label 1 |
| --- | ---: | ---: | ---: |
| Excluded | 312 | 52 | 166 / 146 |
| Calibration | 600 | 100 | 319 / 281 |
| Operating threshold | 600 | 100 | 324 / 276 |
| Unallocated | 13,578 | 2,258 | 7,560 / 6,018 |

Every TRAIN row occurs exactly once, all prior exclusions are preserved, and
selected components do not touch known pilot or native-TEST components. All 13
TRAIN sources with incomplete business identities remain unallocated under the
unchanged ranking. No replacement or balancing was performed. Exact component
disjointness is not proof of strict underlying document disjointness.

Generation calls and HTTP requests were zero. No scores were collected for these
reserved arms; calibration and threshold fitting remain pending. Historical pilot
scores and reservations are untouched. This is post-thesis work.

The next [calibration/threshold protocol](../../../post_thesis/llm_judge/CALIBRATION_THRESHOLD_PROTOCOL.md)
fixes one positive-slope logistic calibration fit on the calibration arm and one
F1-maximizing raw-margin threshold on the other arm, with deterministic ties,
explicit eligibility and failure handling. Commit it before new scoring; token
lengths and the separate inference budget still need to be recorded.
