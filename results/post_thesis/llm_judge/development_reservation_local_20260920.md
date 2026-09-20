# Post-thesis TRAIN development reservation: local validation

The [committed allocation design](../../../post_thesis/llm_judge/DEVELOPMENT_SPLIT_PROTOCOL.md)
at `c982d26960fe6ea47d808c9d8083b0e3b357f175` was implemented and checked locally on
the pinned 15,090-row TRAIN file and completed audit reports. No selection rule,
seed, component count or membership was changed after inspecting these summaries.
This is assistant CPU validation of an uncommitted implementation, not a completed
cluster run. The [structured record](development_reservation_local_20260920.json)
contains input identities, membership hashes, counts and limitations.

| Partition | Rows | Components | Native sources | Label 0 / label 1 |
| --- | ---: | ---: | ---: | ---: |
| Excluded | 312 | 52 | 52 | 166 / 146 |
| Calibration | 600 | 100 | 100 | 319 / 281 |
| Operating threshold | 600 | 100 | 100 | 324 / 276 |
| Unallocated | 13,578 | 2,258 | 2,263 | 7,560 / 6,018 |

Every selected component has six responses. All known pilot- and native-TEST-linked
TRAIN rows are excluded. No component spans two partitions. Thirteen TRAIN sources
with incomplete business identities are all in the unallocated pool; this is an
observed outcome of the unchanged ranking, not an extra filter. Unknown or partial
document overlap remains possible even for records with complete identities.

Calibration tasks are Data2txt 234, QA 234, Summary 132; threshold tasks are
Data2txt 234, QA 240, Summary 126. Both selected sets contain 100 responses from
each of the six original generators. Labels and task/generator metadata are
attached only for offline analysis after allocation, never used in ranking or
included in model inputs. Selected labels are no longer unseen by the researcher;
these are development fitting/selection sets, not an independent validation set.

Ten focused tests passed for exact ranking and order invariance, sibling retention,
label/metadata-independent allocation, pilot/TEST exclusion, identity/count/flag
drift, design/report checksums, atomic replay/conflict handling, model-input
isolation, single-class behavior and preservation of original inputs. The actual
pinned-data manifest was also written and replayed without changes. There were
zero model calls and zero network requests; no calibrator or threshold was fitted.

The local full-manifest hash includes a clearly marked uncommitted code identity.
The cluster hash will differ once this implementation is committed. **Partition
membership hashes and counts must match**; the local check is not a substitute
for pinning the completed cluster artifact. Historical manifests and audit files
remain unchanged. See the [run instructions](../../../post_thesis/llm_judge/DEVELOPMENT_RESERVATION.md).
