# Post-thesis native cross-split audit: completed cluster result

The operator's cluster result at `141f60aacb210a1c3e55062c7b23f44108d54c82`
matches the prior local check. Reconstructing the complete report locally from
pinned native inputs and the historical TRAIN audit reproduces SHA256
`91f87c50a6c1e42d584b33d925ee62d41964093f83f212980d29587592ac1823`.
The [machine-readable record](cross_split_audit_cluster_20260920.json) preserves
the supplied counts, provenance and added exclusion IDs.

| Check | Result |
| --- | ---: |
| Native TRAIN / TEST responses | 15,090 / 2,700 |
| Native TRAIN / TEST source bundles | 2,515 / 450 |
| Shared native IDs / full-context groups | 0 / 0 |
| Shared exact evidence units / cross-split components | 2 / 2 |
| TRAIN rows linked to native TEST | 12 |
| Pilot rows linked to native TEST | 0 |
| Proposed excluded / remaining candidate rows | 312 / 14,778 |
| Sources with incomplete business identities | 20 |
| Model calls | 0 |

The added candidate exclusions are IDs 13698–13703 and 14988–14993. These follow
the preregistered conservative exact-overlap rules; they are not chosen using
labels or prediction errors. Native TEST source content was used. TEST answers,
annotation labels, model scores and processed TEST parquet were not used.
Original pilot reservations and all historical results are unchanged.

This audit detects overlap that source IDs and full-context hashes miss. It does
not establish strict document disjointness or show that any measured performance
was inflated. Incomplete identities and untested fuzzy/partial overlaps remain.
The original local record is retained as a historical pre-release check.

Next is the [development reservation design](../../../post_thesis/llm_judge/DEVELOPMENT_SPLIT_PROTOCOL.md):
100 components for calibration, 100 different components for operating-threshold
selection, all responses per selected component, and deterministic allocation
before label inspection. No IDs have been selected, no calibrator fitted, and
no threshold chosen. There is no new generation allowance.
