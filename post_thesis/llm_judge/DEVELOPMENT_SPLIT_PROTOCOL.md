# Post-thesis development reservation protocol

**Design v1, 2026-09-20. No development IDs selected and no model calls.**
This work is separate from the submitted thesis. The machine-readable design is
[development_reservation_v1.json](configs/development_reservation_v1.json).
Commit this design before implementing or running allocation. It defines data
roles and deterministic selection, not a live inference configuration.

## Evidence and scope

The [completed native cross-split audit](../../results/post_thesis/llm_judge/cross_split_audit_cluster_20260920.json)
matches the local check, including independent reconstruction of its complete
report hash. The audit found two shared exact evidence units connecting 12 TRAIN
candidate rows to native TEST. No original pilot row is linked to native TEST.
Source-ID and full-context intersections alone were both zero.

Preserve the 300 original pilot/sibling exclusions and adopt the 12 additional
exclusions in the forthcoming development manifest: 312 excluded rows and
14,778 eligible candidate rows under the recorded exact-overlap rules. Do not
edit the original pilot manifest or historical TRAIN reservations. Until the new
manifest is created, these remain proposed exclusions rather than selected sets.

The source audit used native TEST source content and response ID/split metadata;
it did not use TEST answers, annotation labels or performance. No processed TEST
or HaluBench file is needed for development reservation. Mixed native release
bytes were parsed, so do not describe the audit as never reading any TEST data.

## Fixed roles and size

| Partition | Selection | Permitted use |
| --- | --- | --- |
| Excluded | All 312 audit-excluded TRAIN rows | Historical development/provenance only |
| Calibration | First 100 eligible components in the fixed ranking | Fit the separately preregistered monotone calibrator |
| Operating threshold | Next 100 eligible components | Select the separately preregistered operating threshold |
| Unallocated | Every remaining eligible component | No inference or fitting under this design |

Keep **all TRAIN responses** in every selected component. A usual native source
has six responses, so approximately 600 rows per arm is a planning estimate, not
an exact row count or a power calculation. Merged components may be larger.
One hundred components per role provides a bounded next experiment with more
source diversity than the adaptively reused 50-example pilot. These sizes are
fixed before inspecting the selected labels or model outcomes. They do not
establish adequate power for small domain-specific differences.

A component is the connected component from the pinned cross-split audit, not
an individual response or merely a native source ID. This keeps all known
source/context/evidence/business-linked siblings in one partition. Do not cap
responses per component, sample just one generator, balance classes, or stratify
by task/model. Select components uniformly by the deterministic hash ranking;
resulting response and task proportions need not exactly match the full pool.
Report those proportions rather than rerolling the split.

## Deterministic allocation specification

Use the exact private report identified by SHA256
`91f87c50a6c1e42d584b33d925ee62d41964093f83f212980d29587592ac1823`,
produced at code revision `141f60aacb210a1c3e55062c7b23f44108d54c82`.
Verify its content hash, prior TRAIN report identity, original pilot manifest,
processed TRAIN file hash, unique IDs, row counts and complete mapping coverage.
Do not accept a new report revision silently.

1. Group the audit's TRAIN mapping by its lowercase 64-character
   `component_sha256`. Reject inconsistent component flags or a component split
   between excluded and eligible rows. Reject any eligible component touching
   pilot or native TEST. Assert all prior exclusions are retained.
2. Require exactly 15,090 mapped TRAIN rows, 312 excluded and 14,778 candidates.
   At least 200 eligible components must exist. On any mismatch stop; do not
   reduce the sample, change grouping rules, or choose another seed.
3. For each eligible component form this exact UTF-8 string, with two newline
   separators and **no trailing newline**:

   ```text
   ragtruth-development-reservation-v1
   20260920
   <component_sha256>
   ```

   Compute SHA256 and sort by lowercase digest hex ascending, breaking a digest
   tie by component hash ascending. This is the only ranking. Do not include
   labels, task, generator identity, answer quality, scores or model errors.
4. Assign ranked components 0–99 to calibration, 100–199 to operating threshold,
   and the rest to unallocated. Keep excluded rows in a separate partition.
   Preserve every row exactly once. Materialize rows by numeric sample-ID order
   within each partition and record the component ranking separately.
5. Only after assignment, attach original TRAIN labels and task/generator
   metadata for offline summaries. Report row/source/component counts, class and
   task composition, and missing-business-identity counts per partition. No
   label repair or balancing. If either selected arm lacks a class, keep the
   manifest and report that fitting/selection cannot proceed; no replacement
   sampling is permitted by v1.

The 20 source records with incomplete business identities (13 TRAIN, seven TEST)
remain a known audit gap. Retain otherwise eligible components and flag them;
absence of a complete identity is not evidence of independence. This protocol
claims disjointness only under the registered native/exact-overlap indicators.
Fuzzy/partial overlap, paraphrases, canonical document identities, HaluBench
source overlap and baseline training exposure remain unverified. Do not claim
strict document disjointness, including between the two development arms.

## Manifest and integrity requirements

The next implementation must produce a separate private manifest containing:

- Design hash and allocator code revision, plus every input/report hash above.
- All row IDs and source/component assignments across all four partitions;
  selected component ranks; explicit exclusion reasons and audit limitations.
- Exact row and component counts, membership hashes, and disjointness checks.
- Separate model-input records containing only ID, answer and context; ID is
  a local routing key and is not included in the judge prompt.
- Separate offline label/metadata records. No TEST predictions or label access.

Hash manifests with the existing canonical JSON `content_hash` convention.
Write atomically, allow identical replay, and reject overwrite by differing
content. Preserve the audit reports and original reservations. Publish aggregate
counts and hashes; keep raw answer/context content and complete manifests private.
The selector must make zero HTTP requests and zero generation calls.

## Freeze sequence after reservation

This design does not select a threshold, fit calibration, or change the candidate.
Keep Qwen3-32B at its pinned revision, BF16/non-thinking, the existing label-score
prompt/hash and primary A=supported/B=unsupported mapping. The swapped mapping
remains diagnostic only; evidence-prompt tuning remains paused.

After validating the manifest, preregister the exact calibration objective,
regularization/bounds, optimizer and failure policy, and the threshold objective,
candidate/tie rules and comparator. Commit these before collecting scores for
these arms. The calibration arm must not select the operating threshold; the
threshold arm must not fit or choose among calibrators. In-sample fit quality is
not independent evidence of calibration quality. Neither arm is prompt-tuning
or model-selection data under this design.

Then perform formatted token-length audits and commit a bounded scoring plan
with exact IDs, token caps, attempt/time budgets, server identity and immutable
run namespaces. Length failures or inference failures do not license replacing
selected examples. Freeze their handling before scoring; report missingness.
No scoring allowance is inherited from completed 50-example pilot runs.

Before held-out evaluation, finish the separate evaluation manifest with metric
and bootstrap definitions, processed-test identity alignment and remaining
cross-benchmark provenance checks. Transfer the frozen prompt/calibrator/threshold
to the existing canonical HaluBench 8k split. Do not create a new HaluBench split
or tune on test outcomes. Any scope amendment must be explicit and versioned.
