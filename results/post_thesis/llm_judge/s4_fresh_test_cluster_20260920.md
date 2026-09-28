# Post-thesis fresh S4 TEST run and legacy score agreement

These are operator-supplied cluster observations from 2026-09-20, recorded
separately from thesis results. They contain no label-based TEST metrics.
The report author did not independently run the weights or receive the private
per-example inference journal.

## Fresh run

All 2,700 examples completed in 169 batches, with no terminal failures or pending
rows. The runner reported answer truncation for 228 examples (8.44%) and context
truncation for 1,740 (64.44%); the overlap between these counts was not supplied.
This matters when comparing S4's visible evidence with the judge's full inputs.

Known batch time was 33.94294817419723 seconds and CUDA-forward time was
18.275184537749738 seconds, with zero unknown-timing batches. These are shared-GPU
measurements excluding model loading, not a controlled speed comparison.

Fresh report SHA256:
`4f502fc775f15332eb78aa5070ef2d056015241d91a67f5c1ceb254ba2f2dea6`.
Private path: `s4-ragtruth-test-fresh-v1/summary.json` under the post-thesis
artifact root. The run did not compute TEST metrics, call the judge, fit models,
read HaluBench or overwrite legacy files.

## Agreement with preserved legacy scores

The [exact supplied agreement record](s4_legacy_score_agreement_20260920.json)
compares all 2,700 fresh scores with the legacy S4 cache. Its canonical report
hash was independently recomputed from the supplied values.

| Observation | Value |
| --- | ---: |
| Fresh scores matching legacy after rounding to four decimals | 2,692 / 2,700 |
| Mean absolute difference | 0.000025352895480062705 |
| Median absolute difference | 0.000025062417984000884 |
| Maximum absolute difference | 0.00005245294570921377 |
| Differences greater than 0.001 | 0 |
| Differences greater than 0.01 | 0 |

Agreement is close. The eight four-decimal mismatches are consistent with small
numerical differences near rounding boundaries; these aggregates do not identify
their cause or establish threshold-decision agreement. No further S4 tuning is
motivated by this result. Preserve fresh input/weight/visibility records for the
post-thesis comparison. Numerical agreement does not prove the historical
cache-to-input/checkpoint linkage or original training exclusions.

Next: fresh S2 features for the fixed metadata-free fusion. MiniCheck provenance
and the final benchmark comparison gate remain pending.
