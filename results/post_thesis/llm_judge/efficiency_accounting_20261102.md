# Post-thesis inference timing and cost accounting

Recorded 2026-11-02 from the full frozen evaluation reports. This extension is
separate from the submitted thesis. Both canonical report hashes were recomputed
and matched their previously recorded values. All published aggregate point
metrics match the uploaded reports exactly. No inference, fitting, threshold
selection or metric recomputation was performed.

The [machine-readable extract](efficiency_accounting_20261102.json) preserves
both complete `efficiency` objects, exact source hashes and verification scope.
This verifies report integrity; it does not independently reproduce model outputs
or resolve the retained historical training-provenance limitations.

## Qwen request latency and client accounting

| Measurement | RAGTruth | HaluBench |
| --- | ---: | ---: |
| Valid examples | 2,700 | 8,000 |
| Request latency mean (ms) | 138.25 | 114.65 |
| Request latency median (ms) | 133.10 | 75.90 |
| Request latency p95 (ms) | 207.98 | 426.73 |
| Successful-request latency sum (s) | 373.27 | 917.22 |
| Charged client time (s) | 3,535.41 | 14,979.67 |
| Charged client time per valid example (s) | 1.3094 | 1.8725 |
| Known input tokens | 3,334,607 | 7,794,485 |
| Known output tokens | 2,700 | 8,000 |
| Attempts with unknown input/output usage | 0 | 1 |

Request measurements are sequential client-request latencies, not isolated GPU
kernel time. No warmup observations were discarded. The p95 uses the frozen
implementation's linear percentile calculation.

Client time includes preflight/journal work and excludes preparation, server
startup and idle time. HaluBench combines the parent and continuation, including
the interrupted window: 8,001 attempts produced 8,000 unique valid scores. Its
latency statistics describe only the 8,000 successful requests; the interrupted
request's latency and token usage are unavailable. The known token totals are
therefore incomplete cost accounting. The gap between client time and request
sums must not be attributed entirely to GPU inference or a specific overhead
without further instrumentation.

## Baseline timing: recorded scopes

All values below are seconds. These are descriptive measurements from runs with
different workloads and timing scopes, not a controlled speed benchmark.

| Measurement | RAGTruth | HaluBench |
| --- | ---: | ---: |
| S4 batch-time sum, loading excluded | 33.94 | 69.65 |
| S4 CUDA-forward sum | 18.28 | 41.68 |
| S2 example-time sum, loading excluded | 453.96 | 130.62 |
| S2 CUDA-forward sum | 201.15 | 79.00 |
| MiniCheck example-time sum | 559.00 | 427.17 |
| MiniCheck initialization, separate | 26.60 | 97.87 |
| Fusion application time | Not measured | Not measured |

CUDA-forward sums are contained within the wider measurement scope; do not add
them to batch/example sums. S2 and S4 were measured on a shared GPU. No common
warmup, concurrency or GPU-sharing experiment was performed, so no speedup ratios
are claimed. Baseline median and p95 latencies are absent from these reports;
aggregate sums cannot reconstruct them. The frozen S2+S4 fusion uses the component
measurements above, with no measured end-to-end fusion latency.

| MiniCheck workload | RAGTruth | HaluBench |
| --- | ---: | ---: |
| Sentence requests / output tokens | 18,935 | 9,158 |
| Known input tokens | 13,661,600 | 4,905,370 |
| Truncated prompts | 0 | 0 |

MiniCheck and Qwen use different tokenizers and request decomposition; native
token counts are not interchangeable compute units.

## Evidence visibility

| Fixed preprocessing outcome | RAGTruth | HaluBench |
| --- | ---: | ---: |
| S4 examples with truncated answer | 228 | 0 |
| S4 examples with truncated context | 1,740 | 850 |
| S2 truncated answer pairs | 0 | 0 |
| S2 truncated context pairs | 159 | 35 |
| S2 empty-pair examples | Not reported in this efficiency object | 1,886 |

Example and pair counts have different denominators. HaluBench's 1,886 S2
empty-pair examples retain the original raw-zero fallback. These visibility
limits belong alongside accuracy comparisons and must not be hidden by the
aggregate timing table.

## Monetary cost and release scope

**Monetary cost is unknown, not zero.** No billing rate or billable duration was
supplied. Server lifetime is also unrecorded here. Neither client time nor
successful-request sums provide billed GPU hours; preparation, loading, startup
and idle time may contribute to billing. No currency or hypothetical rental rate
is assigned.

This record covers final TEST inference accounting. It is not a total project
cost estimate and excludes development pilots, training and unmeasured setup
work. Frozen performance findings and their limitations remain in the
[RAGTruth report](ragtruth_frozen_evaluation_20260928.md) and
[HaluBench report](halubench_frozen_evaluation_20260929.md). The
[qualitative review](qualitative_review_20261102.md) remains selected-case,
AI-assisted analysis confirmed by the researcher, not new benchmark labels.

## Verified source report identities

- RAGTruth: `7bdf3e9c5de63e585a07044f7e33b14f960490c01b1e682f08494da9148d3aac`
- HaluBench: `6919ae8e3d137fc40803a351a15216628d8df2b89fd3c9b2d5908a1fd2e571d0`

Hashes cover the `report` object serialized as sorted-key ASCII JSON with compact
separators and non-finite values disallowed, then UTF-8 encoded and SHA-256 hashed.
The source-file byte hashes are separately retained in the JSON extract. Public
historical console transcriptions remain unchanged; this dated record adds the
subsequent full-report integrity verification.
