# Post-thesis legacy-baseline provenance audit — 2026-09-20

Source: operator-supplied console output, summarized in the adjacent
[JSON record](baseline_provenance_cluster_20260920.json). The complete private
report and individual checkpoint fingerprints have not been supplied to the
assistant; its full report hash has not been independently recomputed here.
These are provenance findings, not new thesis results or TEST performance.

| Check | Reported result |
|---|---|
| S4 OOF TRAIN / final TEST | 15,090 / 2,700 valid scores |
| MiniCheck-7B TRAIN / TEST | 15,090 / 2,700 valid scores |
| S2 TRAIN / TEST | 15,090 / 2,700 valid scores |
| Missing scores | 0 in all six files |
| Label and metadata alignment | Passed for all six files |
| S4 stored OOF assignments | Matched; 3,018 examples in each of five folds |
| Current S4 weights | Present in final and all five fold directories |
| Legacy answer/context identity | Unverified: no stored input hashes |
| Independent checkpoint training exclusion | Unproven |

Both historical TRAIN thresholds reproduce exactly, including their comparison
operators and floating-point values:

| Baseline | Rule | TP | FP | FN | TRAIN F1 |
|---|---|---:|---:|---:|---:|
| S4 | unsupported score `>= 0.55` | 5,147 | 2,085 | 1,574 | 0.737762488353759 |
| MiniCheck-7B | support score `< 0.20000000000000004` | 5,237 | 1,976 | 1,484 | 0.7516865221759724 |

Do not round the MiniCheck comparison to `< 0.2`; the historical float grid
places the selected threshold just above 0.2.

The final S4 directory contains four fingerprinted files, and each fold directory
contains seven. These snapshots identify currently saved files, without proving
that those files generated the historical scores. Matching fold assignments
likewise cannot prove which examples actually trained each checkpoint.

Operator-reported private report SHA256:
`2e85cc46aeb4ade600c3c70680c200338f580dd20875a1a122d71f0988575e7c`.
Code revision: `84a9b48158476da32844d9679bce52b9e28cdbbd`.

No TEST metrics, new transformer calls or judge fitting occurred. Historical
input identity, checkpoint linkage and evidence visibility remain open. The next
[CPU fusion reconstruction](../../../post_thesis/llm_judge/FUSION_RECOVERY.md)
preserves these limitations and the original two-feature TRAIN procedure.
