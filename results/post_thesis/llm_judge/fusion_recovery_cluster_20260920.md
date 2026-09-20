# Post-thesis metadata-free fusion reconstruction — 2026-09-20

The operator reports successful CPU reconstruction and an identical cached
replay. These are **post-thesis reconstruction findings**, not new thesis
results or measured TEST performance. The [JSON observation record](fusion_recovery_cluster_20260920.json)
transcribes the supplied console summaries. The private full report and 2,700
per-example predictions have not been independently received by the assistant.

| Item | Reported result |
|---|---|
| Features | S2 minimum relevance and S4 unsupported score, in that order |
| Task/generator features | None |
| Initial CPU fits | 5 meta-OOF folds + 1 full-TRAIN logistic regression |
| TRAIN meta-OOF threshold | `>= 0.45`, hex `0x1.ccccccccccccdp-2` |
| TRAIN meta-OOF F1 | 0.7379856115107913 |
| TRAIN TP / FP / FN | 5,129 / 2,050 / 1,592 |
| Historical TRAIN reference | Matches after the original four-decimal rounding |
| Saved TEST predictions | 2,700 |
| Replay | Zero new fits; same report hash |

The fitted logit is `-1.3590096505049223 * s2 + 3.155528234679687 * s4
- 1.4168645142264538`; sigmoid gives the unsupported score. S2 is the original
clipped normalization using minimum -11.430 and maximum 10.641. The model
converged in eight iterations. Exact settings are retained in the JSON record.

Report SHA256:
`1c90a003abbda9044767bb9167b1ecc171f7a8a746b87093a143d833ba4ec0d6`.
Code revision: `538e4d6814e8193aa79a9e84e95de910acf016f3`.
The initial run and replay reported the same values. This hash is operator
reported; the next pod auditor checks the preserved full report against it.

No TEST metrics, transformer inference or judge refitting occurred. A matching
rounded reference and repeatable reconstruction do not establish exact historical
predictions, historical cache input text, checkpoint-to-cache linkage or evidence
visibility. The reconstruction remains explicitly limited; `comparison_ready`
remains false. TRAIN meta-OOF F1 is not TEST F1 or fully nested end-to-end validation.

Proceed with the [frozen judge TEST token audit](../../../post_thesis/llm_judge/TEST_TOKEN_AUDIT.md).
That independent length check does not waive the evaluation protocol's baseline
provenance requirements or authorize generation.
