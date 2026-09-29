# Post-thesis frozen HaluBench findings — 2026-09-29

These experiments were conducted after thesis submission and are not part of the
submitted thesis results. The canonical 8,000-example evaluation and cached replay
completed with identical reported hash:
`6919ae8e3d137fc40803a351a15216628d8df2b89fd3c9b2d5908a1fd2e571d0`.

The [machine-readable record](halubench_frozen_evaluation_20260929.json) preserves
all supplied aggregate metrics, paired intervals and 20 source/system rows.
It is a public transcription of operator output, not the full hashed private
report. No independent recomputation of that private report's hash is claimed.

## Main result

The frozen Qwen judge ranks below all three fresh-runtime baselines on RAGTruth,
but above them on aggregate HaluBench AUROC and F1. This is a benchmark-dependent
ranking reversal for this configuration, not evidence that prompted LLM judges
universally outperform trained verifiers.

| System | RAGTruth AUROC | HaluBench AUROC |
| --- | ---: | ---: |
| Qwen3-32B judge | 0.7609 | 0.8172 |
| S4 | 0.8470 | 0.5272 |
| MiniCheck-7B | 0.8776 | 0.7933 |
| S2+S4, metadata-free | 0.8494 | 0.5319 |

The RAGTruth values are from the separately recorded [fresh-runtime frozen
comparison](ragtruth_frozen_evaluation_20260928.md). These are distinct populations;
the table describes a ranking reversal, not a paired cross-dataset effect estimate.
All HaluBench thresholds and judge calibration transfer unchanged from RAGTruth
TRAIN. No HaluBench labels were used for fitting or operating-point selection.

## Aggregate HaluBench metrics

All four systems share 8,000 examples and 7,198 exact-passage bootstrap components.
The positive class is unsupported/hallucinated. AP means sklearn average precision.

| System | AUROC | AP | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B judge | 0.8172 | 0.8405 | 0.7572 | 0.7233 | 0.7945 | 0.7452 |
| S4 | 0.5272 | 0.5025 | 0.4148 | 0.4974 | 0.3558 | 0.4981 |
| MiniCheck-7B | 0.7933 | 0.8344 | 0.7221 | 0.8347 | 0.6362 | 0.7551 |
| S2+S4, metadata-free | 0.5319 | 0.5142 | 0.3622 | 0.5015 | 0.2835 | 0.5009 |

For judge minus MiniCheck-7B, the registered 2,000 paired component-bootstrap draws
produce the following nominal, unadjusted 95% intervals:

- AUROC: +0.0239, interval [+0.0138, +0.0332].
- F1 at the frozen transferred rules: +0.0351, interval [+0.0237, +0.0469].
- AP: +0.0061, interval [-0.0032, +0.0150]; the interval includes zero.

Judge-minus-S4 and judge-minus-fusion AUROC/AP/F1 intervals also exclude zero in
the positive direction. Every supplied paired interval has 2,000 valid draws.
MiniCheck retains higher aggregate precision and accuracy than the judge. Its
recall is lower under the transferred threshold. The AP interval does not support
claiming an established AP advantage over MiniCheck.

## Source heterogeneity

These source AUROCs are descriptive; no source-specific confidence intervals
were computed and no source-specific thresholds were fitted.

| Source | n | Qwen judge | S4 | MiniCheck-7B | Metadata-free fusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| DROP | 560 | 0.4474 | 0.4668 | 0.5110 | 0.5050 |
| FinanceBench | 573 | 0.6036 | 0.5177 | 0.5766 | 0.5105 |
| covidQA | 559 | 0.8064 | 0.5420 | 0.9157 | 0.5842 |
| halueval | 5728 | 0.8598 | 0.5383 | 0.8085 | 0.5396 |
| pubmedQA | 580 | 0.8697 | 0.5158 | 0.8479 | 0.5153 |

`halueval` contributes 5,728/8,000 examples (71.6%). Aggregate performance therefore
depends heavily on this source mix. Pooled AUROC is not a weighted average of
within-source AUROCs; the table must accompany the aggregate result.

The judge's DROP AUROC is below 0.5 in this sample; its FinanceBench AUROC is only
0.6036. MiniCheck has higher descriptive AUROC on DROP and covidQA. The judge has
higher descriptive AUROC on FinanceBench, halueval and pubmedQA. On pubmedQA,
MiniCheck nevertheless has higher F1 (0.7928 versus 0.7490) and accuracy under the
transferred operating rules. Ranking quality and threshold behavior differ.

This run does not demonstrate robust detection across every HaluBench source.
DROP, FinanceBench and covidQA disagreements are priorities for the next offline
error review, without changing this evaluated judge or its thresholds.

## Calibration transfer

| System / probability view | ECE (lower is better) | Brier (lower is better) |
| --- | ---: | ---: |
| Qwen3-32B judge (RAGTruth-calibrated) | 0.0706 | 0.1758 |
| S4 (raw) | 0.2894 | 0.3468 |
| MiniCheck-7B (raw) | 0.1792 | 0.2041 |
| S2+S4, metadata-free (raw) | 0.2500 | 0.3211 |

For the judge, RAGTruth-fitted calibration reduces aggregate ECE from 0.1364 to
0.0706 and Brier from 0.1896 to 0.1758. The calibrated judge has lower aggregate
ECE/Brier than the raw baseline views, with paired intervals excluding zero.
This comparison does not give baselines equivalent post-hoc calibration and
should not be described as an intrinsic calibration advantage of LLMs.

Calibration transfer is not uniformly beneficial by source: on pubmedQA, judge
ECE increases from 0.1513 to 0.1792 and Brier from 0.1792 to 0.1912. The frozen
map remains unchanged; do not recalibrate it after inspecting these TEST results.

## Protocol and practical limits

- The original canonical split is preserved. Its 7,838 TEST groups merge into
  7,198 exact-passage bootstrap components. Exact passages link 994 TEST rows to
  adaptation components, but no adaptation training is used. Strict document
  disjointness is not asserted.
- Fresh input hashes and replay support this runtime comparison. Historical
  training/checkpoint provenance is not retroactively proven. Fresh/legacy
  MiniCheck drift documented on RAGTruth remains a reproduction limitation.
- Evidence windows differ: S4 truncates 850 contexts; S2 preserves 1,886 original
  empty-pair raw-zero features and 35 truncated context pairs. MiniCheck and Qwen
  fit all their respective prompts. This is not a matched-context-capacity ablation.
- Qwen3-32B and MiniCheck-7B differ in scale and execution. Reported timing is
  descriptive, not a controlled speed comparison or a cost-equivalence claim.
- Judge scoring used a preserved 7,652-score parent and a 348-score continuation:
  8,001 attempts yielded 8,000 valid scores. One original interruption retains
  unknown token usage. Known token totals are not complete billing data.

## Next step

The benchmark evaluation phase is complete. Next, perform a bounded CPU-only
paired disagreement review using the already saved predictions and canonical
inputs. Report source composition, false positives/negatives and selected evidence
examples; preserve disagreements with benchmark annotations. No additional GPU
inference is needed for that review. An optional cascade remains separate work
requiring its own development selection and validation; these TEST results must
not be used to choose an allegedly optimal escalation policy.
