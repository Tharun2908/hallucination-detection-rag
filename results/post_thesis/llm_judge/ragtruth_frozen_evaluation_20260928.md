# Post-thesis frozen RAGTruth TEST findings

These experiments were conducted after thesis submission and are not part of the
submitted thesis results. Recorded from the operator's cluster output on
2026-09-28; the [JSON summary](ragtruth_frozen_evaluation_20260928.json) preserves
all printed point metrics and selected paired intervals at their reported precision.

## Result

On the shared 2,700 RAGTruth TEST responses in 450 exact-overlap groups, the frozen
Qwen3-32B judge trails all three baselines on AUROC, average precision and F1.
MiniCheck has the highest point estimates for those three metrics; metadata-free
fusion has the lowest reported Brier score and ECE.

| System | AUROC ↑ | Average precision ↑ | F1 ↑ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen3-32B judge | 0.7609 | 0.6320 | 0.6123 | 0.0868 | 0.1916 |
| S4 DeBERTa | 0.8470 | 0.7726 | 0.7024 | 0.1289 | 0.1728 |
| MiniCheck-7B (fresh runtime) | 0.8776 | 0.8094 | 0.7304 | 0.2719 | 0.2272 |
| S2+S4 (metadata-free) | 0.8494 | 0.7664 | 0.7065 | 0.0547 | 0.1486 |

**Reliability columns use the TRAIN-calibrated judge and raw baseline scores.**
This is not a comparison in which all systems received the same calibration
procedure. Average precision is the registered AP calculation, not trapezoidal
PR area. F1 uses each system's frozen TRAIN/development operating rule.

The frozen judge calibration improves TEST ECE from 0.1366 to 0.0868 and Brier
score from 0.2051 to 0.1916. It does not change ranking or the frozen raw-margin
classification rule. The calibrated judge has lower ECE than raw S4 and raw
MiniCheck, but S4 has a lower Brier score. Fusion has lower ECE and Brier than
the calibrated judge. A lower ECE alone does not establish better detection.

## Paired uncertainty

The registered 2,000 whole-group bootstrap draws use the same sampled groups for
all systems and never refit a model or threshold. All printed intervals have
2,000 valid draws. Differences below are **judge minus baseline**; negative
AUROC/F1 differences favor the baseline.

| Baseline | AUROC difference [nominal 95% interval] | F1 difference [nominal 95% interval] |
|---|---:|---:|
| S4 DeBERTa | -0.0861 [-0.1076, -0.0660] | -0.0901 [-0.1161, -0.0651] |
| MiniCheck-7B (fresh runtime) | -0.1166 [-0.1368, -0.0968] | -0.1181 [-0.1440, -0.0930] |
| S2+S4 (metadata-free) | -0.0885 [-0.1093, -0.0693] | -0.0942 [-0.1201, -0.0700] |

Every judge-minus-baseline AUROC, AP and F1 interval excludes zero in the
baseline's favor. These are nominal, unadjusted intervals conditional on the
fixed systems and group sample; they do not include retraining uncertainty.
The printed comparisons do not establish significance between MiniCheck, S4
and fusion themselves. The JSON also retains the printed reliability differences
and identifies the calibrated-judge-versus-raw-baseline score spaces.

## Frozen configuration and scope

- Judge: pinned Qwen3-32B, BF16, non-thinking, prompt
  `faithfulness-label-score-v1`; primary A=supported / B=unsupported mapping.
  Ranking uses raw class-token log odds, not self-reported probabilities.
- Calibration remains `sigmoid(0.3808096416654668 * margin + 0.3575745057816423)`.
  The operating rule remains raw margin `>= -1.25`.
- Baseline thresholds remain S4 `>= 0.55`, MiniCheck support
  `< 0.20000000000000004`, and metadata-free fusion `>= 0.45`.
- Fresh input/checkpoint linkage supports this runtime comparison. It cannot
  retrospectively prove historical training membership or legacy input provenance.
- MiniCheck's fresh runtime changed 100/2,700 fixed-threshold decisions relative
  to its legacy cache. The cause remains unresolved. Its historical TRAIN
  threshold is transferred unchanged; these are not exact thesis-cache results.
- Training exposure and evidence visibility differ. S4 truncates 228 answers
  and 1,740 contexts; the judge inputs all fit its window. This comparison
  therefore does not isolate model architecture or model size as a cause.
- Exact-overlap components do not establish strict document disjointness.
  Timing scopes differ across systems, so these findings make no normalized
  speedup or monetary-cost claim.

This is a negative result for **this frozen judge configuration on RAGTruth**.
It is not a conclusion about all prompted LLM judges, and it does not replace
submitted thesis results. No model, prompt, calibration or operating rule is
changed in response to these TEST metrics.

## Provenance and next stage

The operator reported private evaluation SHA256:

`7bdf3e9c5de63e585a07044f7e33b14f960490c01b1e682f08494da9148d3aac`

Private report:

```
.artifacts/post_thesis/llm_judge/ragtruth-frozen-four-system-evaluation-v1/report.json
```

This public record transcribes the supplied console summary. It does not contain
the full private report or independently verify its hash. Full reliability tables,
per-system intervals, descriptive slices and efficiency details remain in that
private report; no omitted values are inferred here. The evaluation performed
no model calls, fitting, threshold changes or HaluBench reads.

The next planned stage is cross-domain evaluation on the existing canonical
**group-disjoint 8,000-example HaluBench TEST set**, using the same frozen judge,
calibration and operating rules. Reuse the saved split; do not create another
split, use target labels to retune, or select a new judge after seeing RAGTruth
TEST performance. Disagreement analysis follows the benchmark comparison;
the cascade remains optional.
