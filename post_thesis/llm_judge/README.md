# Post-thesis LLM judge for RAG faithfulness

> These experiments were conducted after thesis submission. They are separate
> from the submitted thesis results and do not replace them.

**Status — 2026-11-02: complete.** The judge, calibration and operating threshold
are frozen. RAGTruth and canonical HaluBench inference, evaluation, cached replay,
selected qualitative review and runtime accounting are complete. The frozen
release is tagged **`post-thesis-llm-judge-v1`** at commit `c1b1cd6`.
No further inference or fitting is required for this release.

## Start here

| What you need | Document |
| --- | --- |
| Findings, figures and interpretation | [Final overview](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md) |
| All public results and development records | [Results index](../../results/post_thesis/llm_judge/README.md) |
| Run guides and their completed outcomes | [Documentation index](DOCUMENTATION_INDEX.md) |
| Frozen scoring and evaluation rules | [Continuous-score protocol](CONTINUOUS_SCORE_PROTOCOL.md), [evaluation protocol](EVALUATION_PROTOCOL.md) |
| Selected-case qualitative analysis | [127-case review](../../results/post_thesis/llm_judge/qualitative_review_20261102.md) |
| Timing, token usage and accounting limits | [Verified accounting](../../results/post_thesis/llm_judge/efficiency_accounting_20261102.md) |

## Research question and result

Under a fixed evaluation protocol, when does a prompted LLM judge improve
cross-domain faithfulness detection, and which failures remain shared with
trained verifiers?

The frozen Qwen3-32B judge trails all three fresh baselines on RAGTruth AUROC,
average precision and F1. It leads aggregate HaluBench AUROC and F1 with the same
prompt and decision rule. This is a benchmark-dependent ranking reversal, not a
universal advantage for either model family.

| System | RAGTruth AUROC | RAGTruth F1 | HaluBench AUROC | HaluBench F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-32B judge | 0.7609 | 0.6123 | 0.8172 | 0.7572 |
| S4 | 0.8470 | 0.7024 | 0.5272 | 0.4148 |
| MiniCheck-7B | 0.8776 | 0.7304 | 0.7933 | 0.7221 |
| S2+S4 (metadata-free) | 0.8494 | 0.7065 | 0.5319 | 0.3622 |

RAGTruth contains 2,700 shared responses in 450 groups. HaluBench contains 8,000
shared responses in 7,198 exact-passage bootstrap components. Each evaluation
uses 2,000 paired group-bootstrap draws and nominal, unadjusted 95% intervals.
F1 uses fixed RAGTruth TRAIN/development operating rules.

MiniCheck has higher HaluBench accuracy and precision; the judge-minus-MiniCheck
average-precision interval includes zero. HaluBench is 71.6% halueval, and DROP
remains a clear judge weakness. See the [final overview](../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md)
for paired intervals, source-level results and full interpretation.

## Frozen judge

| Setting | Recorded choice |
| --- | --- |
| Model | `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137` |
| Serving | BF16, non-thinking, one H200, vLLM 0.29.0, concurrency one |
| Prompt | `faithfulness-label-score-v1`; answer and context only |
| Class labels | A = supported (token 32); B = unsupported (token 33) |
| Ranking score | Unrounded `logP(B) - logP(A)` at the same assistant position |
| Calibrated score | `sigmoid(0.3808096416654668 * margin + 0.3575745057816423)` |
| Unsupported decision | Raw margin `>= -1.25` |
| Development roles | 600 calibration rows and 600 threshold rows; 100 components per arm |
| Transfer | Same prompt, calibration and threshold on canonical HaluBench; no target-domain adaptation |

Exact identities are in [the frozen configuration](configs/frozen_judge_v1.json).
Its execution-pending fields describe the original decision freeze and are
preserved. Later execution and evaluation reports establish completion.
The early JSON-probability and structured-evidence candidates were development
experiments; they are not the final benchmark scoring method.

## Completed study stages

| Stage | Outcome and record |
| --- | --- |
| Probability, binary and evidence development | Completed with mixed findings; [historical diagnostics](DOCUMENTATION_INDEX.md#historical-diagnostics) |
| Native-source and cross-split checks | Completed scoped exact-overlap audits; [source audit](SOURCE_AUDIT.md), [cross-split audit](CROSS_SPLIT_AUDIT.md) |
| Development reservation, scoring and fitting | Completed; 312 TRAIN rows excluded and two 600-row arms frozen; [accepted fit](../../results/post_thesis/llm_judge/development_fit_cluster_20260920.md) |
| Fresh baseline inference and fusion | Completed on both benchmarks; [RAGTruth comparison manifest](COMPARISON_MANIFEST.md), [HaluBench execution record](HALUBENCH_JUDGE_RUN.md) |
| RAGTruth judge and evaluation | 2,700/2,700 scores; [frozen findings](../../results/post_thesis/llm_judge/ragtruth_frozen_evaluation_20260928.md) |
| HaluBench judge and evaluation | 8,000/8,000 scores across parent and continuation; [frozen findings](../../results/post_thesis/llm_judge/halubench_frozen_evaluation_20260929.md) |
| Qualitative review | 127 selected cases reviewed; [confirmed findings](../../results/post_thesis/llm_judge/qualitative_review_20261102.md) |
| Accounting, figures and release | Completed; [accounting](../../results/post_thesis/llm_judge/efficiency_accounting_20261102.md), [figures](../../figures/post_thesis/llm_judge/README.md), tag `post-thesis-llm-judge-v1` |

## Limits retained in the release

- **Legacy provenance:** fresh inference establishes current input/score alignment,
  not historical checkpoint training membership or cache-to-model linkage.
  MiniCheck changes 100/2,700 RAGTruth decisions relative to its legacy cache at
  the fixed threshold; the cause remains unresolved.
- **Evidence visibility:** S4 truncates contexts; S2 preserves 1,886 HaluBench
  empty-pair examples using its original raw-zero fallback. This is a comparison
  of practical systems, not a matched-evidence or architecture-only ablation.
- **Overlap and exposure:** exact-overlap grouping does not prove strict document
  disjointness. Exact passages link 994 HaluBench TEST rows to adaptation
  components; no adaptation training is used in this extension. Foundation-model
  exposure and historical training provenance remain limited.
- **Qualitative scope:** the 127-case review contains AI-assisted judgments
  collectively confirmed by the researcher before unblinding: 28 supported,
  58 unsupported and 41 uncertain. The 22 disagreements among 86 determinate
  cases are not independent annotation, corrected labels or a population error rate.
- **Accounting:** HaluBench retains one interrupted attempt with unknown usage
  and latency. Monetary cost is unknown. Differing timing scopes do not support
  controlled speedup claims.

These are reported limitations, not unfinished runs. A cascade, new prompt,
recalibration, exhaustive pilot adjudication or additional ablation would be a
separate study; none is needed to reproduce this frozen release.

## Reproduction and preservation

Reading results, regenerating figures and replaying completed evaluations are
CPU tasks. New model inference would require the recorded model/runtime resources,
but is unnecessary for the completed release.

| Task | Guide | Required artifacts |
| --- | --- | --- |
| Regenerate the two figures | [Figure README](../../figures/post_thesis/llm_judge/README.md) | Committed aggregate JSON; Matplotlib 3.10.8 |
| Replay RAGTruth evaluation | [TEST_EVALUATION.md](TEST_EVALUATION.md) | Preserved private inputs, predictions, fit and evaluation records |
| Replay HaluBench evaluation | [HALUBENCH_EVALUATION.md](HALUBENCH_EVALUATION.md) | Preserved parent, continuation, baseline and evaluation records |
| Inspect review procedure | [DISAGREEMENT_REVIEW.md](DISAGREEMENT_REVIEW.md) | Public aggregates; private packet/key for case-level review |
| Inspect the original JSON runner | [LEGACY_INTERFACE.md](LEGACY_INTERFACE.md) | Historical engineering reference, not the final scorer |

Run guides preserve their original budgets, commands and revision requirements.
Consult each guide's completion note before using a historical command. Do not
rerun generation, rename runs, reset budgets or refit to reproduce a cached report.
The [documentation index](DOCUMENTATION_INDEX.md) distinguishes these procedures
from the current completed status.

Offline contract tests use local fakes, not live models. From the repository root:

```bash
python -S -m unittest discover -s tests -p "test_llm_judge*.py" -v
```

Passing engineering tests is not evidence of model quality. Full private replay
requires the pinned CPU environment and preserved artifacts; the public repository
alone does not contain all raw inputs and responses.

## Artifact locations and thesis boundary

| Content | Location |
| --- | --- |
| Protocols, implementation and frozen configurations | `post_thesis/llm_judge/` |
| Public aggregates and dated observations | [results/post_thesis/llm_judge/](../../results/post_thesis/llm_judge/README.md) |
| Public figures | [figures/post_thesis/llm_judge/](../../figures/post_thesis/llm_judge/README.md) |
| Private input manifests, journals and raw responses | `.artifacts/post_thesis/llm_judge/<run_id>/` by default |
| Existing comparison-artifact identities | [baseline_manifest.json](baseline_manifest.json) |

Keep private benchmark text, review keys and credentials out of Git. Preserve
private artifacts outside disposable pod storage, including both HaluBench runs.
Existing thesis results and dated machine-readable records remain unchanged.
The implementation baseline is not a verified thesis-submission snapshot;
`post-thesis-llm-judge-v1` identifies this extension, not the original submission.
