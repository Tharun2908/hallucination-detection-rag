# Post-thesis fresh baseline comparison manifest

These experiments are outside the submitted thesis. The frozen judge and
numerical evaluation contract remain unchanged.

## Decision before judge TEST inference

Use the complete fresh S4, fresh MiniCheck-7B and frozen metadata-free fusion
applied to fresh S2/S4 features as the primary post-thesis baselines. Choose
these artifacts for their recorded current input/checkpoint linkage, not on
TEST label performance. Preserve the legacy caches separately. Do not mix
fresh and legacy predictions or select whichever produces a preferred result.

The MiniCheck runtime variant is **not an exact reproduction of the thesis
cache**: 100/2,700 decisions differ at the unchanged historical TRAIN threshold.
Its cause is unresolved. See the [recorded comparison](../../results/post_thesis/llm_judge/minicheck_legacy_score_agreement_20260928.md).

| System | Fixed policy | Interpretation |
| --- | --- | --- |
| S4 | Unsupported score >= 0.55 | Historical TRAIN-derived threshold applied to fresh inference |
| MiniCheck-7B | Support score < 0.20000000000000004 | Transferred historical TRAIN threshold; not optimized for the fresh runtime |
| Metadata-free S2+S4 | Unsupported score >= 0.45 | Frozen historical TRAIN reconstruction applied to fresh features |
| Qwen3 judge | Raw margin >= -1.25 | Frozen disjoint development selection; TEST predictions pending |

The baseline training and threshold-selection evidence has limits. Fresh
inference establishes the current checkpoint, input, tokenization and score
association; it does not independently establish which examples trained the
historical checkpoints. Historical TRAIN caches still lack answer/context
hashes, and fold tags do not independently prove training exclusions. Numerical
agreement cannot repair these gaps.

`comparison_ready` in this new manifest means ready for this **documented fresh
runtime comparison**. Its provenance flags concern current inference artifacts
and the identified fixed historical TRAIN operating policies. It does not
supersede the original reports' historical-provenance warnings. The manifest
also sets `historical_training_provenance_independently_verified=false` and
`exact_thesis_cache_reproduction_claimed=false`. Judge readiness remains false,
so the unchanged all-system evaluation gate stays blocked.

## CPU command

After committing and pulling this change on the pod:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
python -S -m post_thesis.llm_judge.freeze_comparison
```

This uses the existing private reports. It checks their pinned content hashes,
all 2,700 sample IDs and answer/context hashes, order, complete valid scores,
canonical 450-component grouping, frozen fusion dependency/model linkage and
unchanged TRAIN threshold sources. Ground-truth labels and benchmark metadata
are not projected into comparison score records. Canonical manifest integrity
verification does read the existing manifest containing offline labels; no
label metrics or selection is performed.

Output:

```
.artifacts/post_thesis/llm_judge/ragtruth-fresh-comparison-manifest-v1/manifest.json
```

A replay at the same code revision verifies the identical manifest without
rewriting it. Changed or missing pinned sources fail closed; never edit old
reports to make this check pass. This command makes no model calls, refits,
metrics, downloads or package changes. It grants no generation allowance.

## Reporting limits retained

- S4 has a 512-token pair limit (228 answers and 1,740 contexts truncated).
  S2 uses sentence filtering and pair truncation (159 context pairs truncated).
  MiniCheck splits/chunks evidence with no truncated prompts in this run.
  The judge uses complete answer/context inputs. This evaluates practical
  systems with different evidence visibility, not architecture alone.
- Baselines and judge have different training/development exposure. Baseline
  raw probability metrics and judge raw/calibrated metrics must be identified.
- Existing timing scopes and GPU sharing differ. Do not claim a controlled
  speedup from these records.
- Grouping covers registered exact overlaps; strict document disjointness
  remains unproven. No new group split is made.

Next: record a bounded judge TEST execution plan, restore/check the pinned
Qwen serving profile, run the frozen judge, and then evaluate the aligned
systems with the registered metrics. HaluBench remains a later separate stage.
