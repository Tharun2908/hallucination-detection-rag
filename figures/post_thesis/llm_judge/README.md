# Post-thesis frozen result figures

These figures describe the post-submission extension, not submitted-thesis results.

- `frozen_results_overview.svg`: AUROC and frozen-rule F1 point estimates for all
  four systems on each benchmark; no cross-dataset paired effect is implied.
- `paired_frozen_differences.svg`: judge-minus-baseline AUROC/F1 differences with
  registered nominal 95% intervals from 2,000 paired group-bootstrap draws.
  Positive values favor the judge. Intervals are unadjusted and condition on
  fitted systems; no training variability or baseline-vs-baseline test is implied.

Source files are the committed RAGTruth and HaluBench frozen evaluation JSON
summaries in `results/post_thesis/llm_judge`. Their canonical content hashes are
checked by `post_thesis/llm_judge/plot_frozen_results.py`. Those public point
metrics were compared exactly with the full uploaded reports on 2026-11-02.

Associated frozen evaluation run IDs:

- RAGTruth: `ragtruth-frozen-four-system-evaluation-v1`.
- HaluBench: `halubench-frozen-four-system-evaluation-v1`.

AUROC uses the frozen ranking scores; F1 uses frozen TRAIN/development decision
rules. Both plots include all four systems on all 2,700 RAGTruth / 8,000 canonical
HaluBench TEST rows. The HaluBench continuation supplies the interrupted row's
valid score exactly once. Context visibility and unequal training exposure remain
limitations documented in the final overview; these are not matched-evidence
comparisons. No raw/calibrated probability comparison is plotted here.

Do not replace submitted-thesis figures. Future figures must likewise identify
their post-thesis status, associated run IDs, evidence visibility and any excluded
failures. See [the experiment README](../../../post_thesis/llm_judge/README.md).

## Regenerate on CPU

The SVGs are committed; regeneration is optional. Use a separate plotting
virtual environment to avoid changing the pinned inference or fitting packages.
For example, with Python 3.12:

```bash
python -m venv .venv-judge-plots
```

Activate it (`source .venv-judge-plots/bin/activate` on Linux or
`.venv-judge-plots\Scripts\activate` in Windows CMD), then:

```bash
python -m pip install matplotlib==3.10.8
python -m post_thesis.llm_judge.plot_frozen_results
```

An alternative destination is supported with `--output-dir PATH`. The script
reads published aggregates only: no inference, raw input access, new metrics,
bootstrap sampling, fitting or threshold changes. Its fixed SVG salt and omitted
creation timestamp allow repeatable SVG output in the same plotting environment.
Cross-platform/font/backend changes can affect rendering bytes without changing
plotted data.

See the [final overview](../../../results/post_thesis/llm_judge/FINAL_OVERVIEW.md)
for source composition and limitations. No timing speedup chart is provided
because the runs did not share a controlled measurement setup.
