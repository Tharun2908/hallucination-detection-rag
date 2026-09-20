# Post-thesis evaluation mathematics before TEST inference

The [TEST token audit and replay](../../results/post_thesis/llm_judge/test_token_audit_cluster_20260920.md)
are complete. This step implements the quality metrics and uncertainty rules in
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) before collecting judge TEST scores.
It is post-thesis work, not part of the submitted thesis.

## Implemented numerical contract

The [machine-readable contract](configs/evaluation_math_v1.json) pins the numerical
source files, accepted judge freeze, four system identities, historical baseline
operating rules, software versions and statistical choices. The checker rejects
changes to these files or contract. No fitting or threshold-search function is
called by the evaluator.

| Quantity | Rule |
|---|---|
| Judge ranking | Unrounded raw log odds, even when sigmoid saturates |
| MiniCheck ranking | Negative support score; avoids subtraction-induced ties |
| AUROC / average precision | sklearn implementations; AP is not trapezoidal PR area |
| Operating metrics | Frozen thresholds and original comparators, including MiniCheck's float above 0.2 |
| Judge reliability | Both raw sigmoid and frozen calibrated probabilities |
| Baseline reliability | Original unsupported probabilities; MiniCheck uses `1 - support` |
| Brier / ECE | Existing registered implementation; ten bins including 0/1 and empty-bin counts |
| Missing observations | Explicit `None`; retain attempted denominator; no imputation |
| Empty / one-class subsets | AUROC and AP null with reason; empty reliability/accuracy null |
| Zero classification denominators | F1/precision/recall zero under the registered convention |
| Descriptive slices | Every supplied task/generator/source level, including missing/one-class slices |

The primary paired comparison uses the same four-system valid intersection.
Individual valid-subset metrics retain their separate coverage. No substitution
of metadata-aware fusion or a larger favorable pairwise subset is allowed.

Bootstrap: 2,000 draws with `Generator(PCG64(20260920))`. Resample K full groups
with replacement from K sorted eligible group IDs; retain every valid member and
group multiplicity. All systems use identical resampled rows. Invalid ranking
replicates remain in the draw sequence and are counted, never replaced. Intervals
require at least 1,900 valid draws and use linear 2.5/97.5 percentiles. An empty
shared subset produces no intervals, including for classification ratios whose
point value follows the zero-division convention.

Differences are judge minus baseline. Both raw and calibrated judge Brier/ECE
are compared with the same raw baseline probabilities and named explicitly.
Higher ranking/classification scores are better; lower Brier/ECE are better.
Bootstrap group membership and draw hashes are recorded. Intervals remain nominal,
unadjusted and conditional on these fitted systems and this sample.

## Artificial-data validation

Tests cover tied rankings, sigmoid saturation, exact threshold equality, support
orientation, AP, ECE endpoints, missing scores, one-class subsets, unequal group
sizes/multiplicity, common coverage, difference signs, insufficient valid draws,
empty intersections and blocking of unverified comparisons.

The [local validation record](../../results/post_thesis/llm_judge/evaluation_math_local_20260920.json)
records 14 focused passing tests. The complete local synthetic check exercises 2,000 replicates for all four systems,
using only 12 constructed examples in six mixed-label groups. All systems have
2,000 valid AUROC draws; cached inspection performs no new evaluation. These are
implementation checks, not benchmark findings or evidence of model quality.

After applying, committing and pulling the patch, run in the existing CPU environment:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.check_evaluation_math
```

Repeat the Python command once and return the summaries. The checker requires
Python 3.12, NumPy 1.26.4, SciPy 1.14.1 and scikit-learn 1.5.2, already used for
fusion reconstruction. It writes an immutable private synthetic record to
`.artifacts/post_thesis/llm_judge/evaluation-math-synthetic-check-v1/report.json`.
It reads no benchmark prediction file, starts no server and makes no model calls.

## Scope and remaining work

`evaluation_math.py` operates on aligned in-memory arrays. Its provenance gate
requires all four systems to be comparison-ready with verified content/checkpoint/
threshold provenance and documented evidence visibility. The gate **checks status
supplied by a future validated artifact loader; it does not establish provenance**.
Synthetic readiness flags apply only to constructed fixtures. No current legacy
baseline is promoted to ready by these tests.

This freezes the numerical core. A validated benchmark artifact loader, efficiency
reporting and a separately bounded inference runner/configuration still remain.
The actual baseline evidence gaps must be resolved before main paired metrics.
Neither this implementation nor the token audit grants a scoring allowance.
There are no new TEST metrics, HaluBench reads, prompt changes or judge refits.
