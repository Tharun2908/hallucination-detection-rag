# Post-thesis reserved TRAIN offline fitting

This implementation is post-thesis work and does not change submitted thesis results.
The [completed scoring record](../../results/post_thesis/llm_judge/development_scoring_cluster_20260920.md)
reports 600 valid scores in each reserved arm and zero additional attempts on replay.
The actual scores remain in private pod artifacts. No real fit has yet been reported.

## Fixed method

The [preregistered fitting protocol](CALIBRATION_THRESHOLD_PROTOCOL.md) and
`configs/label_score_fit_v1.json` remain byte-for-byte unchanged. Their historical
`fitting_implemented: false` field describes the preregistration, not this implementation.

- Fit `sigmoid(a * raw_margin + b)` on the 600 calibration rows only, with the
  registered regularization, bounds, starting point, analytic gradient and SciPy options.
- Accept only a finite, interior, converged solution satisfying every frozen check.
  On numerical failure, retain diagnostics and no deployable calibrator. Do not restart
  or substitute another model. Independent raw threshold selection still completes.
- Select the raw-margin `>=` threshold on the separate 600 operating-threshold rows.
  Evaluate each distinct margin and an all-negative sentinel. Compare F1 fractions
  exactly; ties select the largest threshold. Save all candidate confusion counts
  and the selected threshold's exact hexadecimal float representation.
- Save raw and calibrated Brier/ECE diagnostics with the registered ten bins.
  Calibration-arm diagnostics are in-sample; threshold-arm diagnostics are development
  results, not an independent validation of the complete pipeline.
- Deployment decisions use the raw threshold. A transformed probability threshold
  is descriptive only. Primary ranking uses the raw margin.

No labels, task metadata or generator information are sent to a model. This command
constructs no network client, loads no model/tokenizer and makes no generation calls.
It reads neither TEST nor HaluBench. Exact-overlap component separation does not
establish strict underlying-document disjointness.

## Input verification

The fitter validates the pinned development manifest and joins labels by sample ID.
It accepts either recorded fresh or cached summary hash from scoring revision
`a341e3e805827a1122cc74dec99c66a749543572`. It checks prepared request identities,
input hashes, fixed request-reference hashes, budgets and historical server snapshots.
It opens each SQLite journal in read-only mode, verifies result checksums, reparses
raw model responses and reconstructs the entire summary before fitting. All 600
valid unique scores per arm are required. Corrupt or incomplete inputs stop the
command without imputation, replacements or a partial fit artifact.

The current fitter revision may differ from the historical scoring revision.
It does not call the historical scoring CLI, recover journals or rewrite source
summaries. The shared scoring lock prevents concurrent scoring while inputs are read.
Fresh/cached summary hashes differ because `new_attempts` is invocation-specific;
that difference does not imply another model call.

## Run on the pod after committing and pulling this change

The GPU server can remain stopped. Preserve the pod's `/workspace` artifacts.
Use a separate CPU environment so the serving packages remain reproducible:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m venv .venv-judge-fit &&
source .venv-judge-fit/bin/activate &&
python -m pip install -r post_thesis/llm_judge/requirements-fit.txt &&
python -m post_thesis.llm_judge.fit_development
```

Python 3.12, NumPy 1.26.4 and SciPy 1.14.1 are mandatory. The command records the
exact Python patch version and fitter Git revision. It requires a clean committed
checkout. No token audit, scoring run or model download needs to be repeated.

Output:

```text
.artifacts/post_thesis/llm_judge/qwen3-ragtruth-development-fit-v1/fit.json
```

Inspect the console and preserve the report. Run the same command again to verify
cached reuse:

```bash
python -m post_thesis.llm_judge.fit_development
```

An identical identity verifies and returns the saved fit without optimization or
rewriting it; its report hash remains unchanged. Changed inputs, environment or
fitter revision reject reuse. Do not delete or modify a fit to bypass this check.
A rejected numerical calibrator still produces a report and independent raw
threshold, but the command exits with status 1 to make the rejection visible.

This step freezes development outputs, not test findings. Report the console output
before preparing any benchmark evaluation. Final evaluation alignment and metrics
remain a separate step under the research protocol.
