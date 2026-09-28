# Post-thesis RAGTruth TEST evaluation

All four systems have complete fresh predictions. Qwen's completed run and
zero-call replay have the same report hash. This CPU-only loader applies the
[numerical contract](EVALUATION_MATH.md) frozen before judge TEST inference.
The loader is implemented after inference and before examining the new TEST
metrics; it does not change that numerical core, prompt, model or operating rules.

## Inputs and interpretation

The command verifies the canonical 2,700 inputs, 450 exact-overlap groups,
completed fresh baseline report hashes and the
[scoped comparison manifest](COMPARISON_MANIFEST.md). It replays Qwen's raw
responses through the existing parser using a read-only SQLite connection,
checks the original server records, request tokens, usage and budget, and
requires the completed report hash:

`ee7baf8a07368dddcce1bb80457d759dee7b1df953f8adb4f372b7327353438b`.

It verifies the private development-fit report against the frozen calibration
and threshold record. Labels and task/generator metadata are joined only after
score alignment; no model is loaded or called. Metadata is used only for
reporting descriptive slices, never as a judge or fusion feature.

The primary baselines are fresh S4, fresh MiniCheck-7B and the frozen
metadata-free fusion applied to fresh S2/S4 features. MiniCheck uses the
transferred historical TRAIN threshold despite documented legacy drift.
Historical training provenance remains limited; differing evidence visibility,
training exposure and raw baseline calibration must accompany reported results.
This is not an exact reproduction of the submitted thesis's cached MiniCheck
results or an architecture-only comparison.

## Frozen calculations

- Ranking: judge raw log odds; S4/fusion unsupported score; negated MiniCheck
  support score. AUROC and **average precision** (not trapezoidal PR area).
- Classification: unchanged TRAIN/development thresholds: judge margin >= -1.25,
  S4 >= 0.55, MiniCheck support < 0.20000000000000004, fusion >= 0.45.
- Reliability: raw and frozen calibrated judge probabilities; raw baseline
  probabilities. Brier score and ECE with the registered ten fixed bins.
- Uncertainty: 2,000 paired whole-group bootstrap draws with the registered
  seed, shared valid intersection and no refitting. Nominal 95% percentile
  intervals; retain valid-draw counts and original undefined-value policy.
- All task and generator slices are descriptive. They do not trigger additional
  threshold selection, calibration, prompt edits or model selection.

The complete paired result includes intervals for each system and judge-minus-
baseline differences. For ECE and Brier, a negative difference favors the judge;
for AUROC/AP/F1, a positive difference favors the judge. Intervals are unadjusted
for multiple comparisons and conditional on these fitted systems and this sample.

Efficiency summaries preserve the original timing scopes. Judge request
mean/median/p95 include the first request; no warmup observations are removed.
Baseline GPU-sharing histories, batching and initialization scopes differ.
There are no speedup claims or invented monetary costs. Unmeasured fusion
application time and total billable server duration remain unknown.

## Run on CPU

Use the existing fitting environment (Python 3.12, NumPy 1.26.4, SciPy 1.14.1,
scikit-learn 1.5.2). The Qwen server is not needed for this command.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.evaluate_test
```

The command prints four-system point metrics and selected paired intervals.
Full reliability tables, bootstrap identities, coverage, confusion matrices,
descriptive slices, efficiency and limitations are saved privately at:

```
.artifacts/post_thesis/llm_judge/ragtruth-frozen-four-system-evaluation-v1/report.json
```

Rerunning at the same clean revision revalidates the inputs and reads the
identical saved result without recomputing bootstrap metrics. Changed source
hashes, inputs, environment or result identity fail closed. Source score caches,
fit files and journals are not rewritten. Local regression checks use only
artificial examples; actual TEST metrics must come from the pod execution.

After reviewing the output, publish aggregate post-thesis findings with the
baseline limitations. HaluBench, disagreement analysis, optional cascades and
release packaging remain later stages. No new split or HaluBench read occurs here.
