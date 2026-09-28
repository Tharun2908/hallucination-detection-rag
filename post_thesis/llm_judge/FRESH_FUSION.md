# Post-thesis frozen fusion on fresh S2/S4 TEST features

The [fresh S2 run and replay](../../results/post_thesis/llm_judge/s2_fresh_test_cluster_20260928.md)
completed all 2,700 examples. Sentence counts match the legacy cache for every
row; maximum raw-min score difference is below 0.000055. Fresh S4 also completed
with close score agreement. These findings support proceeding with the fixed
fusion; they do not establish historical training/cache provenance.

`apply_fresh_fusion.py` now consumes these two pinned fresh reports and the
previously reconstructed fusion, validates their canonical TEST input hashes,
and writes a separate post-thesis prediction report. **No refitting occurs.**

## Fixed application

The full-precision source reports are read-only. Before applying the fusion,
round each fresh S2 raw-min feature and S4 unsupported probability using Python
`round(float, 4)`, matching the historical cache representation on which the
fusion was reconstructed. This rule is fixed before inspecting the new fusion
outputs. Do not choose between rounded/full-precision variants using TEST labels.

Normalize rounded S2 using the preserved TRAIN min/max -11.43 and 10.641,
clipped to [0, 1]. Feature order is `[s2, s4]`; coefficients are
`[-1.3590096505049223, 3.155528234679687]`, intercept
`-1.4168645142264538`. Apply the logistic sigmoid using the same pinned numerical
libraries as reconstruction. Metadata features are absent.

The operating decision remains probability **>= 0.45**, preserving the exact
TRAIN-selected float (`0x1.ccccccccccccdp-2`). The script verifies these values
against the immutable reconstruction report; it never estimates parameters or
selects thresholds. Labels remain confined to the existing manifest integrity
check and are not projected into feature application or agreement calculations.

The output retains per-example input hashes, rounded features, normalized S2,
probabilities and fixed-threshold decisions. It also reports label-free score
differences and decision changes relative to the previously saved fusion
predictions. These comparisons cannot prove historical provenance or accuracy.
Completed output is immutable; repeating the command verifies identical output.

## Run on the pod

After applying, committing and pushing the patch:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.apply_fresh_fusion
```

Run the last command again: expect `New output: False` and an identical hash.
The existing CPU environment needs numpy 1.26.4, scipy 1.14.1 and scikit-learn
1.5.2, as used for reconstruction. No transformer loads, GPU access, API requests,
new fitting, TEST label metrics or HaluBench reads occur.

Output: `.artifacts/post_thesis/llm_judge/ragtruth-fresh-metadata-free-fusion-v1/report.json`.
All original caches, the frozen judge/fit and earlier fusion predictions remain
unchanged. The CLI requires the exact completed S2 and S4 reports; it does not
silently drop failures or accept different inference runs.

## Remaining comparison limits

Fresh inference establishes current input/checkpoint/visibility records. It
does not independently prove legacy S4 training exclusions or the input identity
of old TRAIN features used by the reconstructed fusion. Baseline TRAIN exposure
includes the judge's development rows and differs from judge calibration.
S2 filters sentences and may truncate pairs; S4 truncates answers/contexts.
These are not equal-evidence comparisons with the full-context judge.

The report keeps `comparison_ready: false` until the final comparison explicitly
handles those limitations. This flag does not request another fusion experiment.
MiniCheck baseline preparation is next; the judge, calibrator and threshold
remain frozen. No new thesis result is asserted.
