# Post-thesis metadata-free fusion reconstruction

This work is after thesis submission and is not part of the submitted thesis
results. The completed [baseline provenance audit](../../results/post_thesis/llm_judge/baseline_provenance_cluster_20260920.md)
reproduced the historical S4 and MiniCheck TRAIN thresholds, checked five stored
S4 OOF fold assignments, and fingerprinted six current S4 checkpoint directories.
Legacy input identity, checkpoint-to-cache linkage and evidence visibility remain
unverified. The original serialized fusion model has not been recovered.

## Fixed reconstruction

`recover_fusion.py` reproduces only the `S2_S4` branch of
`fusion/fusion_decomposition_review.py`. It does not execute that script's other
variants or TEST metrics. The original script, aggregate reference, four source
caches, completed provenance report, TEST manifest and frozen judge are pinned.

- Features, in order: normalized S2 minimum relevance, S4 unsupported score.
  S2 uses the original rounded constants **-11.430 and 10.641**, clipped to [0,1].
  Task, generator and labels are used for alignment only, never as features.
- All 15,090 original TRAIN rows, with stored S4 OOF scores. No new exclusions,
  resampling, missing-value imputation or silent row dropping.
- Five stratified meta-model folds, shuffle enabled, seed 42. Logistic regression
  uses the original `max_iter=1000`, seed 42 and explicit sklearn 1.5.2 defaults.
- Select the first maximum TRAIN meta-OOF F1 on `np.arange(.05, .96, .05)`, using
  `score >= threshold`. Preserve the exact float and its hexadecimal form.
- Compare the selected threshold and F1, rounded to four decimals, with the
  original metadata-free TRAIN reference (0.45 and 0.7380). This is a consistency
  check, not proof that historical coefficients or predictions are identical.
- On a match, fit one final logistic model on full TRAIN, save coefficients as
  JSON and generate the 2,700 TEST predictions. Verify JSON coefficient replay
  against sklearn predictions. No TEST metrics are computed.
- On a TRAIN reference mismatch, save that result without a final model or TEST
  predictions. Do not tune settings to match TEST performance. Convergence
  warnings are errors; no automatic solver/iteration fallback is attempted.

The maximum is six small CPU logistic-regression fits per fresh invocation.
There are no transformer/API calls, no judge refit, and no threshold changes to
S4 or MiniCheck. Completed success or mismatch records are immutable: an
identical replay verifies inputs and performs zero new fits. Interrupted runs
without a completed record may repeat the CPU reconstruction. A changed code
revision or input identity is rejected against an existing completed record.

## Run on the pod

After committing/pushing the patch and pulling it on the pod:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.recover_fusion \
  --baseline-dir /workspace \
  --train-parquet .artifacts/post_thesis/llm_judge/source-audit-inputs-v1/train.parquet
```

The existing `requirements-baseline-audit.txt` environment is sufficient. Run the
same command a second time to verify replay. Send the printed summary and hash.

Private output:
`.artifacts/post_thesis/llm_judge/ragtruth-metadata-free-fusion-recovery-v1/report.json`.
It contains the fitted fold models, meta-OOF scores, threshold candidates, final
coefficients, TEST scores and source identities. All original caches/checkpoints,
judge artifacts and thesis results are read-only.

## Interpretation and remaining work

This is a **post-thesis reconstruction from legacy cached features**. A matching
rounded TRAIN reference does not establish exact historical reproducibility.
`target_manifest_input_sha256` describes the evaluation row we aligned to; it
must not be presented as a hash captured during historical baseline inference.
The report deliberately leaves `comparison_ready` false.

The original baseline uses full TRAIN, including rows reserved for judge
calibration/development. Training exposure differs and must be disclosed.
Meta-OOF thresholding is not fully nested end-to-end validation of the S4 training
pipeline, and matching S4 fold tags does not independently prove training
exclusion. No feature visibility or leakage guarantee is added by this recovery.

Next, resolve how the explicitly limited legacy baselines will be reported (or
establish fresh inference provenance where required), then complete the judge
TEST token audit and bounded execution manifest. This command does not authorize
judge TEST inference or use HaluBench.
