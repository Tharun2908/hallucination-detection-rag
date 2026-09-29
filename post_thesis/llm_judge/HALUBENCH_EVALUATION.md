# Post-thesis frozen HaluBench evaluation

All four systems now have fresh scores for the unchanged canonical HaluBench
8,000-example TEST population. The Qwen continuation completed 348/348 requests
and its zero-call replay reproduced hash
`b3bb26a1e568bab732bace6e31d55e3610d06b4bf86a57d386b4d648dd15b937`.
This evaluation is post-thesis work and is excluded from submitted thesis results.
No HaluBench metric values have been recorded by this implementation step.

## Exact-once population and inference provenance

`evaluate_halubench.py` validates the original input manifest, frozen fit, fresh
S4, S2, MiniCheck and fusion reports, historical TRAIN operating policies and
frozen fusion lineage. It opens both judge journals read-only and replays their
raw responses under their original scoring revisions and server snapshots.
It requires the exact reported parent and continuation hashes.

The in-memory population uses the parent's 7,652 valid scores followed by the
continuation's 348 valid scores. All original canonical IDs, indices and input
hashes must match exactly once. The parent's interrupted/pending rows contribute
no replacement score. Both source runs remain unchanged; incomplete populations,
missing or duplicate rows, altered payloads, or failed child rows block evaluation.
The main population remains 8,000 responses and 7,198 exact-passage components.

There were 8,001 client attempts for 8,000 valid scores. Known usage totals
7,794,485 input and 8,000 output tokens, plus one attempt with unknown input/output
usage. Charged client time includes both runs and the timeout; preparation and
server startup/idle time are excluded. Successful-request latency distributions
exclude the interrupted request because its latency was not saved. Unknown usage
and monetary cost are never represented as zero. Timing is descriptive: runtimes,
preprocessing, batch sizes and sharing differ, so no speedup ratio is inferred.

## Unchanged statistical contract

The numerical core is byte-checked against the pre-inference contract
`c092deea43883442914500fedb0627b7938c513ce5a77bc730079161b4425373`.

- All four systems share the complete canonical population.
- AUROC and average precision use the frozen ranking definitions. Average
  precision is sklearn AP, not trapezoidal integration of a PR curve.
- F1, precision, recall and accuracy use the RAGTruth TRAIN operating rules:
  judge margin >= -1.25; S4 unsupported probability >= 0.55; MiniCheck support
  probability < 0.20000000000000004; metadata-free fusion >= 0.45.
- Judge raw and RAGTruth-calibrated probability views are both reported. The
  fixed calibration is sigmoid(0.3808096416654668 * margin + 0.3575745057816423).
  Baselines retain their raw probability views; no calibration parity is implied.
- ECE uses the previously registered ten bins; Brier is also reported.
- 2,000 paired whole-component bootstrap draws use the frozen seed 20260920,
  PCG64, nominal unadjusted 95% percentile intervals and minimum 1,900 valid draws.
  Differences are judge minus baseline; reliability comparisons identify the
  calibrated-judge versus raw-baseline views explicitly.
- Every source (DROP, FinanceBench, covidQA, halueval, pubmedQA) gets descriptive
  metrics. No new slice-specific thresholds, fits or confidence intervals.

No recalibration, threshold selection, target-domain adaptation, prompt changes,
new model calls, alternative split, new baseline fitting or metric-driven model
selection occurs. The loader is added after inference; the numerical procedure
was fixed beforehand. Source slices are descriptive, and aggregate results keep
the original source proportions rather than rebalancing the test set.

Limitations remain visible: S4 truncates 850 contexts; S2 retains 1,886 original
empty-pair raw-zero features and 35 truncated context pairs. MiniCheck and Qwen
fit all their respective prompts. Historical training/checkpoint provenance is
not established retrospectively by fresh score agreement. The exact-passage
components link 994 TEST examples to adaptation components; no adaptation is used,
and strict document disjointness is not asserted.

## CPU commands

Qwen can be stopped with Ctrl+C in its launcher terminal. The PVC artifacts are
sufficient for evaluation; an H200, CUDA, model weights and an HTTP server are
not needed. Use the existing separate CPU fitting environment, preserving the
serving environment's packages.

After applying/pushing this patch on Windows, on the pod:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
.venv-judge-fit/bin/python -m post_thesis.llm_judge.evaluate_halubench
```

Python 3.12, NumPy 1.26.4, SciPy 1.14.1 and scikit-learn 1.5.2 are required and
checked before numerical evaluation. If the persistent fitting environment's
interpreter link is broken on a newly created pod, restore Python 3.12.14 and
reconnect that link, as for the serving environment. Do not recreate an existing
virtual environment or upgrade its numerical packages. The existing
`requirements-baseline-audit.txt` records the compatible CPU stack.

A package-free read-only integrity check is also available when diagnosing a
fresh CPU pod, independently of the fitting environment:

```bash
python3 -B -S -m post_thesis.llm_judge.evaluate_halubench --verify-only
```

Evaluation prints aggregate metrics, paired differences, all descriptive source
metrics and the report hash. Save the output if desired. Private output:
`.artifacts/post_thesis/llm_judge/halubench-frozen-four-system-evaluation-v1/report.json`.
Source records are read-only; only the new evaluation directory is written.

Run the same evaluation command again. It must revalidate the inputs, print
`new evaluation: False` and reproduce the report hash without recomputing the
bootstrap. A changed identity or damaged cached report is rejected. Share the
first result and replay hash before interpreting cross-domain findings.
