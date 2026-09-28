# Post-thesis frozen fusion on fresh features: completed

Operator-supplied cluster observations received 2026-09-28. These are post-thesis
prediction/provenance checks, not new thesis results or label-based TEST metrics.
The full private prediction report was not transferred to the report author.

- All 2,700 fresh S2/S4 pairs produced valid fusion probabilities.
- First invocation: `New output: True`; second: `New output: False`.
- Both invocations report SHA256
  `931476d6815f8959d4073ef13abf595a077bf89da31ff82f9b72571e17296e3d`.
- Features were rounded to four decimals before the fixed TRAIN normalization,
  as registered. Full-precision source inference reports remain unchanged.
- Frozen TRAIN threshold: fusion unsupported probability >= 0.45.
- Legacy comparison, without labels: maximum absolute score difference
  `7.191615074386704e-05`, mean absolute difference
  `1.8497179968726685e-07`, and **zero decision changes** across all 2,700 rows.
- No fitting, transformer/judge calls, TEST label metrics or legacy-file changes.

Private report:
`.artifacts/post_thesis/llm_judge/ragtruth-fresh-metadata-free-fusion-v1/report.json`.

This confirms unchanged operating decisions relative to the reconstructed
legacy fusion under the frozen threshold. It does not establish the historical
training/input provenance of legacy caches, eliminate unequal evidence visibility,
or imply new accuracy measurements. Preserve the report and its limitations.
Fresh S4, S2 and fusion preparation is complete; MiniCheck preparation follows.
