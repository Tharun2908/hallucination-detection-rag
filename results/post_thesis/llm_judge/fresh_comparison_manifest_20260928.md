# Post-thesis fresh comparison manifest: cluster verification

The operator reported successful creation of the manifest on 2026-09-28:

- 2,700 examples and 450 fixed canonical groups.
- S4, MiniCheck-7B and metadata-free S2+S4 ready for the explicitly documented
  fresh-runtime comparison. Historical provenance limits remain in the record.
- Fixed thresholds: S4 unsupported >= 0.55; MiniCheck support strictly
  < 0.20000000000000004; metadata-free fusion unsupported >= 0.45.
- MiniCheck fresh/legacy mean absolute difference 0.02947428918387699,
  maximum 0.3774411483820142, 100 changed fixed-threshold decisions.
- No model calls, fitting, TEST metrics or generation allowance.

Manifest SHA256:
`3e8c25eaaeaf03aa93377f8a7b6c516e6a9f4e8c811305fb88683e78e0b615bb`.

Private path:
`.artifacts/post_thesis/llm_judge/ragtruth-fresh-comparison-manifest-v1/manifest.json`.

This records the operator-supplied output, not an independent assistant replay
of the private reports. The [bounded judge runner](../../../post_thesis/llm_judge/TEST_LABEL_RUN.md)
requires this exact manifest and revalidates its content and baseline links on
the pod. Judge TEST predictions remain pending. The submitted thesis and legacy
artifacts are unchanged.
