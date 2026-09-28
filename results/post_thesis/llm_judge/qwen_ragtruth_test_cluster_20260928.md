# Post-thesis Qwen RAGTruth TEST scoring completed

Operator-supplied H200 output, 2026-09-28. This is inference completion, not
an accuracy result or a submitted thesis result. Both original execution and
cache-only replay reported:

| Field | Result |
| --- | ---: |
| Valid scores | 2,700 / 2,700 |
| Terminal failures / pending | 0 / 0 |
| Original new attempts | 2,700 |
| Replay new attempts | 0 |
| Input tokens | 3,334,607 |
| Output tokens | 2,700 |
| Unknown input/output usage attempts | 0 / 0 |
| Charged client seconds | 3,535.408586259 |
| Remaining registered client seconds | 3,664.591413741 |
| Halt reason | None |

The preparation-only check produced zero generation/HTTP calls. Its plan hash
was `3273a94ff8b7d079267fedb9386adba8a9baa3329d9addbd53168dc4e4d722ec`
and request hash was
`03b6971ddea6334ed76328d6e299a163417bde925cbad6bac3a775b932d12da1`.
The operator restored nvcc 13.0.88 and confirmed `curand.h` on the new pod, then
started the unchanged `label-score-v1` serving profile in session
`server-b98d3e77c89b433a955057846e1d9853`.

Completed and replay report SHA256:
`ee7baf8a07368dddcce1bb80457d759dee7b1df953f8adb4f372b7327353438b`.

Private summary:
`.artifacts/post_thesis/llm_judge/qwen3-ragtruth-test-label-score-v1/summary.json`.

Client duration is approximately 58.9 minutes and includes client work. It is
not GPU kernel time or total billable server lifetime. No monetary cost is
inferred. No TEST metrics, calibration fitting, threshold selection, HaluBench
reads or legacy-file changes occurred during inference/replay.

The assistant has not independently read the private cluster records. The
[CPU evaluator](../../../post_thesis/llm_judge/TEST_EVALUATION.md) verifies them
read-only and applies the already frozen numerical rules. Cluster evaluation
remains pending; this file makes no claim of improved detector performance.
