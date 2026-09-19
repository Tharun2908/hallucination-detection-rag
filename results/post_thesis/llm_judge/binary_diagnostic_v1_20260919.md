# Post-thesis binary synthetic diagnostic — 2026-09-19

**Development evidence only; excluded from the submitted thesis.** This binary
follow-up was selected after inspecting the probability diagnostic on the same
ten constructed inputs. It is neither held-out validation nor a benchmark result.

## Provenance and execution

The [recorded outputs](binary_diagnostic_v1_20260919.json) transcribe the operator's
console results. All ten verdicts and expected-verdict matches were supplied.
The instructed and GitHub-verified checkout was
`d44f430ba83175914c6cf90b08bc8480c83507df`; private launcher records and raw cluster
response payloads were not independently downloaded to verify deployment.

- Run ID: `qwen3-binary-synthetic-diagnostic-v1`.
- Prompt: unchanged `faithfulness-binary-diagnostic-v1`; SHA256
  `e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683`.
- Case SHA256:
  `92e8287dfd4559a8722c47d0047f122fb65036d319efd4d669651973dab9aff2`.
- Model/settings and resource limits:
  [committed plan](../../../post_thesis/llm_judge/configs/binary_diagnostic_v1.json).
- Execution: 10/10 valid verdicts, 10 new attempts, zero failures or pending cases.
- Known usage: 6,285 input + 80 output = 6,365 tokens; unknown usage on zero attempts.
- Charged client time: 7.542062098626047 seconds, excluding server startup/idle time.

The client window is not a controlled speed comparison with the probability
run. Rental cost remains unknown. No new generation allowance is implied by
unused time.

## Matched observations

| Target | Probability: short | Probability: embedded | Binary: short | Binary: embedded |
| --- | ---: | ---: | --- | --- |
| Supported attribute | 0.0 | 0.0 | supported | supported |
| Contradicted attribute | 1.0 | 0.1 | unsupported | unsupported |
| Unknown attribute asserted as fact | 1.0 | 0.1 | unsupported | unsupported |
| Supported numerical claim | 0.0 | 0.0 | supported | supported |
| Contradicted numerical claim | 1.0 | 0.166666 | unsupported | unsupported |

Probability values come from the [preserved prior diagnostic](synthetic_diagnostic_v1_20260919.md).
All ten binary verdicts matched the synthetic expectations: all six unsupported
cases and all four supported controls. In particular, the three embedded errors
were identified without labeling their supported controls as unsupported.

This strengthens the hypothesis that the output-elicitation formulation affects
behavior on these examples. It does not prove the probability judge internally
recognized the errors and merely encoded them incorrectly. Wording, schema and
response space changed together; there is one execution per constructed input.
There is no estimate of population accuracy, stability or generalization.

Verdicts remain strings. They are not probabilities zero and one, and no AUROC,
ECE, Brier score, probability calibration or operating threshold is inferred
from these verdicts.

## Next development step

Keep the binary formulation fixed and use the existing 50-example TRAIN pilot to
check realistic answers and supported controls. Record precision, recall, F1,
coverage and per-example failures after execution, with original labels intact.
Revisit previously missed cases 9350, 6040, 6486 and 12052 within the full paired
analysis; do not evaluate only those cases or exclude disputed annotations.

The [binary TRAIN pilot guide](../../../post_thesis/llm_judge/BINARY_PILOT.md)
starts with a separate token-only audit. Its result must be pinned before a
separate scoring plan is enabled. Continuous scoring remains an open design
decision; this diagnostic does not replace the planned continuous-score study.
Neither RAGTruth TEST nor HaluBench is involved.
