# Post-thesis RAGTruth TEST token audit — completed 2026-09-20

The pod audit and cached replay match the independent local token counts. All
2,700 frozen inputs fit without truncation. This result concerns input lengths,
not judge TEST performance, and is outside the submitted thesis.

| Item | Result |
|---|---:|
| TEST inputs counted | 2,700 / 2,700 |
| Pending / overlength | 0 / 0 |
| Total input tokens | 3,334,607 |
| Minimum / maximum input length | 649 / 2,849 |
| Output allowance per example | 1 token |
| Model limit | 32,768 tokens |
| Initial / replay new TEST tokenizations | 2,700 / 0 |
| Generation / HTTP requests | 0 / 0 |

Audit SHA256:
`ee66f093aef059363151cc714cca1ee5cc318ba443dd9e1a9b3e8db51674407c`.
Code revision: `5bc2f4f82a5ccd80ad5199643ce9f177fc4b0a96`.

Both invocations reported the same hash. Independently computed local token rows,
with the operator's committed revision substituted for the local validation marker,
reproduce that full hash exactly. The original cluster file was not transferred;
this is an independent reconstruction, not a claim to have read its filesystem.
The [JSON record](test_token_audit_cluster_20260920.json) preserves these distinctions.

The pod also verified the immutable fusion report and all 2,700 saved predictions
without new fits. Legacy comparison readiness remains false. The private fit and
fusion reports have not been independently transferred to the assistant.

No model weights, inference, fitting, benchmark metrics or HaluBench access were
needed. The [evaluation numerical core](../../../post_thesis/llm_judge/EVALUATION_MATH.md)
is the next pre-inference implementation step. Baseline provenance and a separately
bounded execution plan remain required; this audit grants no scoring allowance.
