# Post-thesis fresh MiniCheck RAGTruth TEST — cluster completion

Operator-reported inference and cached replay completed on 2026-09-28.
The runner was introduced in `ea7e912a0e581a2f2d9fed79c6b80a89dccffeeb`;
the private report binds the actual runtime code revision.

| Field | Observed value |
| --- | ---: |
| Valid examples | 2,700 / 2,700 |
| Initial new attempted examples | 2,700 |
| Replay new attempted examples | 0 |
| Terminal failures / pending | 0 / 0 |
| Sentence generation requests | 18,935 |
| Input tokens | 13,661,600 |
| Output tokens | 18,935 |
| Shortest / longest input | 240 / 2,236 tokens |
| Truncated prompts | 0 |
| Known example-processing seconds | 558.9951405017637 |
| Known initialization seconds | 26.604065424762666 |
| Unknown timing / usage records | 0 |

Both invocations reported the same summary SHA256:
`365f4d3b097ff1361bd1ccbea911c846d8b26560bd7882d1834801d79e9237f2`.
The replay verified cached files and prepared inputs but performed no new
example inference. It had no halt reason.

Private summary:
`.artifacts/post_thesis/llm_judge/minicheck-ragtruth-test-fresh-v1/summary.json`.
This public record transcribes the supplied summaries; it does not reconstruct
the private report or its per-example scores.

Model and decoding follow the successful synthetic-v2 configuration: pinned
Bespoke-MiniCheck-7B, BF16 H200, saved-tokenizer adapter, native InternLM2,
FlashAttention, FlashInfer sampler disabled, upstream prompt and sentence/chunk
aggregation. The historical TRAIN threshold remains strictly support
`< 0.20000000000000004`.

Example-processing time is about 9.32 minutes. It includes CPU preparation and
token checks inside each example call, plus completed generation. Initialization
is separate. This is not isolated CUDA-kernel timing or a controlled cross-system
speedup result. Initial whole-dataset preparation and report I/O are not included.

Fresh S4, S2, metadata-free fusion and MiniCheck TEST predictions are now present.
The Qwen judge TEST run and final comparative evaluation are still pending.
No TEST label metrics, fitting, HaluBench reads or legacy-file changes occurred
in this MiniCheck run. Close legacy agreement, if observed next, cannot prove
historical input/checkpoint/runtime provenance.

## Next: numerical agreement without labels

```bash
python -S -m post_thesis.llm_judge.compare_minicheck_legacy \
  --legacy-file /workspace/minicheck_results_test_7b.json
```

This verifies the pinned fresh report, original legacy cache and canonical
TEST input alignment. It reports score differences and changes at the existing
TRAIN threshold. It neither evaluates correctness nor selects a new threshold.
The twenty largest score differences are listed with IDs and scores only.
Rerunning preserves an identical report and makes no model calls.
