# Post-thesis v2 synthetic diagnostic — 2026-09-19

**Development evidence only; excluded from the submitted thesis.** These ten
constructed examples are not benchmark test data or a population sample. The
[recorded console results](synthetic_diagnostic_v1_20260919.json) preserve the
operator-supplied scores and accounting. Raw cluster journals and response
payloads were not independently downloaded.

## Provenance and execution

- Commit: `772319d4cd7bee4c5619d14adfa8cc70ba48e573`.
- Run: `qwen3-v2-synthetic-diagnostic-v1`.
- Prompt: unchanged `faithfulness-development-v2`; SHA256
  `7771610009b40b5cce476fa4abda9ecb6bd025ecf71d635d7d3e7feb3c70135a`.
- Case SHA256:
  `92e8287dfd4559a8722c47d0047f122fb65036d319efd4d669651973dab9aff2`.
- Model/settings and budget:
  [committed plan](../../../post_thesis/llm_judge/configs/synthetic_diagnostic_v1.json).

All 10 examples returned valid scores in 10 new attempts, without terminal
failures or pending cases. Known usage was 6,725 input plus 95 output tokens
(6,820 total); no attempt had unknown usage. Charged client time was
8.137147818226367 seconds. This excludes server startup/idle time and is not
individual inference latency or a controlled speed comparison. Rental cost is
unknown. The unused time is not a new generation allowance.

## Results

| Target claim | Short answer | Embedded answer | Embedded minus short |
| --- | ---: | ---: | ---: |
| Supported attribute | 0.0 | 0.0 | 0.0 |
| Contradicted attribute | 1.0 | 0.1 | -0.9 |
| Unknown attribute asserted as fact | 1.0 | 0.1 | -0.9 |
| Supported numerical claim | 0.0 | 0.0 | 0.0 |
| Contradicted numerical claim | 1.0 | 0.166666 | -0.833334 |

Each matched short/embedded pair used identical context and an identical target
claim. The embedded answer added 12 explicitly supported sentences, with the
target after the sixth. Attribute evidence variants held the answer fixed;
numerical variants held the context fixed. See the
[prespecified diagnostic protocol](../../../post_thesis/llm_judge/DIAGNOSTIC.md).

All six supported-versus-unsupported comparisons had the expected score order.
However, the score assigned to each unsupported target dropped sharply when it
was surrounded by supported content; the supported controls remained at zero.
The judge distinguished unknown attributes and numerical contradictions in the
isolated cases. A broad claim that it cannot handle those rules is therefore not
supported by this diagnostic.

The pattern is consistent with dilution by supported answer content on these
constructed examples. It does not prove internal averaging, a particular fraction
calculation, or an attention mechanism. The value 0.166666 does not reveal how it
was produced. Length, content and target position changed together, so their
individual effects are not isolated. One deterministic-setting observation per
input also does not establish repeat stability or causal mechanism.

No threshold is selected, no probability is recalibrated and no final benchmark
performance is claimed. Expected synthetic labels and all observed outputs are
retained; low embedded scores are not replaced by their short-answer scores.

## Next development decision

Keep v2 unchanged. Prepare a separate binary-only diagnostic on the exact ten
inputs, asking whether any unsupported claim exists, with structured
`supported`/`unsupported` output. Preserve this run, both TRAIN pilots and all
thesis artifacts. The [binary diagnostic protocol](../../../post_thesis/llm_judge/BINARY_DIAGNOSTIC.md)
records the new output contract, bounded execution plan and interpretation limits.
No binary GPU results have been collected at this step.
