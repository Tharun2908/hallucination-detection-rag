# Post-thesis development prompt v2

This revision is motivated by inspected **TRAIN development** outcomes. It is
not part of the submitted thesis, not a final prompt freeze, and not a held-out
validation result. The [v1 pilot report](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_20260919.md)
and its 50 predictions are preserved unchanged.

## Recorded revision and rationale

| Item | Value |
| --- | --- |
| Version | `faithfulness-development-v2` |
| Prompt SHA256 | `7771610009b40b5cce476fa4abda9ecb6bd025ecf71d635d7d3e7feb3c70135a` |
| Source | `DEVELOPMENT_PROMPT_V2` in `prompts.py` |
| Preparation manifest | Existing 50 examples; SHA256 `ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25` |
| Model/decoding/schema | Same pinned Qwen3-32B H200 profile and one-field JSON schema |
| v2 scoring | Not implemented or executed at this step; budget/audit pin pending |

The v1 pilot produced valid responses but scores only in `[0, 0.2]`. Manual
inspection of its five zero-scored positives found clear missed additions,
unknown-value inferences, and a numerical relationship error, plus a concern
about one annotated span. Original benchmark labels remain unchanged.

v2 makes the following general checks explicit:

1. Check each factual assertion, including small details and disclaimers.
2. Treat missing/null/None/unreported values as unknown unless defined otherwise.
   Unknown establishes neither presence nor absence; other context may supply
   the information, so check it too.
3. Check numerical relationships, direction, entity association, negation and
   claims about information being absent from the context.
4. Estimate whether **any** claim is unsupported. Do not dilute a clear error by
   averaging over a mostly correct answer. Uncertainty is about support, not
   the fraction of claims affected. No particular score distribution is required.

The prompt contains no dataset names, sample IDs, pilot answers, labels, generator
identities, or benchmark-specific examples. Inputs remain answer and context
only. No rationale field, thinking mode, or output-token increase is introduced.

This bundles several instruction changes. A later paired improvement would not
isolate an individual cause. An upward score shift alone is not evidence of
better discrimination. The same 50 examples remain development data; compare
ranking, probabilities and concrete errors without selecting thresholds on them
or presenting their results as held-out performance. Threshold fitting remains
reserved for separate source-development data after the prompt freeze.

## Preservation rules

`DEVELOPMENT_PROMPT` remains v1, with its exact original text and hash. All
existing default request construction, synthetic smoke and v1 scoring settings
remain v1. `get_prompt()` selects an explicit recorded version and refuses unknown
versions. Request keys include the selected prompt identity and text.

The preparation manifest's `initial_prompt` correctly remains v1. It records the
historical preparation, not a restriction preventing separately named prompt
experiments. v2 uses its exact answer/context inputs and records its selected
prompt in a new audit. Do not recreate that manifest, overwrite the v1 audit, or
edit the v1 execution plan/budget. Historical runs retain their code revisions;
their guards intentionally reject rerunning them from a changed checkout.

## Next cluster action: token audit only

Apply, commit and push this patch, then pull it on the H200 cluster. Start the
same pinned server from the updated clean checkout, with the CUDA and cache
environment in [SERVING.md](SERVING.md). After startup completes, in the client
terminal run:

```bash
python -m post_thesis.llm_judge.audit_pilot \
  --prompt-version faithfulness-development-v2 \
  --expected-manifest-sha256 ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25
```

The default output directory is separate:
`.artifacts/post_thesis/llm_judge/ragtruth-train-pilot-50-token-audit-v2/`.
The command uses `/tokenize` and never generates scores. Its limit remains one
tokenization attempt per example, at most 60 seconds per operation and a
300-second invocation deadline. Unknown/failed/interrupted accounting and cache
rules are the same as the v1 audit. Passing another prompt version's reserved
default directory is rejected, and any changed cached identity fails closed.

Check that it reports v2, 50 counted inputs, no overlength cases and zero
generation calls. Retain the printed **audit checksum and audit code revision**
along with the new total/minimum/maximum input counts. Stop the server when done
so its resource record is finalized. The audit itself is not inference latency
or a declaration of free GPU usage.

After reviewing that output, freeze a separate v2 execution plan and budget
before any v2 scoring. The current `run_pilot.py` is still the frozen v1 scorer;
do not run it expecting v2 or try to spend v1's remaining allowance on v2.
