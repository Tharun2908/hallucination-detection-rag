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
| v2 scoring | Complete on commit `662af1dd2e94059f6cdb6f5eacf14c84577fe293`; 50/50 valid scores |

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

## Completed token audit — historical procedure

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

The operator completed this audit on commit `f0645464dbe1803921b8a8545fad980eaae4e424`.
Preserve that audit unchanged; do not rerun it from the new scoring commit.
The following plan pins the reported checksum and validates the complete local
audit, including per-request identities/counts, before any scoring call.


### Operator-reported audit evidence

The following comes from the supplied cluster console output. The private audit
file was not independently downloaded; the scorer verifies its content hash and
provenance against these pins on the cluster.

| Field | Value |
| --- | --- |
| Audit SHA256 | `fb28d1bab2bc2cd64d650af26dce5fad8c7374b5997acf8d6afe720d59bbeaae` |
| Audit code revision | `f0645464dbe1803921b8a8545fad980eaae4e424` |
| Counted / failed / pending | 50 / 0 / 0 |
| Minimum / maximum input tokens | 763 / 3,028 |
| Total input tokens | 62,674 |
| Output allowance per example | 128 |
| Model limit | 32,768 |
| Overlength inputs / generation calls | 0 / 0 |

The revised prompt adds 275 input tokens per example relative to v1: 13,750
additional input tokens over 50 requests. This is a length observation, not
inference latency or evidence of improved verification.

## Completed separately bounded v2 scoring — historical procedure

[configs/ragtruth_pilot_50_v2.json](configs/ragtruth_pilot_50_v2.json) pins the
existing preparation manifest, v2 prompt, v2 audit and unchanged model profile.
It authorizes only the same 50 TRAIN development inputs, with one attempt each,
concurrency one, a 60-second request timeout and a separate **600-second cumulative
client budget across resumes**. The output allowance remains 128 per input,
6,400 total; audited input plus maximum requested output is 69,074 tokens.
This is not a monetary cost estimate. GPU rental cost remains unknown.

The fixed run ID is `qwen3-ragtruth-train-pilot-50-v2`. It has its own journal,
request cache and execution budget under the ignored artifact directory. The v1
configuration, results, budget and preparation manifest are unchanged. No unused
v1 allowance transfers to v2. This does not authorize a stability repeat or a
benchmark test run.

Apply, commit and push the scoring patch, then pull it on H200. No new packages
are needed. Start the pinned launcher from that same clean scoring commit using
the existing CUDA/cache environment. After `Application startup complete`, use
the directory name printed by that launcher in the client terminal:

```bash
python -m post_thesis.llm_judge.run_pilot \
  --pilot-version v2 \
  --server-session-id server-REPLACE_WITH_SESSION_ID
```

This command generates v2 scores. It reads the existing preparation manifest and
v2 token audit, re-tokenizes each exact request and checks its audited count
before completion. The judge sees the v2 prompt plus answer/context only. Neither
test set is loaded. Labels stay offline; no threshold or calibration is fitted.
The CLI prints both pilot and prompt versions to make the selection visible.

The same budget, cancellation, interruption, error and usage accounting described
in [PILOT.md](PILOT.md) applies. Resume from the **same scoring commit** with the
same command. Successful predictions are cached; terminal failures are preserved,
not retried, and only pending rows can consume the remaining allowance. Changed
code/artifacts, unknown elapsed windows and alignment failures retain their
existing fail-closed behavior. For inspection with zero HTTP calls:

```bash
python -m post_thesis.llm_judge.run_pilot --pilot-version v2 --max-new-attempts 0
```

Retain the full v2 run directory and all historical v1 artifacts. Stop the server
when finished so its resource record is finalized. Client execution time excludes
server startup and idle time, and overlapping resource windows must not be added.
The operator supplied the summary and paired predictions; the
[paired report](../../results/post_thesis/llm_judge/ragtruth_pilot_v1_v2_20260919.md)
records development findings. Do not rerun this historical scoring procedure
from a newer commit. The [synthetic probability diagnostic](DIAGNOSTIC.md) also
completed, as did the [binary-output follow-up](BINARY_DIAGNOSTIC.md). Next is
the [binary TRAIN token audit](BINARY_PILOT.md), preserving v2 and both pilot runs.
