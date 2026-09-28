# Post-thesis MiniCheck fresh/legacy score comparison

Operator-supplied cluster output recorded on 2026-09-28. This is a score
agreement check, not a TEST performance evaluation or a submitted thesis result.
The assistant has not independently read the complete private cluster reports.

| Observation | Value |
| --- | ---: |
| Aligned examples | 2,700 |
| Exact score matches | 0 |
| Matches after rounding fresh scores to four decimals | 59 |
| Mean absolute difference | 0.02947428918387699 |
| Median absolute difference | 0.015970105865495687 |
| Maximum absolute difference | 0.3774411483820142 |
| Mean signed fresh minus legacy | -0.0023662641868128992 |
| Differences above 0.001 | 2,426 |
| Differences above 0.01 | 1,635 |
| Differences above 0.1 | 145 |
| Changed fixed-threshold decisions | 100 / 2,700 (3.70%) |

Higher scores mean more supported. The policy remains support score strictly
less than `0.20000000000000004` (`0x1.999999999999bp-3`) means unsupported.
The largest difference was sample `3081`, TEST index `567`: fresh
`0.10665885161798579`, legacy `0.4841`; its decision changed.

This difference is materially larger than four-decimal rounding. The check
used no labels, model calls or threshold fitting, so it does not establish
whether the fresh or historical system is more accurate. It also does not
invalidate the historical results or prove historical input/checkpoint lineage.

## Source review and interpretation

The historical `signals/minicheck_baseline.py` submits groups of 16 answers to
`scorer.score`. The fresh runner submits one answer's sentence/chunk prompts per
call. The pinned upstream implementation aggregates the submitted prompts in
one `llm.generate` call. The prompt, chunking and min-over-answer/max-over-context
aggregation were compared with that implementation. Runtime and tokenizer
loading also differ; the actual historical runtime is not independently known.
Batching, runtime and tokenizer differences are candidates, not established
causes. Do not attribute the drift to a specific cause from these aggregates.

Reference source: [MiniCheck inference at pinned upstream revision](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/minicheck/inference.py).

Use **fresh MiniCheck on the documented runtime with the transferred historical
TRAIN threshold** in the primary post-thesis comparison. Retain both caches and
make no TEST-derived threshold adjustment. The fresh runtime's current input
and checkpoint linkage is documented; historical training lineage remains
limited. This scope is recorded in the
[comparison manifest](../../../post_thesis/llm_judge/COMPARISON_MANIFEST.md).
No new MiniCheck inference is needed for this manifest step.

## Artifact identities

- Fresh report: `365f4d3b097ff1361bd1ccbea911c846d8b26560bd7882d1834801d79e9237f2`.
- Legacy file: `4e20383ab51a20c53934db5073d04e124e22079bc9a8bd99d5253eee7e320bcf`.
- Agreement report: `60ca5e5e1c2a3ce5a49c40ee7e2ee004b055ffd30876ee95dc9739a9d69a9489`.
- Private report: `.artifacts/post_thesis/llm_judge/minicheck-fresh-legacy-agreement-v1/report.json`.

The new manifest command will verify the full private report's content hash on
the pod. No report has been rewritten, no threshold changed and no judge TEST
scoring or HaluBench evaluation has been performed by this step.
