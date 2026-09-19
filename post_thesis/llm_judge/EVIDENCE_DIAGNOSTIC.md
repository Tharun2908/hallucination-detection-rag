# Post-thesis: structured evidence-checking diagnostic

**Development only. Excluded from the submitted thesis.** This step freezes a
new diagnostic prompt/output contract, implements an offline parser and defines
synthetic controls. **No model responses have been collected for this contract.**
There is no inference runner, token audit, execution plan or generation budget
for it yet. The final benchmark prompt is not frozen.

The [binary TRAIN result](../../results/post_thesis/llm_judge/ragtruth_binary_pilot_20260919.md)
detected 13/24 labeled positives with three false positives. The
[focused review](../../results/post_thesis/llm_judge/ragtruth_binary_error_review_20260919.md)
confirmed four persistent misses and identified source/annotation ambiguity
among apparent false positives. The synthetic binary result was 10/10, so
synthetic success alone is insufficient evidence of realistic grounding.

## Question and scope

Does requesting an explicit problematic claim and evidence comparison alongside
the verdict recover persistent grounding errors without increasing false positives?
Are the returned evidence statements themselves faithful to the supplied text?

Keep the initial model candidate Qwen3-32B on H200 and the same answer/context-only
input boundary. A later TRAIN comparison must reuse the original 50 examples,
labels and order. Do not select a replacement pilot or read benchmark TEST data.
The change bundles prompt wording, output schema and potentially output allowance;
it does not isolate one causal mechanism or prove the model's internal reasoning.

The output identifies **one selected problematic claim**; it is not an exhaustive
claim list or proof that every answer claim was checked. For an unsupported verdict, the
judge selects one problematic assertion. For a supported verdict, it emits no
claim witness; success cannot be inferred from the absence of a witness.

## Frozen development contract

The [descriptor](evidence_contract_v1.json) pins the resolved prompt, response
schema and synthetic cases. It is an offline contract record, not an execution
configuration. Any substantive change requires a new version and a separate
request identity; completed probability and binary contracts stay unchanged.

| Field | Identity |
| --- | --- |
| Prompt | `faithfulness-evidence-diagnostic-v1` |
| Prompt SHA256 | `21ba57c9ec668c495a16e4635660c4504f26b5056638d4b17e6424e32ce52f4c` |
| Request contract | `evidence-diagnostic-request-v1` |
| Schema title | `faithfulness_evidence_verdict` |
| Schema SHA256 | `ecf37ce6cc1775c9af61d7b30754f5fd84c9c7ccf6a8845907a6347cc8807f12` |
| Synthetic cases SHA256 | `2e7aee4037a531ff971c47ade95172c3011b9bc8d874da4cea5c124484df4936` |

`evidence_contract.py` preserves the binary support criteria and replaces its
output instructions with a five-field object. `evidence_request()` accepts only
a `JudgeInput` and configuration; messages contain system instructions plus the
exact answer/context JSON. IDs, labels, task/generator metadata and expected
issues remain offline. Prompt, schema, contract and configuration participate in
the request key, separating future caches from probability/binary requests.

All five keys are required, in any JSON object order:

| Verdict / issue | answer_quote | context_quote | explanation |
| --- | --- | --- | --- |
| `supported` / `none` | null | null | null |
| `unsupported` / `contradiction` | Exact answer excerpt | Exact conflicting context excerpt | Brief evidence comparison |
| `unsupported` / `insufficient_support` | Exact answer excerpt | Relevant exact context excerpt, or null | Brief statement of what lacks support |

Quotes must be nonempty contiguous substrings of the decoded original input.
Limits are **400 characters for answer_quote**, **600 for context_quote** and
**320 for explanation**. Characters are Python Unicode code points; these limits
are not token budgets. Explanations are concise evidence comparisons, not
step-by-step reasoning traces. No probabilities or additional keys are accepted.

Illustrative parser fixture (not a model result), for answer `The venue offers
outdoor seating.` and context `{'OutdoorSeating': False}`:

```json
{
  "answer_quote": "The venue offers outdoor seating.",
  "context_quote": "{'OutdoorSeating': False}",
  "issue_type": "contradiction",
  "explanation": "The supplied field explicitly rules out outdoor seating.",
  "verdict": "unsupported"
}
```

With an unknown field instead, a positive availability assertion lacks support;
unknown does not explicitly contradict availability. Other passages may still
supply support. A missing context quote is allowed only for insufficient support
or the all-null supported record; it cannot establish that no supporting evidence
exists anywhere in the context.

## Parser guarantees and limits

`parse_evidence(text, item)` rejects malformed/non-object JSON, duplicate or extra
keys, unknown enums, wrong types, empty/whitespace strings, length violations and
inconsistent field combinations. It rejects input longer than 16,384 characters
before decoding. It checks quote membership **after JSON decoding**, with no
case folding, Unicode normalization, whitespace repair, fuzzy matching, stitched
fragments, or automatic quote correction. Errors use sanitized codes.

The JSON schema bounds individual fields; cross-field consistency and quote
membership are enforced by the parser. A successful parse does **not** establish
that the claim is complete, evidence is relevant, a contradiction exists, support
is absent, or the explanation is correct. Exact quotes can still be cherry-picked,
misleading or unrelated. Conflicting source passages require semantic review.

A future runner must classify malformed/quote-invalid, refused and incomplete
responses as failures with no usable verdict, preserve raw responses and known
usage privately, and report coverage. It must not silently salvage a verdict from
an invalid evidence record, retry, or turn a failure into supported. This module
does not implement that runner; callers must check backend refusal/incompleteness
before invoking the parser. Prior responses never pass through the new parser.

## Synthetic controls and offline checks

`evidence_cases.py` keeps the original ten attribute/numeric short-versus-embedded
inputs unchanged and adds four absence-claim controls:

| Cases | Count | Expected verdict / issue |
| --- | ---: | --- |
| Original supported attribute/numeric pairs | 4 | supported / none |
| Original explicit attribute/numeric contradictions | 4 | unsupported / contradiction |
| Original unknown positive-availability claims | 2 | unsupported / insufficient_support |
| Explicit false attribute, negative-availability claim (short/embedded) | 2 | supported / none |
| Unknown attribute, negative-availability claim (short/embedded) | 2 | unsupported / insufficient_support |

Total: **14 synthetic inputs**, six expected supported and eight expected
unsupported. Context is identical across each short/embedded pair. Added supported
sentences also change answer composition and target position, so the comparison
does not isolate length alone. Expectations are manually authored and offline.
The tests use canned response fixtures to exercise parsing, not model inference;
there is no new synthetic accuracy result.

Run from the repository root, with standard-library Python only:

```bash
python -S -m unittest discover -s tests -p test_llm_judge_evidence_contract.py -v
```

The existing Linux full-suite job and Windows standard-library judge discovery
include these tests. No new packages or GPU environment changes are required.

## Subsequent execution and analysis, not enabled here

1. Implement a separate evidence runner and versioned inference configuration,
   preserving private response/usage records and cumulative budget controls.
   The existing 128-token binary profile must not be silently repurposed for this
   larger schema. Freeze the output allowance and audit the exact new requests.
2. Record a bounded synthetic plan for the 14 fixed inputs before model calls.
   Report valid coverage, verdict/issue correctness, exact-quote failures and
   manual evidence quality. Do not call parser success model accuracy.
3. If the implementation is sound, audit the exact original 50 TRAIN inputs and
   commit a separate bounded pilot configuration before executing. No inherited
   unused time or attempt allowance carries over from completed runs.
4. Report precision/recall/F1 against the unchanged labels with total and
   class-specific coverage, all four persistent misses and all three prior false
   positives, plus any new false positives. Invalid records remain missing.
   Retain disagreement cases; do not relabel them to improve the result.
5. Manually review returned claim completeness, evidence relevance, missing
   support versus contradiction, and explanation faithfulness. These are new
   outputs to assess, not recovered explanations for the prior judge's decisions.

A verdict improvement accompanied by fabricated or misleading evidence is not
sufficient to accept this contract. A separate held-out validation plan is still
needed for final deployment claims. Binary verdicts and counts of issue witnesses
are not calibrated probabilities; the study's continuous-score design remains
unresolved. No test thresholds, HaluBench adaptation or final benchmark results
are introduced by this diagnostic.
