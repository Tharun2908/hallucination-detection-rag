# Post-thesis: evidence schema-v2 synthetic findings

**Development only; excluded from the submitted thesis.** This is a repeated,
adaptively inspected set of 14 synthetic examples, not a benchmark or held-out
evaluation. The [operator-supplied records](evidence_diagnostic_v2_20260919.json)
contain 14 parsed evidence objects and original response text for two errors.
The remaining raw responses and the private journal were not supplied.

## Observed result

| Measure | Evidence schema v1 | Evidence schema v2 |
| --- | --- | --- |
| Contract-valid records | 11/14 | 14/14 |
| Terminal invalid outputs | 3 | 0 |
| Expected verdict matches among valid records | 11/11 | 12/14 |
| Expected issue matches among valid records | 11/11 | 12/14 |
| Expected supported examples correctly classified | 3, with 3 missing | 4, with 2 false positives |
| Expected unsupported examples correctly classified | 8/8 | 8/8 |
| Generation input tokens | 13,571 | 13,571 |
| Generation output tokens | 933 | 918 |
| Measured client seconds | 39.85682361200452 | 80.18416656414047 |

V2 completed its single invocation with 14 new attempts, no failures or pending
inputs, and no unknown token-usage records. Total generation usage was 14,489
tokens. Client time excludes server startup/idle time and includes audit and
request work; these single runs do not establish a latency regression or its
cause. Runtime compilation/cache state was not separately measured.

All inputs, prompt text, parser and model profile were retained. The intended
scoring revision was `afa92aeebdb071e0d2f35eaf19c6fb6d19e113e9`; the supplied
console does not independently verify the revision stored in private artifacts.
The serving session was `server-c1d6c592edde4952b674469c0b894658`. The automatic
backend-selection policy was retained; the selected compiler is not attested.

## Manual evidence review

| Example | Observation |
| --- | --- |
| `attribute-supported-short` | The answer says seating is available, the quoted field is True, and the explanation confirms availability. The unsupported/contradiction labels conflict with all three. |
| `absence-explicit-false-short` | The answer says seating is unavailable, the quoted field is False, and the explanation confirms unavailability. The unsupported/contradiction labels are again inconsistent. |
| `attribute-unknown-embedded` | Correct unsupported/insufficient-support labels, but the explanation still says None means “unknown or not offered.” Unknown does not establish absence. |

No additional concern was identified when comparing the other 11 records with
these synthetic fixtures. This is a descriptive review, not independent
annotation or proof of correctness on other inputs. Four supported outputs have
null evidence fields, so they provide no explanatory text to assess.

The two false positives retain the exact quotes and explanations from their
previous invalid supported outputs; their issue type and verdict have changed.
The numeric supported short example now has all-null evidence and a supported
verdict. Historical v1 invalid outputs remain missing in official comparisons;
their raw verdicts are not retroactively counted as valid predictions.

## Field-order hypothesis

Both supplied v2 raw responses generate members in this order:
`answer_quote`, `context_quote`, `explanation`, `issue_type`, `verdict`.
Once a non-null answer quote is emitted, no valid completion of schema v2 can
choose the supported branch, which requires null evidence fields. The final
unsupported verdict is consistent with this structural constraint even though
the explanation supports the answer. This is evidence for investigating an
early field choice that excludes the correct branch; it does not reveal why the
model chose that quote or prove that reordering will repair the judgment.

Repository inspection found that `canonical_json()` sorts dictionary keys.
The schema's property ordering and saved parsed records are therefore
alphabetized; a printed parsed object alone cannot establish generation order.
The two raw strings supply that evidence for these two cases only.

Next: an [offline verdict-first candidate](../../../post_thesis/llm_judge/EVIDENCE_ORDER_V3.md)
with identical validation rules and prompt. Verify native order behavior before
preparing any new bounded model run. No TRAIN/test scoring, threshold fitting,
probability conversion, manual verdict correction or final prompt freeze follows
from the current results.
