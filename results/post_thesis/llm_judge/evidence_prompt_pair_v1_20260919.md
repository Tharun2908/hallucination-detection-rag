# Post-thesis paired evidence prompt findings — 2026-09-19

**Adaptive synthetic development only; excluded from the submitted thesis.**
Both prompts completed the same 26-case comparison under unchanged schema v3
and the evidence-v1 model profile. The inputs were audited before generation,
and which prompt ran first alternated by case. This report records the observed
benefit and the rationale regression, without declaring a general winner.

## Paired observations

| Observation | Prompt v1 | Prompt v2 |
| --- | ---: | ---: |
| Valid records | 26/26 | 26/26 |
| Expected verdict matches | 24/26 | 25/26 |
| Expected issue-type matches | 24/26 | 25/26 |
| Raw order matches (operator-reported) | 26/26 | 26/26 |
| Insufficient-support records | 6 | 7 |
| Null context quotes within those records | 0/6 | 4/7 |
| Original 14: expected verdict matches | 13/14 | 14/14 |
| New 12: expected verdict matches | 11/12 | 11/12 |
| Input tokens | 23,853 | 23,931 |
| Output tokens | 1,460 | 1,547 |

All 52 exported evidence records were revalidated against the exact full fixtures.
There are no missing predictions in this run. Both prompts correctly classify
all 12 expected-supported inputs. V1 misses two of the 14 expected-unsupported
inputs; v2 misses one. The only verdict change is the embedded unknown-absence
case below. There is also a rationale regression that verdict counts conceal.

## Three cases that determine the interpretation

### Embedded absence: a correction in this run

`absence-unknown-embedded` claims that the venue does not offer outdoor seating.
The full context contains `OutdoorSeating: None` and unrelated supported facts;
it provides no evidence of absence. V1 returns supported. V2 returns unsupported
with insufficient support and correctly explains that unknown is not false.
This corrects the previously observed verdict failure on this particular fixture.
It does not establish that the broader class of absence errors is solved.

### Embedded positive claim: a rationale regression

`attribute-unknown-embedded` claims that the venue offers outdoor seating against
the same unknown field. Both prompts return the expected unsupported verdict and
insufficient-support issue type. V1 correctly explains the absence of support.
V2 instead says None indicates that the venue does not offer outdoor seating.
That is an incorrect inference of absence, and it is inconsistent with the
insufficient-support distinction. Its null context quote is schema-valid but
does not make the explanation correct. Correct verdict and issue labels conceal
this semantic defect. The explanation must remain recorded unchanged.

### False claim about the context: a shared miss

`false-absence-assertion` says the context gives no information about the archive's
opening time. The second passage explicitly says the archive opens at nine in
the morning. Both prompts return supported with null evidence fields, missing
the contradiction. The fact that the answer describes the context itself does
not exempt it from faithfulness verification.

## Full-fixture review and limits

All 26 inputs and both exported outputs were reviewed. Beyond the three cases
above, no additional semantic issue was identified in this assistant review.
V1's 12 unsupported explanations were consistent with their fixtures. Of v2's
13 unsupported explanations, 12 were consistent and one contained the clear
unknown-as-absence error. Supported outputs were checked against the whole
fixture answer even though their explanation fields are null by contract.
These are descriptive review notes, not independent human ratings or a new
validated semantic-accuracy metric.

Both prompts handle the new paraphrase, cross-passage support, nested entity,
review-praise, event-advertisement and unknown-field-resolution controls in this
small suite. Both also ignore the injected instruction in its single synthetic
control; this does not establish general prompt-injection robustness. The new
controls are short and cannot establish recovery from the longer TRAIN failures.
The aggregate gain of one verdict therefore does not demonstrate overall semantic
superiority. Each prompt has two identified problematic case outputs: v1 has two
verdict misses; v2 has one miss and one incorrect unsupported rationale.

V2 uses null context quotes more often, as requested by its prompt. This can
reduce extraction demands, but both arms already achieved full validity here.
Neither improved TRAIN quote coverage nor a causal benefit from any individual
prompt sentence has been established. The combined prompt revision changes
multiple instructions, not one isolated factor.

## Resources and provenance

The operator reported one completed 52-attempt invocation, zero failures/pending
requests and no unknown token usage. Combined usage was 47,784 input and 3,007
output tokens (50,791 total), with 135.697805 client seconds for the shared window.
The window includes preflight/audit and client work, excludes server startup and
idle time, and is not an arm-specific latency comparison. Rental cost is unknown.

Scoring revision: `6abf9c9eb7e809341c61e5ca62d95c1a3f8fa877`.
Plan SHA256: `009518138bee162010c1b71d9597e4df61e04ee86dd3b5d968f0134f70b171ce`.
Fixture SHA256: `afe59f1545bbd3866d368372ea3c43ad0028398ce1b6b3df87016cae35731594`.

[Machine-readable observations](evidence_prompt_pair_v1_20260919.json) preserve all
26 paired exported records, review tags, both prompt identities, the exact plan
and the attachment checksum. The cluster journals were not independently
inspected. Raw field-order counts come from the operator's console; the parsed
export's member order cannot independently verify generated order. Raw response
strings and server resource records remain authoritative on the cluster.

## Next development step

Preserve both prompts and this mixed result. Stop tuning on these 26 fixtures.
Keep v2 and schema v3 fixed as an experimental candidate for checking extraction
and reasoning on the original 50 TRAIN development examples. That requires a
separate token-only audit, followed by a recorded scoring budget after lengths
are known. The aim is to test whether the synthetic behavior transfers to those
longer inputs, not to claim an independent validation gain.

This findings change adds no generation command, budget or new prompt. Retain
the known false-absence miss and unknown-as-absence rationale regression in the
next review. There is no benchmark freeze, threshold fitting or test evaluation;
continuous-score/calibration design and native-source overlap auditing remain
unresolved. All prior artifacts and submitted thesis results stay unchanged.
