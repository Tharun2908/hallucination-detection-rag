# Post-thesis: verdict-first live synthetic findings

**Development only; excluded from the submitted thesis.** The operator supplied
the execution summary, field-order report and 14 parsed evidence records. The
[observation JSON](evidence_diagnostic_v3_20260919.json) retains these findings.
Original response strings, journal and server record were not supplied. The
instructed scoring revision was `7c4823789ba0125e85d085baf0e2bab564ca417b`, not
independently verified from the private artifacts.

## Observed comparison on the same 14 synthetic inputs

| Measure | Schema v2 | Verdict-first schema v3 |
| --- | --- | --- |
| Contract-valid records | 14/14 | 14/14 |
| Expected verdict / issue matches | 12/14 / 12/14 | 13/14 / 13/14 |
| True positives / false positives | 8 / 2 | 7 / 0 |
| True negatives / false negatives | 4 / 0 | 6 / 1 |
| Reported target field-order matches | Not measured across all 14 | 14/14 |
| Generation input / output tokens | 13,571 / 918 | 13,571 / 807 |
| Measured client seconds | 80.18416656414047 | 32.9030169933103 |

Positive means unsupported. V3 had 14 new attempts, zero terminal failures,
zero pending inputs, a completed input audit and no unknown token usage. Total
generation usage was 14,378 tokens. Client time includes audit/request work and
excludes server startup/idle; the two single runs do not establish a speedup.
Compilation, cache state and other runtime variation were not separately timed.

The order report was derived by the runner from original response text, not the
alphabetically saved evidence objects. All 14 reported `verdict`, `issue_type`,
`answer_quote`, `context_quote`, `explanation`. This demonstrates observed live
order compliance on these requests, not universal enforcement or identification
of the backend automatically selected by vLLM.

## Manual evidence review

Both v2 false positives, `attribute-supported-short` and
`absence-explicit-false-short`, now have supported/none verdicts with null evidence.
The seven unsupported judgments have quotations and explanations consistent with
the supplied synthetic fixtures. In particular, the previous “unknown or not
offered” explanation in `attribute-unknown-embedded` now describes missing
support without conflating an unknown field with absence.

One new false negative remains: `absence-unknown-embedded` is supported/none with
all-null evidence, although the unknown field does not justify claiming absence.
Its short counterpart is correctly marked insufficient support. This is a
length/context-sensitive failure on this paired fixture; it does not establish
its general frequency or internal cause. The all-null output gives no explanatory
text to inspect. Do not correct its verdict or infer a hidden rationale.

There are seven supported outputs, six correct by the fixture expectations and
one false negative; they provide no explanations to review. The manual review
is descriptive, not an independent annotation study.

## Decision

The two corrections plus one new miss support field order as a useful variable
to investigate. A net gain of one on an adaptively inspected set of 14 is not
statistical or cross-domain evidence of superiority. Preserve the original v1,
v2 and v3 runs, prompts, schemas and labels.

Hold the current evidence prompt v1 / schema v3 fixed for the next paired
**development** comparison on the original 50-example TRAIN pilot. Stop revising
it to chase this small synthetic set. First perform a [token-only pilot audit](../../../post_thesis/llm_judge/EVIDENCE_PILOT.md)
with the 512-token output allowance. No scoring budget is granted by that audit;
record its checksum and lengths before preparing a bounded evidence pilot.

The original 50 examples have already informed probability and binary development;
they are not held-out validation. Context-group exclusions remain unchanged and
the native-source overlap audit remains unresolved. No test sets, new labels,
threshold fitting or final benchmark prompt freeze are introduced. These
categorical verdicts do not solve the continuous-score/calibration question in
the broader LLM-judge research plan.
