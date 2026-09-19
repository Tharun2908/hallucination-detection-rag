# Post-thesis binary pilot: focused source review — 2026-09-19

**Development analysis only; excluded from the submitted thesis.** Read alongside
[the frozen binary pilot result](ragtruth_binary_pilot_20260919.md).

## Selection and evidence

Review the three binary false positives against the original dataset labels,
plus the four persistent misses already identified during probability-pilot
analysis. This selection is label- and outcome-informed. It is not blinded,
independent human annotation or an estimate of error-category prevalence.

Observations were checked against the locally available TRAIN parquet from
`wandb/RAGTruth-processed`, revision
`eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b`. The file SHA256 was verified as
`c14ae31ff459c829edc860bda034ee2dbc0a11107b7511195a32bb4ab1ee8000`.
Original labels and task assignments for all 50 exported IDs match that file and
the preserved paired v1/v2 CSV. This review uses the supplied context and answer;
it does not investigate real-world truth or read benchmark TEST data.

Below are analyst paraphrases of relevant evidence, not judge explanations.
Raw context/answer text stays outside these public result artifacts. The processed
labels identify answer-level conflict/unsupported categories, not an independently
verified rationale for every span. No whole-answer label is changed.

## Persistent false negatives

All four have original label 1 and binary verdict `supported`.

| ID | v1 / v2 score | Source observation | Review conclusion |
| --- | --- | --- | --- |
| 6040 | 0.0 / 0.1 | The answer adds sandwiches and overcooked entrees to its description, neither established by the supplied fields/reviews. It also presents bar-service praise as coming from another reviewer, although it belongs to the same review as the praised appetizers. | Clear unsupported additions and an attribution problem remain undetected at answer level. The earlier sandwich finding was not the only evidence issue. |
| 6486 | 0.0 / 0.1 | WiFi and music fields are unknown, while the answer asserts both are unavailable; the reviews do not establish their absence. | Clear unknown-to-absence error. Unknown is not evidence of non-availability. |
| 12052 | 0.0 / 0.1 | The context describes Q3 as separating the lowest 75% (highest 25% above it), while the answer substitutes the highest 75%. It also assigns the lower-half middle value to the median rather than Q1. | Clear relational/numerical contradictions remain undetected. This is not resolved merely by recognizing the numbers 25/50/75. |
| 9350 | 0.0 / 0.0 | Outdoor seating is unknown in the structured record; none of the supplied reviews establishes it. The answer asserts that seating is available. | Clear unknown-to-availability error. |

These observations show failures on realistic answers despite correct binary
judgments on simplified synthetic examples. They do not identify the model's
internal mechanism, prove it ignored a particular token, or isolate answer length
from context complexity, claim composition or position.

## False positives against original labels

All three have original label 0 and binary verdict `unsupported`. They remain
false positives in the published metrics, irrespective of the observations below.

| ID | v1 / v2 score | Source observation | Review conclusion |
| --- | --- | --- | --- |
| 10044 | 0.2 / 0.2 | Most specific attributes and review summaries in the answer are directly supported. The evidence also contains a closure report followed by a later positive review; the answer's present-tense business description could raise a temporal-status question. | The answer looks largely supported. The source's temporal inconsistency is a possible ambiguity, not proof of why the judge flagged it or sufficient basis to relabel it. Treat as an unresolved apparent false positive. |
| 8469 | 0.2 / 0.25 | The answer credits the business with ingredient quality and custom fillings that are not clearly established. In the review, the customer and their friend prepared fillings. The structured takeout flag is false, but a review explicitly describes collecting an order. | Candidate unsupported elaboration/attribution, plus internally conflicting evidence about takeout. A simple takeout-flag contradiction alone is insufficient to settle the case. Possible annotation disagreement warrants review; the judge's actual trigger is unknown. |
| 2902 | 0.2 / 0.2 | The answer generalizes the article into community outrage and a police-reform discussion. The article's rhetorical opening and comparisons make the precise attribution/scope debatable. The answer also announces a word count that does not match the supplied summary. | Mixed summarization-scope and output-self-description questions. The count concerns the answer's own form rather than source faithfulness, so it should not silently redefine the benchmark label. Keep the apparent false positive unresolved. |

No explanation was requested from the judge. Consequently, none of the candidate
issues above can be attributed to its decision. A fresh explanatory call would be
a separate experiment, not recovered evidence of the original reasoning.

## Consequences for the next design decision

1. Binary elicitation helped sensitivity on this inspected pilot but did not
   repair four clear, previously known grounding failures.
2. Distinguish missing evidence, explicit contradiction, conflicting source
   statements, and plausible summary generalization during further analysis.
3. Preserve disagreement cases and original labels. Independent adjudication,
   with a predefined rubric and reviewer process, would be needed to substantiate
   annotation corrections; this analyst review does not provide that evidence.
4. A later diagnostic could test explicit claim/evidence checking before the
   answer-level verdict. Specify its contract, synthetic controls, token allowance
   and decision rule before execution. Do not infer calibrated probabilities from
   binary verdicts or from a count of flagged claims.

The current update only records findings. No new model requests, source-group
split changes, threshold fitting or final benchmark evaluation are performed.
