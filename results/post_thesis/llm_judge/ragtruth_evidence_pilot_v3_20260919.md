# Post-thesis evidence-v3 TRAIN pilot findings — 2026-09-19

**Adaptive development only; excluded from the submitted thesis.** The original
50-example pilot completed all 50 attempts with prompt
`faithfulness-evidence-diagnostic-v1`, verdict-first schema v3 and scoring revision
`dbf1bfc0e03dbf0b88d8b0914bbdbb0d224f1808`. This is not a held-out comparison.

## Coverage and missing outputs

| Result | Count |
| --- | ---: |
| Valid evidence records | 37 / 50 (74%) |
| Valid supported verdicts | 27 |
| Valid unsupported verdicts | 10 |
| Context quote not in input | 11 |
| Answer quote not in input | 2 |
| Raw responses in expected field order | 50 / 50 |
| Pending attempts | 0 |

The errors are the first parser failures, not exhaustive lists of all defects.
Several rejected responses contain reconstructed dictionary fragments, new braces
or inserted ellipses. Without comparing every raw quote to its full input, the
exact textual mismatch is not assigned a more specific cause per example.
All 13 rejected responses have raw `unsupported` verdicts and
`insufficient_support` issue types. Their verdicts remain **missing** under the
evidence contract; they are not salvaged or treated as supported.

Against the existing TRAIN labels, retaining every selected example:

| Evidence outcome | Label 0 | Label 1 | Total |
| --- | ---: | ---: | ---: |
| Supported | 20 | 7 | 27 |
| Unsupported | 3 | 7 | 10 |
| Invalid / missing | 3 | 10 | 13 |
| Total | 26 | 24 | 50 |

Thus valid-only counts are TP=7, FP=3, TN=20, FN=7, with another 13 missing.
Do not compare a valid-only score on these 37 selected cases to a full-50 binary
score. Missingness is associated with both the raw verdict and reference label.
The earlier binary run remains a development reference, not independent test
performance. No probability calibration, AUROC, threshold fitting or test metrics
are introduced here.

## Targeted semantic review against full original inputs

The seven cases below were selected for inspection, not sampled randomly. Full
inputs came from the SHA256-pinned TRAIN parquet. Its reconstructed original
pilot manifest matches the recorded digest. Benchmark labels were retained;
these notes are not replacement annotations or independent human adjudication.

| ID | Observation | Assessment |
| --- | --- | --- |
| 17143 | The source explicitly says to flatten the turkey and brush with oil, salt and pepper. | The judge rejects an instruction directly supported by its own quote. Label 0; false positive. |
| 13266 | The source gives a restoration period of 1925–1933, following a sentence about designation in 1925. | Label 0; false positive against the benchmark. The judge's insistence on a distinct later start is not established by the source; exact timing language remains interpretive. |
| 10409 | The checkout review is nested inside the Santa Barbara business record. | The judge ignores that structural association when demanding location confirmation. Label 0; false positive. |
| 6352 | Other reviews praise the coffee and describe the cafe as cute and cozy. Posters advertise local art events. | The explanation overlooks positive review evidence and blurs posters with actual events. Label 1 matches the verdict, but the stated justification is unreliable; other additions in the answer may still be unsupported. |
| 13151 | The first passage explicitly lists turnips among the ingredients. The judge cites a different passage. | The explanation falsely says turnips never appear. Label 1 matches the verdict, but this is the wrong justification. A correct answer-level label does not validate the rationale. |
| 15849 | The full input mentions allergic reactions but never identifies a vaccine as their referent. | The vaccine-specific attribution is unsupported by the supplied context; the core rationale is grounded. This assesses textual support only, not medical truth. |
| 15087 | The full input gives a health-benefits chapter heading, without the claimed medical/dental/prescription details. | The cited unsupported expansion is a reasonable rationale against this supplied context. |

Cases 5584, 13013 and 6946 have locally plausible rationales in their supplied
quote pairs, but this step did not independently review their full inputs. The
27 supported records contain null evidence fields by design: that is not proof
of full support. Seven disagree with positive reference labels and still need
claim-level review. Do not describe all 37 valid records as semantically checked.

Earlier rejected records also expose separate concerns: 9256 interprets unknown
`None` as absence; 11582, 8469 and 10395 describe explicit contradictions while
selecting insufficient support. These observations concern the raw explanations;
they do not restore those records to valid predictions.

## Resource observations and next decision

The operator reported 77,574 input tokens, 4,403 output tokens (81,977 total), no
unknown token usage and 144.001107 client seconds. This window excludes startup
and idle time; rental cost is unknown. The remaining 455.998893 seconds do not
permit retrying the 50 already attempted inputs. Retain the original journal,
summary and budget files unchanged.

The verdict-first order held across all 50 cases, but it did not ensure exact
quotes or correct reasoning. Preserve this negative result and the strict
parser. Do not freeze this evidence candidate for full benchmarks yet.

Any next development revision should be designed separately around short exact
quotes, the existing optional null context quote for insufficient support,
checking the entire supplied context, respecting nested-record associations,
and distinguishing unknown values from explicit contradictions. Test both
verdicts and rationales, including benign paraphrases and support appearing in
another passage. Keep model, schema, inputs and budgets controlled when testing
a prompt change. No new prompt, schema, inference command or scoring allowance
is introduced in this findings step. Continuous-score/calibration design and
native-source overlap auditing remain unresolved.

## Provenance

[Machine-readable results](ragtruth_evidence_pilot_v3_20260919.json) retain the
execution plan, all 50 aligned verdict/status rows, labels and missingness.
Execution and outputs were supplied by the operator; private cluster journals
were not independently inspected. Full raw evidence remains in those journals.
All previous probability, binary and evidence runs remain unchanged.
