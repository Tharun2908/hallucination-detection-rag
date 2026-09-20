# Post-thesis: evidence prompt-v2 original TRAIN pilot findings

**Adaptive development only; excluded from submitted thesis results.**
The same original 50 TRAIN examples have already informed several diagnostic
iterations. These findings do not estimate held-out generalization. Keep prompt
v2 fixed and pause further evidence-prompt tuning.

## Provenance and reproducibility

Scoring revision: `2265ac33683274383a42ec0be2e8df3a91fde86b`.
Prompt: `faithfulness-evidence-diagnostic-v2`; schema: v3, verdict first.
Model/profile: unchanged pinned Qwen3-32B, `evidence-v1`, one H200.
The [execution plan](../../../post_thesis/llm_judge/configs/ragtruth_pilot_50_evidence_prompt_v2.json)
and full identities are copied into the [machine-readable findings](ragtruth_evidence_prompt_v2_pilot_20260920.json).

The operator supplied the execution summary and all 50 exported records: 43
parsed evidence objects and seven rejected raw responses. The private cluster
journal and server resources were not independently inspected. The supplied
objects reproduce all 43 successful parses and seven first validation errors
against the checksum-pinned TRAIN parquet. Labels and order agree with the
committed v1 report, and the reconstructed original manifest matches its pin.
The raw-order count below comes from the operator's console; parsed objects
cannot establish the original wire order of successful responses.

The accompanying JSON preserves all supplied evidence objects, rejected raw
responses, offline labels, paired statuses and targeted assistant review notes.
Full inputs remain in the existing private data artifacts. This is targeted
assistant review, not exhaustive claim-level annotation or independent human
adjudication. No model calls were made while preparing this report.

## Structural coverage and resources

| Quantity | Evidence prompt v1 | Evidence prompt v2 |
| --- | ---: | ---: |
| Attempted examples | 50 | 50 |
| Valid evidence records | 37 (74%) | 43 (86%) |
| Terminal failures | 13 | 7 |
| Pending examples | 0 | 0 |
| First error: context quote absent | 11 | 5 |
| First error: answer quote absent | 2 | 2 |
| Reported correct raw field order | 50 | 50 |
| Known input tokens | 77,574 | 77,724 |
| Known output tokens | 4,403 | 4,097 |
| Charged client seconds | 144.0011 | 211.9602 |

Both runs report no attempts with unknown token usage. V2 finished within its
600-second cumulative allowance, leaving 388.0398 seconds; this remainder is not
permission to retry terminal failures. Client timings include preflight/runner
windows but exclude server startup and idle time. No rental price was supplied.
A single pair of runs does not establish a repeatable latency difference.

The seven remaining invalid outputs were also invalid in v1. They remain missing;
their raw verdicts are not predictions. No previously valid record became invalid.
All six recovered IDs have positive offline labels and valid unsupported verdicts:
`11000`, `11031`, `9317`, `9256`, `10395`, `11363`. Four use a null context quote;
two use exact excerpts. In all valid v2 insufficient-support records, 5/8 use a
null context quote. Null is allowed by the contract but does not prove support
is absent from the full context.

## Paired label agreement, with missingness explicit

| Outcome across the same 50 | V1 | V2 |
| --- | ---: | ---: |
| Valid and label-correct | 27 | 32 |
| Valid and label-incorrect | 10 | 11 |
| Missing due to invalid output | 13 | 7 |

V2 valid-only confusion counts are TP 12, FP 3, TN 20, FN 8. Missing outputs
comprise four positive and three negative labels. These descriptive counts are
not full-coverage classifier metrics; do not discard missing rows when making
claims about comparative performance.

On the **37 examples valid under both prompts**, label agreement changes from
**27/37 to 26/37**. Three verdicts change:

| ID | Offline label | V1 | V2 | Interpretation |
| --- | ---: | --- | --- | --- |
| 10409 | 0 | unsupported | supported | Corrected verdict; v1 questioned a review's business association |
| 13151 | 1 | unsupported | supported | Label-agreement regression; v1's matching verdict already had a wrong rationale |
| 14682 | 0 | supported | unsupported | New false positive involving the scope of “solely” |

The all-50 gain is explained by six recovered label-matching verdicts offset by
one net loss on shared valid examples. It does not establish better semantic
reasoning. Label agreement alone also does not validate the selected evidence.

## Targeted semantic findings

- **17143:** Persistent rejection of supported instructions. The judge quotes
  support for brushing turkey with oil, salt and pepper, then excludes it because
  it is outside the numbered instructions. All supplied context is admissible.
- **13266:** The objection about restoration timing becomes a contradiction
  claim. A restoration-period heading of 1925–1933 does not establish that work
  did not begin in 1925; the explanation lacks explicit conflicting evidence.
- **6352:** The full reviews praise coffee and the cozy atmosphere. The rationale
  overlooks this support. Posters advertising art events versus hosting events
  is a separate distinction that the rationale does not identify.
- **9317:** Valid extraction and a matching positive label, but a questionable
  rationale. A review describes blasting music; “can be loud” does not require
  evidence that the event recurs. Other unsupported additions in the answer are
  not established by the selected explanation.
- **9256:** The recovered rationale plausibly identifies missing support for
  reservation channels and takeout. It no longer treats an unreported field as
  proof of unavailability.
- **13151:** V1 incorrectly said turnips were absent despite the first passage
  mentioning them. V2 removes that objection but accepts the whole answer,
  including an unsupported garnish instruction and an absence claim about potato
  cooking times despite a 5–7 minute blanching instruction in the context.
- **14682:** The context lists several group characteristics. “Not solely based
  on race” is a faithful paraphrase; the judge misreads the scope of “solely.”
- **11363:** Only one review text praises service; the five-star review praises
  food. Rejecting majority praise of service is defensible, but mixed star
  ratings alone are not the right evidence for that conclusion.
- **10044:** The invalid raw output asserts contradiction while its explanation
  concludes the selected claim is supported. This remains a missing output.
- **2902:** The invalid raw rationale reads a conviction into “arrested and
  charged.” Those words alone do not imply conviction. Do not salvage its verdict.
- **7629:** The invalid explanation groups an unknown outdoor-seating field with
  explicitly unavailable WiFi as contradictions. Unknown alone is not conflict.

These notes illustrate persisting errors and limited improvements, not an
exhaustive correctness certification of the other outputs.

## Decision and next research step

Preserve both runs, all seven missing v2 outputs and the unchanged exact-quote
parser. Pause further evidence-prompt tuning on this repeatedly inspected pilot.
The evidence diagnostic has identified failure mechanisms; the candidate is
not ready for a benchmark freeze.

The next step is a written continuous-score design before implementation:
define what the score represents, how it is extracted, how failed scoring stays
missing, and how calibration and operating thresholds use development data
only. A continuous ranking score can support AUROC/AUPRC without being a
calibrated probability; calibration requires a separate, explicit protocol.
Binary verdicts must not be relabeled as probabilities. The original 50 remain
development data. Native-source overlap controls and the existing canonical
HaluBench split must be resolved/respected before held-out evaluation.

This report introduces no new prompt, model, generation allowance, threshold
fitting, test metrics or change to submitted thesis results.
