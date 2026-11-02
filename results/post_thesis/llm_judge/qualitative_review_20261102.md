# Post-thesis qualitative review: confirmed judgments and unblinding

Recorded 2026-11-02. These findings are separate from the submitted thesis.
The frozen [RAGTruth](ragtruth_frozen_evaluation_20260928.md) and
[HaluBench](halubench_frozen_evaluation_20260929.md) evaluations are unchanged.
The accompanying [JSON record](qualitative_review_20261102.json) contains aggregate
counts, source-file hashes and verification limits.

## Review design and actual procedure

The [disagreement-review protocol](../../../post_thesis/llm_judge/DISAGREEMENT_REVIEW.md)
selected cases deterministically across benchmark task/source, comparison baseline
and correctness state. Both-correct and both-wrong strata provide agreement
controls. Selection occurred after aggregate TEST results were known. The union
contains 127 cases from 124 benchmark-qualified exact-overlap components, below
the maximum of 128. Components can recur across strata. This is a diagnostic
sample, not a representative sample of either benchmark.

The assistant reviewed complete answer/context pairs in opaque-ID order and saved
all 127 judgments in 13 batches on 2026-09-29. Case-level labels, scores, system
predictions and selection memberships were withheld. Aggregate findings were
already known, and the domain could be inferred from the text. No external factual
lookup or reconstruction of missing questions was used.

On 2026-11-02 the researcher stated in the conversation that they had checked the
document and agreed with all decisions. This collective confirmation preceded
opening the uploaded key in the review conversation. It is **human-confirmed
AI-assisted review**, not independent human annotation, an independently logged
second pass over every context, or an inter-rater reliability study. The exact
assistant model/version was not captured; it must not be inferred retrospectively.

The key was then joined by `review_id`. Every original judgment and caveat was
preserved. Confirmation retains an `uncertain` decision as uncertain. Historical
pending-confirmation fields in the original review are retained; the later
confirmation is a separate record.

## Findings within the selected packet

| Benchmark | Cases | Supported | Unsupported | Uncertain | Binary review agrees with label | Binary review differs from label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAGTruth | 48 | 15 | 32 | 1 | 31 | 16 |
| HaluBench | 79 | 13 | 26 | 40 | 33 | 6 |
| Total | 127 | 28 | 58 | 41 | 64 | 22 |

There are 86 determinate binary review decisions. Of these, 64 agree with the
benchmark label and 22 differ. The remaining 41 cases are excluded from binary
agreement counts, not assigned a default binary label.

| Direction of discrepancy | RAGTruth | HaluBench | Total |
| --- | ---: | ---: | ---: |
| Review unsupported, benchmark label 0 | 16 | 4 | 20 |
| Review supported, benchmark label 1 | 0 | 2 | 2 |

Here benchmark label 1 denotes hallucinated/unsupported and label 0 the converse.
**These are review–label discrepancies, not 22 established annotation errors.**
A discrepancy can reflect a benchmark issue, review error, a stricter support
criterion, ambiguous wording or missing information. No benchmark labels or
reported performance estimates were changed.

### Missing questions and interpretable answer relations

Forty of the 41 uncertain cases are from HaluBench: 15 DROP, 13 FinanceBench,
eight halueval and four covidQA cases. These include bare numbers, names, repeated
list entries and fragments whose intended relation is not stated. Some financial
table excerpts also lack the headings or dates needed to resolve an answer.
For example, a percentage may match a plausible calculation without establishing
which metric and years were requested. Exact textual presence is not sufficient
to resolve an unspecified relationship. The sole uncertain RAGTruth case concerns
listed business hours alongside dated closure reports.

This identifies an interpretation limit in these selected cases under the fixed
answer/context-only interface. It does not imply that every short answer requires
a question or that the same uncertainty rate holds across HaluBench. Introducing
questions would be a separate protocol, not an adjustment to these frozen results.

### Scope, attribution and unsupported additions

Recorded discrepancies include extending a free-admission weekend to an entire
event, changing an agreed settlement into an amount sought in damages, reversing
a source comparison and adding amenities absent from the supplied description.
These are descriptions of reviewed text, not automatically established causes
of a verifier's prediction.

Some retained judgments are narrower: declared versus paid dividends, an ambiguous
rental-unit antecedent, or a categorical causal conclusion drawn from an observed
association. Collective confirmation does not remove those interpretive caveats.
They should not be presented as equally conclusive evidence of benchmark error.

### Consensus is not adjudication

The private case inventory contains both kinds of disagreement with consensus:
an answer with ungrounded amenities that all four systems flag despite label 0,
and a settlement/status mismatch that all four systems accept despite the review
finding it unsupported. Another case is labeled 1, while all four systems and the
review accept a proposition supported by the full supplied context. These examples
illustrate the need to inspect evidence; they do not establish a new system ranking.

## Verification and provenance limits

The analysis checked all 127 unique IDs and exact set equality across packet,
frozen review and key, packet order, exact quote membership, and the packet's file
SHA256 against the saved review. Selection memberships agree with the stored
predictions and benchmark labels. Quote membership alone does not prove a verdict.

During repository integration, the serialization in `prompts.content_hash` and
opaque-ID construction in `disagreement_review.build_review` were inspected at
commit `e051c37e88e25634b53f2ab9c4ec77448bf856b2`. All 127 answer/context hashes match
the key, and all 127 opaque IDs reproduce from the protocol seed, benchmark and
sample ID. This resolves the input-serialization check left open in the earlier
private join report. That earlier report is preserved unchanged. These checks do
not independently rerun the upstream selection or attest historical inference.

The following private artifacts are fingerprinted in the public JSON:

- `review_packet.json` and `review_key.json`;
- `post_thesis_ai_review_complete_127.json`, the frozen pre-unblinding annotations;
- `post_thesis_review_unblinded_20261102.json`, the later confirmation and join;
- `post_thesis_qualitative_review_20261102.md`, the detailed private case report;
- `join_post_thesis_review.py`, the standard-library reproduction script.

The full-population review report hash supplied by the operator was
`e90b388df8a4ab617d93890aff3da29a40aada7a9fd21bc62db7609de5f12511`.
That private report was not provided for independent hash recomputation during
this integration. Its reported hash is not interchangeable with a raw file hash.
The public JSON is an aggregate derivative, not the private report or case archive.

## Reporting boundaries

- The actual review was AI-assisted and collectively confirmed, not independently
  annotated by two reviewers. No inter-rater statistic is reported.
- The selected cases are not representative. These counts do not estimate
  population label-error rates or replace the full-benchmark comparisons.
- No confidence interval or model ranking is inferred from this selected review.
- No model calls, prompt edits, fitting, threshold changes, benchmark relabeling
  or revisions to frozen TEST metrics were performed.
- Evidence-window truncation, empty-pair preprocessing and runtime differences
  remain possible contextual factors, not demonstrated causes of these cases.

The qualitative result supplements the frozen evaluation: apparently simple
binary labels can conceal ambiguity in input interpretation and support criteria.
The original benchmark results remain the quantitative findings of this study.
