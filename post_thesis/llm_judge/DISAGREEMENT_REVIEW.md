# Post-thesis frozen TEST disagreement review

This is post-thesis work and is not part of the submitted thesis results.
This procedure is defined **after** inspecting aggregate RAGTruth and HaluBench
TEST results. It is descriptive error analysis, not independent validation or
permission to tune the evaluated judge.

## Fixed inputs and counts

`disagreement_review.py` loads both completed evaluations and their exact input
manifests, verifies the pinned report hashes, replays saved judge responses
read-only, and reuses the frozen evaluation decision functions. The aligned
input digest and all four confusion matrices must match the completed reports.
It does not recompute the bootstrap, load model weights, make network requests,
fit anything, or change thresholds, labels, prompts or original artifacts.

It reports all 2,700 RAGTruth and 8,000 canonical HaluBench rows:

- Judge versus each baseline: both correct, judge only correct, baseline only
  correct, both wrong. Here “correct” means agreement with the benchmark label.
- All four correct, all four wrong, and mixed decisions, plus all observed
  four-bit decision patterns in the recorded system order.
- The same counts by RAGTruth task and HaluBench source.

Four binary classifiers cannot all disagree pairwise. Mixed decisions express
that situation without inventing an “all disagree” category.

## Reproducible manual sample

For each of the three RAGTruth tasks and five HaluBench sources, select two cases
from each correctness state for judge versus MiniCheck and judge versus
metadata-free fusion. Rank candidates by a fixed SHA256 key, defined in `POLICY`,
without using score magnitude or reading their text. Select at most one response
per exact-overlap component within a stratum. Short strata stay short; do not
redistribute quotas. Deduplicate across the two comparisons while recording all
selection memberships. The upper bound is **128 unique cases**; the actual count
can be smaller. No command-line quota or seed search is offered.

The sample deliberately balances disagreement types and sources. It does not
estimate their population frequencies. Components can recur across strata;
the full-population counts provide denominators. Both-correct and both-wrong
pair strata provide agreement controls; the other two classifiers can differ.

Selection is fixed before opening the review cases, but after aggregate TEST
results were known. The source slice itself is not a semantic error category.

## CPU commands

After committing and pulling this change, use the existing pinned CPU fitting
environment. No Qwen server, CUDA toolkit or GPU pod is required.

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.disagreement_review
```

Repeat the last command to verify identical outputs with zero new files. Both
runs verify source artifacts; no model calls are made. If a recreated pod has a
broken environment interpreter symlink, restore that interpreter first; do not
replace the frozen package versions.

Outputs are private under:

```text
.artifacts/post_thesis/llm_judge/frozen-test-disagreement-review-v1/
    report.json
    creation.json
    review_packet.json
    review_key.json
    annotation_template.json
```

`report.json` contains population counts, selected IDs and strata, hashes and
limitations. `creation.json` records the initial code revision and package
versions. Outputs are immutable; reruns refuse changed generated files and
leave separately named annotation files alone. If interrupted while exporting,
rerun the same command to finish missing files without changing existing ones.

## Human review

1. Copy `annotation_template.json` to `annotations_reviewer_1.json`. Keep the
   template unchanged. Read `review_packet.json` and the instructions here;
   keep `review_key.json` and the selected-case strata closed initially.
2. Review the complete answer against the complete supplied context. Judge
   support by this context, not external factual correctness. Record
   `independent_verdict` as `supported`, `unsupported`, or `uncertain`.
3. Fill reviewer/date, rationale and applicable categories: `unsupported_addition`,
   `contradiction`, `partial_support`, `absence_claim`, `numeric_claim`, `paraphrase`,
   `long_context_grounding`, `source_specific`, `other`. These are overlapping
   descriptive tags, not automatically detected causes. Use an empty category
   list when no tag applies. Explain `other` in the rationale.
4. Copy minimal exact answer/context quotes where useful. Use null when no
   supporting context span exists; do not manufacture a quotation of absence.
   Missing information is not an explicit negative assertion. Check whether
   other passages support the claim before marking it unsupported.
5. Save the independent annotations before opening the key. Then compare with
   labels and frozen predictions in a separately named adjudication file. Keep
   uncertainty or suspected annotation problems explicit. Do not overwrite
   benchmark labels or update published metrics from these judgments.

The packet hides source names, labels and model outcomes, but domain may be
recognizable from the text. A reviewer familiar with previous cases is not fully
blind. If a second reviewer is available, annotate independently before
adjudication; otherwise describe the work as single-reviewer qualitative analysis.
A category, plausible rationale or matching quote alone does not prove why a
model failed. Evidence-window truncation and S2's existing empty-pair fallback
require checking recorded preprocessing before attributing errors to them.

This step generates the pack only. It does not auto-fill semantic categories or
claim the manual review is complete. Share the console summary first; no need to
paste all full-context records or commit the private packet publicly.
