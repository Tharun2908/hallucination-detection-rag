# Post-thesis binary TRAIN pilot — 2026-09-19

**Development only. Excluded from the submitted thesis.** This is the same
adaptively inspected 50-example TRAIN sample used for probability v1/v2. It is
not held-out evidence, a new benchmark split, or a final judge configuration.

## Provenance and execution

The operator supplied the completed execution summary and all 50 verdicts in
[the CSV](ragtruth_binary_pilot_20260919.csv). IDs, ordering, original labels and
tasks match the preserved [paired probability CSV](ragtruth_pilot_v1_v2_20260919.csv).
The [machine-readable record](ragtruth_binary_pilot_20260919.json) contains metrics,
class-specific coverage, file checksums, the frozen plan and reported resource
usage. Private response journals and the cluster summary file were not downloaded
for independent inspection. Do not interpret the public record as a journal audit.

- Run: `qwen3-ragtruth-train-pilot-50-binary-v1`.
- Scoring revision: `4b87517a7687df5d8684f9a28db7dbffe4278304`.
- Prompt: `faithfulness-binary-diagnostic-v1`; unchanged from the synthetic run.
- Prompt SHA256: `e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683`.
- Audit SHA256: `55c8b27a10cf0f26c6dfcb35d6882d2c9b400a8fa90bb20689ab7746348f181b`.
- Preparation manifest SHA256: `ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25`.
- Model/profile: pinned Qwen3-32B, BF16, one H200, non-thinking, temperature zero,
  seed zero, sequential requests, at most 128 output tokens per request. Exact
  model/tokenizer/profile pins remain in the [serving profile](../../../post_thesis/llm_judge/SERVING.md)
  and [execution plan](../../../post_thesis/llm_judge/configs/ragtruth_pilot_50_binary_v1.json).

The operator reported 50/50 valid verdicts, 50 new attempts, no failures or pending
inputs, and complete usage: **60,474 input + 400 output = 60,874 tokens**.
Charged client time was **53.16031185211614 seconds**, with
546.8396881478839 seconds of the cumulative allowance remaining. This includes
client preflight and runner bookkeeping, excludes server startup/idle time, and
is not a controlled speed comparison with v1 or v2. Per-attempt timings and server
state were not matched. Rental cost is unknown; unused allowance is not a new
experiment budget. No additional model calls were made for this analysis.

## Results against unchanged labels

`unsupported` is the positive class. Coverage is 50/50 overall, 24/24 for positive
labels and 26/26 for negative labels, with zero failures or pending examples in
either class. These are answer-level decisions, not claim-level detection counts.

| Outcome | Count |
| --- | ---: |
| True positives | 13 |
| False positives | 3 |
| True negatives | 23 |
| False negatives | 11 |

| Metric | Value |
| --- | ---: |
| Precision | 0.8125 |
| Recall | 0.5416666666666666 |
| F1 | 0.65 |
| Accuracy | 0.72 |

Small task slices are descriptive only; they do not establish task robustness.

| Task | n | TP | FP | TN | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QA | 18 | 3 | 0 | 11 | 4 | 1.0000 | 0.4286 | 0.6000 |
| Data2txt | 24 | 9 | 2 | 7 | 6 | 0.8182 | 0.6000 | 0.6923 |
| Summary | 8 | 1 | 1 | 5 | 1 | 0.5000 | 0.5000 | 0.5000 |

False positives against the dataset labels: `10044`, `8469`, `2902`.
False negatives: `14984`, `1772`, `10151`, `6352`, `6040`, `12892`, `6486`,
`9317`, `13151`, `12052`, `9350`. Original labels are retained for all metrics.

## Comparison and interpretation

At the previously illustrated, **unfitted 0.5 threshold**, probability v1 detects
0/24 positives with zero false positives, and v2 detects 3/24 with zero false
positives. Binary detects 13/24 with three false positives. It retains the three
v2 positive detections (`15849`, `6946`, `9256`) and detects ten further labeled
positives. This is a decision comparison at that specific illustrative threshold;
it does not establish superiority over a properly calibrated probability judge.
No threshold search was performed. Binary outputs are strings, not probabilities;
no AUROC, average precision, Brier score or ECE is reported for them.

All four previously reviewed errors (`6040`, `6486`, `12052`, `9350`) still receive
`supported`. The synthetic 10/10 result therefore did not fully transfer to
realistic answers. The output-format change improves sensitivity on this sample,
but the residual failures are consistent with additional grounding difficulties.
The probability-to-binary change also changes prompt wording and schema, so it
does not isolate a single causal mechanism. Neither response format reveals which
claim the model noticed or why it made its decision.

The [focused source review](ragtruth_binary_error_review_20260919.md) distinguishes
clear missed evidence problems from source/annotation ambiguity among the three
apparent false positives. It is label-informed analyst review, not independent
re-annotation. Disputed cases are not removed, relabeled or used to report an
improved alternative metric.

## Reproduce the aggregate counts offline

From the repository root, using only the committed CSV and Python's standard
library (no model, dataset download or API calls):

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path
p = Path('results/post_thesis/llm_judge/ragtruth_binary_pilot_20260919.csv')
rows = list(csv.DictReader(p.open(encoding='utf-8')))
assert len(rows) == len({r['sample_id'] for r in rows}) == 50
assert all(r['status'] == 'ok' and r['verdict'] in ('supported', 'unsupported')
           and r['label'] in ('0', '1') for r in rows)
c = Counter((r['label'], r['verdict']) for r in rows)
tp, fp = c['1', 'unsupported'], c['0', 'unsupported']
tn, fn = c['0', 'supported'], c['1', 'supported']
print(dict(TP=tp, FP=fp, TN=tn, FN=fn))
print(dict(precision=tp/(tp+fp), recall=tp/(tp+fn),
           f1=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/len(rows)))
PY
```

This snippet verifies this fully observed frozen result; it is not a general
failure-handling evaluator for future runs. The full execution runner preserves
failed verdicts as missing, and undefined metrics must remain undefined.

## Decision after review

Preserve probability v1/v2, both synthetic diagnostics and this binary run as
separate development results. The focused seven-case review is complete; the
remaining false negatives are listed but were not exhaustively adjudicated here.
No prompt revision, additional GPU run, threshold fitting or test-set expansion
is introduced by this record. The continuous-score design and final judge freeze
remain unresolved; the canonical test sets and thesis artifacts remain unchanged.

Before another experiment, specify the hypothesis and output contract that could
address a reviewed failure. A possible next diagnostic is structured claim/evidence
checking followed by a binary aggregation decision on the same development inputs.
That is a proposal, not an implemented or budgeted run, and it would not by itself
supply calibrated continuous scores. Prefer one bounded intervention over another
uncontrolled sequence of prompt edits.
