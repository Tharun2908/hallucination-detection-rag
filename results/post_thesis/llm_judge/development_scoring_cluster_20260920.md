# Post-thesis reserved TRAIN scoring completed — 2026-09-20

These are post-thesis development execution results, outside the submitted thesis.
The operator supplied both fresh and cached console summaries from scoring commit
`a341e3e805827a1122cc74dec99c66a749543572`.

| Arm | Valid | Input tokens | Output tokens | Client seconds | Cached new attempts |
|---|---:|---:|---:|---:|---:|
| Calibration | 600/600 | 705,198 | 600 | 867.9455 | 0 |
| Operating threshold | 600/600 | 746,401 | 600 | 721.2148 | 0 |

Both arms report zero terminal failures, pending rows, unknown token-usage attempts
and halt reasons. Each stayed within its separate 1,800-second cumulative client
budget. Aggregate recorded client time is 1,589.1603 seconds (26.49 minutes).
This excludes serving startup/idle time and is not an end-to-end GPU billing figure.
No monetary cost is inferred without an hourly rate and complete session duration.

The [machine-readable record](development_scoring_cluster_20260920.json) preserves
all four operator-reported summary hashes. Fresh and cached summaries contain
600 versus 0 `new_attempts`; their different hashes alone do not indicate changed
predictions. The private files were not supplied to the report author, so their
hashes and full responses have not yet been independently reconstructed here.
The [offline fitter](../../../post_thesis/llm_judge/OFFLINE_FITTING.md) will verify
those exact files and raw response replay on the pod before using their margins.

Execution coverage is not predictive accuracy. No calibration parameters,
threshold, AUROC, Brier/ECE or test results can be inferred from these summaries.
The frozen method is now implemented and locally tested on artificial data;
actual fitting remains pending on the pod. No model or prompt was changed, and
no TEST or HaluBench evaluation occurred in this step.
