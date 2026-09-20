# Post-thesis calibration and operating-threshold protocol

**Design v1, 2026-09-20. Commit before collecting scores for either reserved arm.**
The [configuration](configs/label_score_fit_v1.json) fixes the statistical choices;
it is not an inference or fitting command. No reserved-arm scores were available
when these rules were specified. The original 50-example pilot was adaptively
inspected and is excluded. This work is not part of the submitted thesis.

## Fixed inputs and roles

The [completed cluster reservation](../../results/post_thesis/llm_judge/development_reservation_cluster_20260920.json)
and its identical replay have SHA256
`56b77ada74b638720586f93835ed801d8f090d7a04b9d1f1272a60e2677d7362`
at allocator revision `6f056d1f0018eb8d7ae0894279dd1074b70f1662`.
The complete hash was independently reconstructed locally. Preserve this manifest.

| Role | Fixed data | Use |
| --- | --- | --- |
| Calibration | 600 responses / 100 components; 319 negative, 281 positive | Fit one fixed monotone map |
| Operating threshold | 600 responses / 100 different components; 324 negative, 276 positive | Select one threshold under the rule below |
| Excluded | 312 responses | No fitting or selection |
| Unallocated | 13,578 responses | No fitting, selection or inference in this experiment stage |

Partition IDs, memberships and exclusions do not change. Labels and metadata stay
offline. The judge receives only the fixed prompt plus answer/context. Retain
Qwen3-32B, BF16/non-thinking, its pinned model/tokenizer revision, the current raw
logprob transport and the primary A=supported/B=unsupported mapping. No swapped
mapping selection, label averaging, new evidence prompt, per-task model or
retraining is introduced. Pin the serving/software identities again in the later
scoring plan; this statistical design does not authorize changes to them.

Let `m = log P(B) - log P(A)` be the unrounded, validated raw margin and `y=1`
mean unsupported. Retain float64 margins. Do not recover margins from rounded
sigmoid scores, round near ties, clip margins, standardize them, or weight rows by
class/task/generator. All responses have equal weight, matching the response-level
evaluation target. Component grouping addresses data allocation; six sibling
responses are not six independent source observations for uncertainty estimates.

## One fixed calibration fit

Fit `p_cal = sigmoid(a*m + b)` on the calibration arm only. Use positive slope
`0.001 <= a <= 100` and `-20 <= b <= 20`. Minimize:

```text
z_i = a*m_i + b
J(a,b) = mean(logaddexp(0, (1 - 2*y_i)*z_i))
         + (0.0001 / 2) * ((a - 1)**2 + b**2)
```

The regularizer is fixed around the identity map `(a,b)=(1,0)`. It is not chosen
by cross-validation or by inspecting these arms. There is no class balancing,
standardization, alternate penalty, temperature-only comparison or isotonic fit.
The loss uses a stable softplus expression and a numerically stable sigmoid.
Analytic derivatives are:

```text
dJ/da = mean((sigmoid(z_i) - y_i)*m_i) + 0.0001*(a - 1)
dJ/db = mean(sigmoid(z_i) - y_i)       + 0.0001*b
```

Use Python 3.12, NumPy 1.26.4 and SciPy 1.14.1 in an isolated future analysis
environment. Call `scipy.optimize.minimize`, method `L-BFGS-B`, analytic Jacobian,
float64 arithmetic, start `(1,0)`, bounds above, and exactly these options:
`ftol=1e-12`, `gtol=1e-8`, `maxiter=10000`, `maxfun=20000`, `maxls=50`, `maxcor=10`.
No random restarts or changed optimizer settings after observing outcomes.
SciPy documents separate function-improvement and projected-gradient stopping
criteria; a success flag alone is not the entire acceptance rule here. See the
[versioned optimizer documentation](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-lbfgsb.html).

Accept only if the optimizer reports success; parameters, objective and gradient
are finite; the maximum absolute analytic gradient is at most `1e-6`; both
parameters are more than `1e-6` from each of their bounds; and the final objective
is no greater than its initial value plus `1e-12`. Otherwise record failure and
retain the failed fit diagnostics. Do not widen bounds, change penalties, restart,
or silently substitute another calibrator. A raw-score evaluation and independent
threshold fit may still proceed under their own registered conditions; report
calibrated results as unavailable rather than relabeling raw scores as calibrated.

A successful positive-slope map preserves exact margin ordering; calibration is
not expected to improve AUROC or average precision. Use raw margins for ranking
to avoid floating-point sigmoid saturation. Calibration quality is evaluated
separately and can worsen out of domain. Never choose whether to show calibrated
results based on which looks better: show raw and successfully fitted calibrated
results together, with the failure policy above if fitting fails.

## One fixed operating-threshold selection

Use only the operating-threshold arm. Select on **raw margin**, so fitting a
positive-slope calibrator cannot alter the selected classification policy.
Predict unsupported when `m >= t` (equality is positive).

Candidates are every distinct observed finite margin, plus a symbolic
`above_max` candidate predicting every row negative. The minimum observed margin
already predicts every row positive. Sort numeric candidates ascending and put
`above_max` last. Compute positive-class F1 as `2*TP / (2*TP + FP + FN)`, with a
zero denominator assigned zero. Maximize that fraction. Compare fractions using
exact integer cross-multiplication, not floating tolerances. If F1 ties, choose
the **largest threshold**, favoring fewer positive predictions on this set.
Process equal margins together; never separate tied rows by ID or label.

Preserve the complete candidate/confusion-count table, the chosen margin and its
exact float representation (`float.hex` alongside round-trippable JSON). Store
`above_max` as a named sentinel, never nonstandard JSON infinity. Both-class
eligibility makes an all-negative optimum impossible, but its definition closes
the candidate set. No calibration-arm or TEST outcome may break a tie.

The deployed comparator remains `raw_margin >= frozen_t`. If calibration succeeds,
record `sigmoid(a*frozen_t+b)` as a descriptive calibrated threshold only; do not
classify using rounded probabilities that can merge distinct margins. Report the
fixed reference policy `m >= 0` as a secondary diagnostic, never choose between it
and the registered F1 policy based on test results. This does not claim that F1
maximization is an optimal deployment policy for every application.

## Eligibility, failures and separation

Before each arm's operation, verify the reservation/config hashes and exact IDs,
input hashes, model, prompt, tokenizer/template, extraction mode, serving profile
and class mapping. Require exactly all 600 unique valid scores from that arm and
both classes. Cross-arm inputs, duplicates, missing or corrupt scores, or unknown
provenance block that operation. Never impute a score, discard failures to fit a
convenient subset, replace examples, borrow siblings, or reroll membership.
The other arm's operation can proceed if its own inputs are complete; no final
combined pipeline is claimed until required artifacts are available.

Any recovery from terminal inference failures needs a separately versioned,
explicitly bounded plan preserving the original attempts. It does not permit
editing outcomes. Record coverage, failure codes and token/time costs before
considering recovery. The eventual inference plan must fix attempt limits and
handling of overlength inputs before scoring. This protocol grants zero calls.

Never use the threshold arm to tune calibration, or the calibration arm to choose
a threshold. Neither arm may tune prompts/models. Selected labels have already
been summarized; these are development sets, not a new blinded validation study.
Calibration-arm reliability measurements are in-sample diagnostics. Threshold-arm
reliability measurements are development diagnostics, and its chosen F1 is
selection performance, not an unbiased performance estimate for the full pipeline.

## Reliability summaries and final evaluation boundary

For raw `sigmoid(m)` and any accepted calibrated probabilities, predefine Brier as
`mean((p-y)**2)`. ECE uses ten fixed equal-width bins with edges
`0, 0.1, ..., 1`: intervals `[left,right)`, except the last `[0.9,1]`.
ECE is `sum(n_bin/n * abs(mean_p_bin - mean_y_bin))`; empty bins contribute zero
and have undefined bin means. Record edges/counts/means and use actual float64
values without rounding before bin assignment. Raw-score ECE is reliability
analysis of an uncalibrated relative class score. Neither ECE nor Brier chooses
calibrator settings or changes the threshold rule.

A separate evaluation manifest must still pin benchmark alignment, exact ranking
metric conventions, source-component uncertainty estimates, baseline coverage,
latency/cost accounting and any cross-benchmark provenance limitations before
TEST scoring. Transfer the exact frozen prompt, accepted calibrator and raw-margin
threshold to the existing canonical HaluBench 8k test split. Do not tune them with
HaluBench labels or create a new split. Raw and calibrated results remain separate.

The reservation establishes disjointness under the recorded native/exact-overlap
components only. Strict underlying document disjointness, fuzzy/partial overlap,
HaluBench overlap and baseline training exposure remain unverified. Subsequent
reports must preserve those limitations and the post-thesis designation.

Next: the [CPU token-length auditor](DEVELOPMENT_TOKEN_AUDIT.md) is implemented for these exact 1,200 reserved inputs and locally validated with the pinned tokenizer. The cluster audit and replay now match; the [separate bounded scoring configuration](DEVELOPMENT_SCORING_RUN.md) pins its identity and permits one request per reserved input. No fitting code, parameter estimates, selected threshold or new model calls are introduced.

Implementation update after completed reserved scoring: [CPU-only fitting](OFFLINE_FITTING.md) now implements this unchanged statistical design. Parameter estimates and the selected threshold remain pending on the pod; the original configuration is preserved as the preregistration artifact.
