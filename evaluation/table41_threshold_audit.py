import json
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Helpers
# ============================================================

def load(path):
    with open(path) as f:
        return json.load(f)


def ece(scores, labels, n_bins=10):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (scores >= bins[i])
                & (scores <= bins[i + 1])
            )
        else:
            mask = (
                (scores >= bins[i])
                & (scores < bins[i + 1])
            )

        if mask.sum() == 0:
            continue

        total += (
            mask.sum()
            * abs(
                labels[mask].mean()
                - scores[mask].mean()
            )
        )

    return total / len(labels)


def threshold_metrics(y, pred):
    return {
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
    }


def ranking_metrics(y, hall_scores):
    return {
        "auroc": roc_auc_score(
            y, hall_scores
        ),
        "auprc": average_precision_score(
            y, hall_scores
        ),
        "ece": ece(
            hall_scores, y
        ),
    }


def sweep_support(
    train_support,
    train_y,
    thresholds,
):
    """
    High score = support/faithfulness.
    Predict hallucination when support < threshold.
    """
    best_t = None
    best_f1 = -1

    for t in thresholds:
        pred = (
            train_support < t
        ).astype(int)

        f1 = f1_score(
            train_y,
            pred,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def sweep_hall(
    train_scores,
    train_y,
    thresholds,
):
    """
    High score = hallucination.
    Predict hallucination when score >= threshold.
    """
    best_t = None
    best_f1 = -1

    for t in thresholds:
        pred = (
            train_scores >= t
        ).astype(int)

        f1 = f1_score(
            train_y,
            pred,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def report(
    name,
    threshold,
    train_f1,
    y_test,
    hall_test,
    pred_test,
):
    tm = threshold_metrics(
        y_test,
        pred_test,
    )

    rm = ranking_metrics(
        y_test,
        hall_test,
    )

    result = {
        "threshold": threshold,
        "train_f1": train_f1,
        **tm,
        **rm,
    }

    print(
        f"{name:<22} "
        f"thr={threshold:>5.2f}  "
        f"P={tm['precision']:.4f}  "
        f"R={tm['recall']:.4f}  "
        f"F1={tm['f1']:.4f}  "
        f"AUROC={rm['auroc']:.4f}  "
        f"AUPRC={rm['auprc']:.4f}  "
        f"ECE={rm['ece']:.4f}"
    )

    return result


results = {}


# ============================================================
# S1 — NLI
# high nli_score = support
# original script grid: 0.10 ... 0.90 step .05
# ============================================================

tr = load(
    "/workspace/nli_results_train_v2.json"
)
te = load(
    "/workspace/nli_results_test_v2.json"
)

tr_support = np.array(
    [float(r["nli_score"]) for r in tr]
)
te_support = np.array(
    [float(r["nli_score"]) for r in te]
)

tr_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in tr]
)
te_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in te]
)

grid = np.arange(
    0.10, 0.901, 0.05
)

t, train_f1 = sweep_support(
    tr_support,
    tr_y,
    grid,
)

results["S1"] = report(
    "S1 NLI",
    t,
    train_f1,
    te_y,
    1.0 - te_support,
    (te_support < t).astype(int),
)


# ============================================================
# S2 — Relevance MIN aggregation
#
# IMPORTANT:
# Table 4.1 uses raw_min_relevance, not the mean relevance_score.
#
# Normalize using TRAIN min/max only.
# ============================================================

tr = load(
    "/workspace/relevance_results_train_v2.json"
)
te = load(
    "/workspace/relevance_results_test_v2.json"
)

tr_raw = np.array(
    [float(r["raw_min_relevance"]) for r in tr]
)
te_raw = np.array(
    [float(r["raw_min_relevance"]) for r in te]
)

tr_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in tr]
)
te_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in te]
)

mn = tr_raw.min()
mx = tr_raw.max()

tr_support = np.clip(
    (tr_raw - mn) / (mx - mn),
    0.0,
    1.0,
)

te_support = np.clip(
    (te_raw - mn) / (mx - mn),
    0.0,
    1.0,
)

grid = np.arange(
    0.10, 0.901, 0.05
)

t, train_f1 = sweep_support(
    tr_support,
    tr_y,
    grid,
)

results["S2"] = report(
    "S2 Relevance",
    t,
    train_f1,
    te_y,
    1.0 - te_support,
    (te_support < t).astype(int),
)

results["S2"]["train_min"] = float(mn)
results["S2"]["train_max"] = float(mx)


# ============================================================
# S3
# ============================================================

print(
    "\nS3 Cross-model        "
    "SKIPPED — current v5 cache contains only 10 rows"
)


# ============================================================
# S4 — supervised verifier
# high = hallucination
# OOF train predictions used for threshold selection
# ============================================================

tr = load(
    "/workspace/signal4_results_train_oof.json"
)
te = load(
    "/workspace/signal4_results_test.json"
)

tr_score = np.array(
    [float(r["signal4_score"]) for r in tr]
)
te_score = np.array(
    [float(r["signal4_score"]) for r in te]
)

tr_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in tr]
)
te_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in te]
)

grid = np.arange(
    0.05, 0.951, 0.05
)

t, train_f1 = sweep_hall(
    tr_score,
    tr_y,
    grid,
)

results["S4"] = report(
    "S4 Fine-tuned",
    t,
    train_f1,
    te_y,
    te_score,
    (te_score >= t).astype(int),
)


# ============================================================
# S5 — BERTScore precision
# high = support
# canonical script grid = 0 ... 1 step .01
# ============================================================

tr = load(
    "/workspace/signal5_v2_precision_results_train_mean.json"
)
te = load(
    "/workspace/signal5_v2_precision_results_test_mean.json"
)

tr_support = np.array(
    [float(r["signal5_score"]) for r in tr]
)
te_support = np.array(
    [float(r["signal5_score"]) for r in te]
)

tr_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in tr]
)
te_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in te]
)

grid = np.arange(
    0.0, 1.001, 0.01
)

t, train_f1 = sweep_support(
    tr_support,
    tr_y,
    grid,
)

results["S5"] = report(
    "S5 BERTScore",
    t,
    train_f1,
    te_y,
    1.0 - te_support,
    (te_support < t).astype(int),
)


# ============================================================
# S6 — distilled MiniCheck verifier
# stored internally as signal8_score
# high = hallucination
#
# train file contains 13,581 actual model-training rows;
# validation rows were separated for checkpoint selection.
# ============================================================

tr = load(
    "/workspace/signal8_results_train.json"
)
te = load(
    "/workspace/signal8_results_test.json"
)

tr_score = np.array(
    [float(r["signal8_score"]) for r in tr]
)
te_score = np.array(
    [float(r["signal8_score"]) for r in te]
)

tr_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in tr]
)
te_y = np.array(
    [int(r["ground_truth_hallucination"]) for r in te]
)

grid = np.arange(
    0.05, 0.951, 0.05
)

t, train_f1 = sweep_hall(
    tr_score,
    tr_y,
    grid,
)

results["S6"] = report(
    "S6 Distilled",
    t,
    train_f1,
    te_y,
    te_score,
    (te_score >= t).astype(int),
)


# ============================================================
# MiniCheck helper
#
# minicheck_score = SUPPORT probability
# canonical threshold selection uses support threshold.
# ============================================================

def audit_minicheck(
    key,
    name,
    train_path,
    test_path,
):
    tr = load(train_path)
    te = load(test_path)

    tr_support = np.array(
        [
            float(r["minicheck_score"])
            for r in tr
            if r["minicheck_score"] is not None
        ]
    )

    tr_y = np.array(
        [
            int(r["ground_truth_hallucination"])
            for r in tr
            if r["minicheck_score"] is not None
        ]
    )

    valid_te = [
        r for r in te
        if r["minicheck_score"] is not None
    ]

    te_support = np.array(
        [
            float(r["minicheck_score"])
            for r in valid_te
        ]
    )

    te_y = np.array(
        [
            int(r["ground_truth_hallucination"])
            for r in valid_te
        ]
    )

    grid = np.arange(
        0.10, 0.901, 0.05
    )

    t, train_f1 = sweep_support(
        tr_support,
        tr_y,
        grid,
    )

    results[key] = report(
        name,
        t,
        train_f1,
        te_y,
        1.0 - te_support,
        (te_support < t).astype(int),
    )


audit_minicheck(
    "MC_roberta",
    "MiniCheck RoBERTa",
    "/workspace/minicheck_results_train_roberta.json",
    "/workspace/minicheck_results_test_roberta.json",
)

audit_minicheck(
    "MC_7B",
    "MiniCheck-7B",
    "/workspace/minicheck_results_train_7b.json",
    "/workspace/minicheck_results_test_7b.json",
)


# ============================================================
# Save
# ============================================================

out = {
    "protocol": (
        "Threshold chosen to maximize F1 on train-side "
        "scores only, then applied unchanged to RAGTruth test."
    ),
    "note_s3": (
        "S3 omitted because current consistency_results_*_v5 "
        "files contain only 10 debug examples."
    ),
    "results": results,
}

with open(
    "/workspace/table41_threshold_audit_results.json",
    "w",
) as f:
    json.dump(
        out,
        f,
        indent=2,
    )


print("\n" + "=" * 120)
print("OLD TABLE 4.1 VALUES FOR COMPARISON")
print("=" * 120)

print(
    "S1   old P=.3973 R=.9003 F1=.5513"
)
print(
    "S2   old P=.5489 R=.7381 F1=.6296"
)
print(
    "S3   old P=.3584 R=.9894 F1=.5262  [not audited yet]"
)
print(
    "S4   old P=.6748 R=.7349 F1=.7036"
)
print(
    "S5   old P=.4573 R=.8749 F1=.6007"
)
print(
    "S6   old P=.5973 R=.6967 F1=.6432"
)
print(
    "MC-R  old P=.3964 R=.8378 F1=.5381"
)
print(
    "MC7B  old P=.7147 R=.7572 F1=.7353"
)

print(
    "\nSaved: /workspace/table41_threshold_audit_results.json"
)
