import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

S2_MIN = -11.430
S2_MAX = 10.641

FUSION_THRESHOLD = 0.40
MC_HALL_THRESHOLD = 0.80

LATENCY_RATIO = 10.77

ESCALATION_RATES = [
    0, 5, 10, 20, 30, 50, 75, 100
]

OUT = Path(
    "/workspace/cascade_threshold_reaudit_results.json"
)


def load(path):
    with open(path) as f:
        return json.load(f)


def norm_s2(x):
    return float(
        max(
            0.0,
            min(
                1.0,
                (x - S2_MIN) / (S2_MAX - S2_MIN),
            ),
        )
    )


def compute_ece(scores, labels, n_bins=10):
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

        total += mask.sum() * abs(
            labels[mask].mean()
            - scores[mask].mean()
        )

    return float(total / len(labels))


def metrics(y, scores, preds):
    return {
        "f1": float(
            f1_score(
                y,
                preds,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y,
                preds,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                preds,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(
                y,
                scores,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                scores,
            )
        ),
        "ece": compute_ece(
            scores,
            y,
        ),
    }


# ============================================================
# Load canonical cached scores
# ============================================================

rel_train = {
    r["idx"]: r
    for r in load(
        "/workspace/relevance_results_train_v2.json"
    )
}

rel_test = {
    r["idx"]: r
    for r in load(
        "/workspace/relevance_results_test_v2.json"
    )
}

s4_train = {
    r["idx"]: r
    for r in load(
        "/workspace/signal4_results_train_oof.json"
    )
}

s4_test = {
    r["idx"]: r
    for r in load(
        "/workspace/signal4_results_test.json"
    )
}

mc_test = {
    r["idx"]: r
    for r in load(
        "/workspace/minicheck_results_test_7b.json"
    )
}


def extract(rel_map, s4_map):
    common = sorted(
        rel_map.keys() & s4_map.keys()
    )

    X = []
    y = []
    cats = []
    idxs = []

    for idx in common:
        r2 = rel_map[idx]
        r4 = s4_map[idx]

        if (
            r2["raw_min_relevance"] is None
            or r4["signal4_score"] is None
        ):
            continue

        X.append(
            [
                norm_s2(
                    float(
                        r2["raw_min_relevance"]
                    )
                ),
                float(
                    r4["signal4_score"]
                ),
            ]
        )

        y.append(
            int(
                r4[
                    "ground_truth_hallucination"
                ]
            )
        )

        cats.append(
            [
                r4["task_type"],
                r4["model"],
            ]
        )

        idxs.append(idx)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        cats,
        idxs,
    )


X_train, y_train, cats_train, _ = extract(
    rel_train,
    s4_train,
)

X_test, y_test, cats_test, test_idxs = extract(
    rel_test,
    s4_test,
)

assert len(y_train) == 15090
assert len(y_test) == 2700


# ============================================================
# Canonical final S2 + S4 + metadata fusion
# ============================================================

ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

ohe.fit(cats_train)

X_train_full = np.hstack(
    [
        X_train,
        ohe.transform(cats_train),
    ]
)

X_test_full = np.hstack(
    [
        X_test,
        ohe.transform(cats_test),
    ]
)

clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

clf.fit(
    X_train_full,
    y_train,
)

fusion_scores = clf.predict_proba(
    X_test_full
)[:, 1]

fusion_preds = (
    fusion_scores >= FUSION_THRESHOLD
).astype(int)


# ============================================================
# Align MiniCheck
# minicheck_score = support probability
# MC hallucination score = 1 - support
# ============================================================

mc_scores = []

for idx in test_idxs:
    r = mc_test[idx]

    if r["minicheck_score"] is None:
        raise RuntimeError(
            f"Missing MiniCheck score at idx={idx}"
        )

    mc_scores.append(
        1.0
        - float(r["minicheck_score"])
    )

mc_scores = np.asarray(
    mc_scores,
    dtype=float,
)

mc_preds = (
    mc_scores >= MC_HALL_THRESHOLD
).astype(int)


# ============================================================
# Endpoint verification
# ============================================================

fusion_endpoint = metrics(
    y_test,
    fusion_scores,
    fusion_preds,
)

mc_endpoint = metrics(
    y_test,
    mc_scores,
    mc_preds,
)

print("=" * 90)
print("ENDPOINT VERIFICATION")
print("=" * 90)

print(
    "Fusion      "
    f"F1={fusion_endpoint['f1']:.4f} "
    f"AUROC={fusion_endpoint['auroc']:.4f} "
    f"AUPRC={fusion_endpoint['auprc']:.4f} "
    f"ECE={fusion_endpoint['ece']:.4f}"
)

print(
    "MiniCheck   "
    f"F1={mc_endpoint['f1']:.4f} "
    f"AUROC={mc_endpoint['auroc']:.4f} "
    f"AUPRC={mc_endpoint['auprc']:.4f} "
    f"ECE={mc_endpoint['ece']:.4f}"
)

if abs(
    fusion_endpoint["f1"] - 0.7262
) > 0.0002:
    raise RuntimeError(
        "Fusion endpoint does not match canonical result."
    )

if abs(
    mc_endpoint["f1"] - 0.7260
) > 0.0002:
    raise RuntimeError(
        "MiniCheck endpoint does not match "
        "train-selected-threshold result."
    )


# ============================================================
# Cascade
# ============================================================

confidence = np.abs(
    fusion_scores - 0.5
)

results = []

print("\n" + "=" * 100)
print("CORRECTED RAGTRUTH CASCADE")
print("=" * 100)

print(
    f"{'Esc':>5} "
    f"{'F1':>8} "
    f"{'P':>8} "
    f"{'R':>8} "
    f"{'AUROC':>8} "
    f"{'AUPRC':>8} "
    f"{'ECE':>8} "
    f"{'Cost':>8}"
)

for esc_rate in ESCALATION_RATES:

    n_escalate = int(
        len(y_test)
        * esc_rate
        / 100
    )

    escalate_mask = np.zeros(
        len(y_test),
        dtype=bool,
    )

    if n_escalate > 0:
        escalate_idx = np.argsort(
            confidence
        )[:n_escalate]

        escalate_mask[
            escalate_idx
        ] = True

    # Continuous deployed score
    cascade_scores = fusion_scores.copy()

    cascade_scores[
        escalate_mask
    ] = mc_scores[
        escalate_mask
    ]

    # Binary decisions use each verifier's
    # independently train-selected threshold
    cascade_preds = fusion_preds.copy()

    cascade_preds[
        escalate_mask
    ] = mc_preds[
        escalate_mask
    ]

    m = metrics(
        y_test,
        cascade_scores,
        cascade_preds,
    )

    cost = (
        1.0
        + LATENCY_RATIO
        * (esc_rate / 100.0)
    )

    row = {
        "escalation_pct":
            esc_rate,
        "n_escalated":
            int(n_escalate),
        "relative_cost":
            float(cost),
        **m,
    }

    results.append(row)

    print(
        f"{esc_rate:>4}% "
        f"{m['f1']:>8.4f} "
        f"{m['precision']:>8.4f} "
        f"{m['recall']:>8.4f} "
        f"{m['auroc']:>8.4f} "
        f"{m['auprc']:>8.4f} "
        f"{m['ece']:>8.4f} "
        f"{cost:>7.2f}x"
    )


output = {
    "protocol": {
        "dataset":
            "RAGTruth test",
        "n":
            int(len(y_test)),
        "fusion":
            "canonical S2+S4+task+generator logistic regression",
        "fusion_threshold":
            FUSION_THRESHOLD,
        "minicheck_support_threshold":
            1.0 - MC_HALL_THRESHOLD,
        "minicheck_hallucination_threshold":
            MC_HALL_THRESHOLD,
        "threshold_selection":
            "all component thresholds selected from train-side scores; no test threshold tuning",
        "escalation_rule":
            "lowest absolute fusion confidence |p-0.5| first",
        "relative_cost_formula":
            "C(r) = 1 + 10.77*r",
    },
    "fusion_endpoint":
        fusion_endpoint,
    "minicheck_endpoint":
        mc_endpoint,
    "cascade":
        results,
}

with open(
    OUT,
    "w",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )

print(
    "\nSaved:",
    OUT,
)
