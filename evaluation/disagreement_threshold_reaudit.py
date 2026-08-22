import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score


# ============================================================
# Configuration
# ============================================================

S2_MIN = -11.430
S2_MAX = 10.641

FUSION_THRESHOLD = 0.40
MC_HALL_THRESHOLD = 0.80

OUT_JSON = Path(
    "/workspace/disagreement_threshold_reaudit_results.json"
)

OUT_SUMMARY = Path(
    "/workspace/disagreement_threshold_reaudit_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

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


# ============================================================
# Load cached scores
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


# ============================================================
# Build canonical fusion features
# ============================================================

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
# Canonical final fusion
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
# MiniCheck alignment
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

fusion_f1 = f1_score(
    y_test,
    fusion_preds,
    zero_division=0,
)

fusion_auc = roc_auc_score(
    y_test,
    fusion_scores,
)

mc_f1 = f1_score(
    y_test,
    mc_preds,
    zero_division=0,
)

mc_auc = roc_auc_score(
    y_test,
    mc_scores,
)

print("=" * 90)
print("ENDPOINT VERIFICATION")
print("=" * 90)

print(
    f"Fusion     F1={fusion_f1:.4f} "
    f"AUROC={fusion_auc:.4f}"
)

print(
    f"MiniCheck  F1={mc_f1:.4f} "
    f"AUROC={mc_auc:.4f}"
)

assert abs(
    fusion_f1 - 0.7262
) < 0.0002

assert abs(
    fusion_auc - 0.8749
) < 0.0002

assert abs(
    mc_f1 - 0.7260
) < 0.0002

assert abs(
    mc_auc - 0.8754
) < 0.0002


# ============================================================
# Error-overlap buckets
# ============================================================

bucket_names = [
    "both_correct",
    "both_wrong",
    "lightweight_wins",
    "minicheck_wins",
]

buckets = {
    name: []
    for name in bucket_names
}

per_example = []

for i, idx in enumerate(test_idxs):
    y = int(y_test[i])

    fp = int(
        fusion_preds[i]
    )

    mp = int(
        mc_preds[i]
    )

    fusion_correct = (
        fp == y
    )

    mc_correct = (
        mp == y
    )

    if (
        fusion_correct
        and mc_correct
    ):
        bucket = "both_correct"

    elif (
        not fusion_correct
        and not mc_correct
    ):
        bucket = "both_wrong"

    elif (
        fusion_correct
        and not mc_correct
    ):
        bucket = "lightweight_wins"

    else:
        bucket = "minicheck_wins"

    r4 = s4_test[idx]

    record = {
        "idx": int(idx),
        "ground_truth": y,
        "fusion_score":
            float(fusion_scores[i]),
        "fusion_pred":
            fp,
        "minicheck_hallucination_score":
            float(mc_scores[i]),
        "minicheck_pred":
            mp,
        "task_type":
            r4["task_type"],
        "model":
            r4["model"],
        "bucket":
            bucket,
    }

    per_example.append(record)
    buckets[bucket].append(record)


# ============================================================
# Breakdown helpers
# ============================================================

def breakdown(records, key):
    counter = Counter(
        r[key]
        for r in records
    )

    return {
        str(k): {
            "n": int(v),
            "pct_within_bucket":
                float(v / len(records))
                if records
                else 0.0,
        }
        for k, v in sorted(
            counter.items()
        )
    }


# ============================================================
# Results
# ============================================================

n = len(per_example)

bucket_summary = {}

print("\n" + "=" * 90)
print("CORRECTED DISAGREEMENT / ERROR OVERLAP")
print("=" * 90)

for name in bucket_names:
    records = buckets[name]

    bucket_summary[name] = {
        "n":
            len(records),
        "pct":
            len(records) / n,
        "positive_rate":
            float(
                np.mean(
                    [
                        r["ground_truth"]
                        for r in records
                    ]
                )
            )
            if records
            else None,
        "by_task_type":
            breakdown(
                records,
                "task_type",
            ),
        "by_model":
            breakdown(
                records,
                "model",
            ),
    }

    print(
        f"{name:<20} "
        f"{len(records):>4} "
        f"({len(records)/n:>6.2%})"
    )


# ============================================================
# Additional complementarity summaries
# ============================================================

fusion_correct_n = (
    len(buckets["both_correct"])
    + len(buckets["lightweight_wins"])
)

mc_correct_n = (
    len(buckets["both_correct"])
    + len(buckets["minicheck_wins"])
)

either_correct_n = (
    len(buckets["both_correct"])
    + len(buckets["lightweight_wins"])
    + len(buckets["minicheck_wins"])
)

oracle_accuracy = (
    either_correct_n / n
)

print("\n" + "=" * 90)
print("COMPLEMENTARITY")
print("=" * 90)

print(
    f"Fusion correct:       "
    f"{fusion_correct_n}/{n} "
    f"({fusion_correct_n/n:.2%})"
)

print(
    f"MiniCheck correct:    "
    f"{mc_correct_n}/{n} "
    f"({mc_correct_n/n:.2%})"
)

print(
    f"At least one correct: "
    f"{either_correct_n}/{n} "
    f"({oracle_accuracy:.2%})"
)

print(
    f"Both wrong:           "
    f"{len(buckets['both_wrong'])}/{n} "
    f"({len(buckets['both_wrong'])/n:.2%})"
)


# ============================================================
# Save JSON
# ============================================================

output = {
    "protocol": {
        "dataset":
            "RAGTruth test",
        "n":
            int(n),
        "fusion":
            "canonical S2+S4+task+generator logistic regression",
        "fusion_threshold":
            FUSION_THRESHOLD,
        "minicheck_support_threshold":
            1.0 - MC_HALL_THRESHOLD,
        "minicheck_hallucination_threshold":
            MC_HALL_THRESHOLD,
        "threshold_selection":
            "component thresholds selected from train-side scores only",
    },

    "endpoint_verification": {
        "fusion_f1":
            float(fusion_f1),
        "fusion_auroc":
            float(fusion_auc),
        "minicheck_f1":
            float(mc_f1),
        "minicheck_auroc":
            float(mc_auc),
    },

    "buckets":
        bucket_summary,

    "complementarity": {
        "fusion_correct_n":
            int(fusion_correct_n),
        "minicheck_correct_n":
            int(mc_correct_n),
        "either_correct_n":
            int(either_correct_n),
        "oracle_accuracy":
            float(oracle_accuracy),
    },

    "per_example":
        per_example,
}

with open(
    OUT_JSON,
    "w",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# Save readable summary
# ============================================================

lines = []

lines.append(
    "Corrected RAGTruth disagreement analysis"
)

lines.append(
    "=" * 55
)

lines.append(
    f"Fusion threshold: {FUSION_THRESHOLD}"
)

lines.append(
    f"MiniCheck hallucination threshold: {MC_HALL_THRESHOLD}"
)

lines.append("")

for name in bucket_names:
    b = bucket_summary[name]

    lines.append(
        f"{name}: "
        f"{b['n']} "
        f"({b['pct']:.2%})"
    )

lines.append("")

lines.append(
    f"Fusion correct: "
    f"{fusion_correct_n}/{n} "
    f"({fusion_correct_n/n:.2%})"
)

lines.append(
    f"MiniCheck correct: "
    f"{mc_correct_n}/{n} "
    f"({mc_correct_n/n:.2%})"
)

lines.append(
    f"At least one correct: "
    f"{either_correct_n}/{n} "
    f"({oracle_accuracy:.2%})"
)

lines.append(
    f"Both wrong: "
    f"{len(buckets['both_wrong'])}/{n} "
    f"({len(buckets['both_wrong'])/n:.2%})"
)

OUT_SUMMARY.write_text(
    "\n".join(lines)
)


print(
    "\nSaved:",
    OUT_JSON,
)

print(
    "Saved:",
    OUT_SUMMARY,
)
