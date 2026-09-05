import json
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

SPLIT = "/workspace/halubench_group_split.json"
S2S4 = "/workspace/halubench_final_s2s4_scores.json"
MC = "/workspace/halubench_per_example_scores.json"
OUT = "/workspace/halubench_groupfix_thresholdfix_results.json"

COEF_S2 = -1.359009650504923
COEF_S4 = 3.1555282346796876
INTERCEPT = -1.4168645142264538

FUSION_THRESHOLD = 0.45
S4_THRESHOLD = 0.55
MC_THRESHOLD = 0.80

ESCALATION_RATES = [0, 5, 10, 20, 30, 50, 75, 100]
MC_COST_RATIO = 10.77


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
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

    return float(total / len(labels))


def metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)

    return {
        "f1": float(
            f1_score(labels, preds, zero_division=0)
        ),
        "precision": float(
            precision_score(labels, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, zero_division=0)
        ),
        "auroc": float(
            roc_auc_score(labels, scores)
        ),
        "auprc": float(
            average_precision_score(labels, scores)
        ),
        "ece": ece(scores, labels),
    }


with open(SPLIT) as f:
    split = json.load(f)

with open(S2S4) as f:
    final_rows = json.load(f)

with open(MC) as f:
    mc_rows = json.load(f)


print("Full S2/S4 rows:", len(final_rows))
print("Full MiniCheck rows:", len(mc_rows))

assert len(final_rows) == 14000
assert len(mc_rows) == 14000

test_ids = [
    int(i)
    for i in split["test_filtered_indices"]
]

train_ids = [
    int(i)
    for i in split["train_filtered_indices"]
]

assert len(test_ids) == 8000
assert len(train_ids) == 6000
assert set(test_ids).isdisjoint(train_ids)


final_by_idx = {
    int(r["idx"]): r
    for r in final_rows
}

mc_by_idx = {
    int(r["idx"]): r
    for r in mc_rows
}

assert len(final_by_idx) == 14000
assert len(mc_by_idx) == 14000
assert set(final_by_idx) == set(mc_by_idx)


# Verify score-file alignment.
for idx in range(14000):
    a = final_by_idx[idx]
    b = mc_by_idx[idx]

    assert int(a["label"]) == int(b["label"])
    assert a["source"] == b["source"]


labels = np.array([
    int(final_by_idx[i]["label"])
    for i in test_ids
])

s2_support = np.array([
    float(final_by_idx[i]["s2_support"])
    for i in test_ids
])

s4 = np.array([
    float(final_by_idx[i]["s4_score"])
    for i in test_ids
])

mc = np.array([
    float(mc_by_idx[i]["mc_hall"])
    for i in test_ids
])


fusion_logits = (
    INTERCEPT
    + COEF_S2 * s2_support
    + COEF_S4 * s4
)

fusion = sigmoid(fusion_logits)


print("\n" + "=" * 85)
print("GROUP-DISJOINT FIXED-8K ENDPOINTS")
print("=" * 85)

print(
    "N:",
    len(labels),
    "positive rate:",
    f"{labels.mean():.6f}"
)


endpoint_results = {}

for name, scores, threshold in [
    ("S4_zero_shot", s4, S4_THRESHOLD),
    ("Fusion_S2S4", fusion, FUSION_THRESHOLD),
    ("MiniCheck7B", mc, MC_THRESHOLD),
]:
    m = metrics(
        labels,
        scores,
        threshold,
    )

    endpoint_results[name] = m

    print(
        f"{name:<18} "
        f"F1={m['f1']:.4f}  "
        f"AUROC={m['auroc']:.4f}  "
        f"AUPRC={m['auprc']:.4f}  "
        f"ECE={m['ece']:.4f}"
    )


# ------------------------------------------------------------
# Corrected cascade on exactly the same group-disjoint 8k
# ------------------------------------------------------------

uncertainty = np.abs(
    fusion - 0.5
)

ranked = np.argsort(
    uncertainty
)

cascade_results = []

print("\n" + "=" * 85)
print("GROUP-DISJOINT FIXED-8K CASCADE")
print("=" * 85)

for rate in ESCALATION_RATES:

    n = int(
        len(labels) * rate / 100
    )

    mask = np.zeros(
        len(labels),
        dtype=bool,
    )

    if n:
        mask[
            ranked[:n]
        ] = True

    scores = fusion.copy()

    scores[
        mask
    ] = mc[
        mask
    ]

    preds = (
        fusion >= FUSION_THRESHOLD
    ).astype(int)

    preds[
        mask
    ] = (
        mc[
            mask
        ] >= MC_THRESHOLD
    ).astype(int)

    m = {
        "f1": float(
            f1_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "ece": ece(
            scores,
            labels,
        ),
    }

    # Sequential cost:
    # lightweight verifier always runs,
    # MiniCheck additionally runs on fraction r.
    cost = (
        1.0
        + MC_COST_RATIO * rate / 100.0
    )

    row = {
        "escalation_rate": rate,
        "n_escalated": n,
        "cost": cost,
        **m,
    }

    cascade_results.append(
        row
    )

    print(
        f"{rate:3d}%  "
        f"F1={m['f1']:.4f}  "
        f"AUROC={m['auroc']:.4f}  "
        f"AUPRC={m['auprc']:.4f}  "
        f"ECE={m['ece']:.4f}  "
        f"Cost={cost:.2f}x"
    )


output = {
    "split": {
        "definition":
            "source_ds + normalized question + normalized passage",
        "train_n": 6000,
        "test_n": 8000,
        "group_overlap": 0,
        "paired_group_overlap": 0,
    },
    "positive_rate": float(labels.mean()),
    "endpoints": endpoint_results,
    "cascade": cascade_results,
}


with open(OUT, "w") as f:
    json.dump(
        output,
        f,
        indent=2,
    )


print("\nSaved:", OUT)
