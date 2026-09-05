import json
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

S2_MIN, S2_MAX = -11.430, 10.641

def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

def compute_ece(probs, labels, n_bins=10):
    # Same ECE implementation used by evaluation/complete_metrics.py
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])

        if mask.sum() == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()

        ece += mask.sum() * abs(bin_acc - bin_conf)

    return round(float(ece / len(probs)), 4)

def best_threshold(scores, labels):
    best_f1 = 0.0
    best_t = 0.5

    for t in [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# ------------------------------------------------------------------
# Load cached scores
# ------------------------------------------------------------------

with open("/workspace/relevance_results_train_v2.json") as f:
    rel_train = {r["idx"]: r for r in json.load(f)}

with open("/workspace/signal4_results_train_oof.json") as f:
    s4_train = {r["idx"]: r for r in json.load(f)}

with open("/workspace/relevance_results_test_v2.json") as f:
    rel_test = {r["idx"]: r for r in json.load(f)}

with open("/workspace/signal4_results_test.json") as f:
    s4_test = {r["idx"]: r for r in json.load(f)}

with open("/workspace/minicheck_results_test_7b.json") as f:
    mc_test = {r["idx"]: r for r in json.load(f)}


# ------------------------------------------------------------------
# Reconstruct canonical OOF S2+S4 fusion
# ------------------------------------------------------------------

def extract(rel_map, s4_map):
    common = sorted(rel_map.keys() & s4_map.keys())

    X = []
    y = []
    cats = []
    idxs = []

    for idx in common:
        r2 = rel_map[idx]
        r4 = s4_map[idx]

        if r2["raw_min_relevance"] is None or r4["signal4_score"] is None:
            continue

        X.append([
            norm_s2(r2["raw_min_relevance"]),
            r4["signal4_score"],
        ])

        y.append(int(r2["ground_truth_hallucination"]))
        cats.append([r2["task_type"], r2["model"]])
        idxs.append(idx)

    return np.array(X), np.array(y), cats, idxs


X_train, y_train, cats_train, _ = extract(rel_train, s4_train)
X_test, y_test, cats_test, test_idxs = extract(rel_test, s4_test)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(cats_train)

X_train_full = np.hstack([
    X_train,
    ohe.transform(cats_train),
])

X_test_full = np.hstack([
    X_test,
    ohe.transform(cats_test),
])

clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

clf.fit(X_train_full, y_train)

train_prob = clf.predict_proba(X_train_full)[:, 1]
test_prob = clf.predict_proba(X_test_full)[:, 1]

fusion_threshold, fusion_train_f1 = best_threshold(
    train_prob,
    y_train,
)

print(
    f"Fusion threshold: {fusion_threshold:.2f} "
    f"(train F1={fusion_train_f1:.4f})"
)


# ------------------------------------------------------------------
# Align MiniCheck scores
# ------------------------------------------------------------------

mc_scores = np.array([
    1.0 - mc_test[idx]["minicheck_score"]
    if idx in mc_test and mc_test[idx]["minicheck_score"] is not None
    else np.nan
    for idx in test_idxs
])

valid_mask = ~np.isnan(mc_scores)

test_prob = test_prob[valid_mask]
y_test = y_test[valid_mask]
mc_scores = mc_scores[valid_mask]

print(f"Valid examples: {len(y_test)}")

# Canonical Chapter 4 MiniCheck threshold
MC_THRESHOLD = 0.85

print(f"MiniCheck threshold: {MC_THRESHOLD:.2f}")


# ------------------------------------------------------------------
# Cascade evaluation
# ------------------------------------------------------------------

confidence = np.abs(test_prob - 0.5)

escalation_rates = [
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
]

results = []

for esc_rate in escalation_rates:

    n_escalate = int(
        len(confidence) * esc_rate / 100
    )

    escalate_mask = np.zeros(
        len(confidence),
        dtype=bool,
    )

    if n_escalate > 0:
        escalate_idx = np.argsort(confidence)[:n_escalate]
        escalate_mask[escalate_idx] = True

    # --------------------------------------------------------------
    # Continuous deployed cascade score
    # --------------------------------------------------------------

    cascade_scores = test_prob.copy()

    cascade_scores[escalate_mask] = (
        mc_scores[escalate_mask]
    )

    # --------------------------------------------------------------
    # Binary decisions:
    # each component uses its own train-selected threshold
    # --------------------------------------------------------------

    preds = (
        test_prob >= fusion_threshold
    ).astype(int)

    preds[escalate_mask] = (
        mc_scores[escalate_mask] >= MC_THRESHOLD
    ).astype(int)

    cost = 1.0 + 10.0 * (esc_rate / 100.0)

    f1 = round(
        f1_score(y_test, preds, zero_division=0),
        4,
    )

    precision = round(
        precision_score(y_test, preds, zero_division=0),
        4,
    )

    recall = round(
        recall_score(y_test, preds, zero_division=0),
        4,
    )

    auroc = round(
        roc_auc_score(y_test, cascade_scores),
        4,
    )

    auprc = round(
        average_precision_score(
            y_test,
            cascade_scores,
        ),
        4,
    )

    ece = compute_ece(
        cascade_scores,
        y_test,
    )

    result = {
        "escalation_rate": esc_rate,
        "n_escalated": n_escalate,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "cost": round(cost, 1),
    }

    results.append(result)

    print(
        f"{esc_rate:3d}% | "
        f"F1={f1:.4f} | "
        f"AUROC={auroc:.4f} | "
        f"AUPRC={auprc:.4f} | "
        f"ECE={ece:.4f} | "
        f"Cost={cost:.1f}x"
    )


# ------------------------------------------------------------------
# Endpoint sanity checks
# ------------------------------------------------------------------

r0 = results[0]
r100 = results[-1]

print("\n" + "=" * 72)
print("ENDPOINT CHECKS")
print("=" * 72)

print("\nr=0 — should match Chapter 4 fusion:")
print(
    f"F1={r0['f1']:.4f}, "
    f"AUROC={r0['auroc']:.4f}, "
    f"AUPRC={r0['auprc']:.4f}, "
    f"ECE={r0['ece']:.4f}"
)

print("\nr=100 — should match Chapter 4 MiniCheck-7B:")
print(
    f"F1={r100['f1']:.4f}, "
    f"AUROC={r100['auroc']:.4f}, "
    f"AUPRC={r100['auprc']:.4f}, "
    f"ECE={r100['ece']:.4f}"
)


# ------------------------------------------------------------------
# Find best F1 operating point
# ------------------------------------------------------------------

best = max(
    results,
    key=lambda x: x["f1"],
)

print("\n" + "=" * 72)
print("BEST CASCADE F1")
print("=" * 72)

print(
    f"Escalation = {best['escalation_rate']}%\n"
    f"F1         = {best['f1']:.4f}\n"
    f"AUROC      = {best['auroc']:.4f}\n"
    f"AUPRC      = {best['auprc']:.4f}\n"
    f"ECE        = {best['ece']:.4f}\n"
    f"Cost       = {best['cost']:.1f}x"
)


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

OUT = "/workspace/cascade_ch6_results.json"

with open(OUT, "w") as f:
    json.dump(
        {
            "fusion_threshold": fusion_threshold,
            "minicheck_threshold": MC_THRESHOLD,
            "n_test": int(len(y_test)),
            "results": results,
            "best_f1_operating_point": best,
        },
        f,
        indent=2,
    )

print(f"\nSaved to {OUT}")
