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
    confusion_matrix,
)


# ============================================================
# Load files
# ============================================================

with open("/workspace/relevance_results_train_v2.json") as f:
    rel_train = {ex["idx"]: ex for ex in json.load(f)}

with open("/workspace/signal4_results_train_oof.json") as f:
    s4_train = {ex["idx"]: ex for ex in json.load(f)}

with open("/workspace/signal5_v2_precision_results_train_mean.json") as f:
    s5_train = {ex["idx"]: ex for ex in json.load(f)}


with open("/workspace/relevance_results_test_v2.json") as f:
    rel_test = {ex["idx"]: ex for ex in json.load(f)}

with open("/workspace/signal4_results_test.json") as f:
    s4_test = {ex["idx"]: ex for ex in json.load(f)}

with open("/workspace/signal5_v2_precision_results_test_mean.json") as f:
    s5_test = {ex["idx"]: ex for ex in json.load(f)}


# ============================================================
# Constants
# ============================================================

S2_MIN, S2_MAX = -11.430, 10.641


def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))


def compute_ece(probs, labels, n_bins=10):
    """
    ECE for hallucination probability.
    probs: model probability for label 1 = hallucination
    labels: 0/1 hallucination labels
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])

        if mask.sum() == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)

    return float(ece)


# ============================================================
# Feature extraction
# ============================================================

def extract(rel_map, s4_map, s5_map):
    """
    Build both feature sets on the SAME common rows.

    Base:
        S2 + S4

    Plus:
        S2 + S4 + corrected S5

    Metadata:
        task_type + model
    """
    common = sorted(rel_map.keys() & s4_map.keys() & s5_map.keys())

    base_numeric = []
    plus_numeric = []
    categorical = []
    labels = []
    kept_indices = []

    for idx in common:
        r2 = rel_map[idx]
        r4 = s4_map[idx]
        r5 = s5_map[idx]

        needed = [
            r2.get("raw_min_relevance"),
            r4.get("signal4_score"),
            r5.get("signal5_score"),
        ]

        if any(x is None for x in needed):
            continue

        assert (
            r2["ground_truth_hallucination"]
            == r4["ground_truth_hallucination"]
            == r5["ground_truth_hallucination"]
        )

        s2_score = norm_s2(r2["raw_min_relevance"])
        s4_score = float(r4["signal4_score"])
        s5_score = float(r5["signal5_score"])

        labels.append(int(r2["ground_truth_hallucination"]))

        base_numeric.append([
            s2_score,
            s4_score,
        ])

        plus_numeric.append([
            s2_score,
            s4_score,
            s5_score,
        ])

        categorical.append([
            str(r2.get("task_type", "unknown")),
            str(r2.get("model", "unknown")),
        ])

        kept_indices.append(idx)

    return (
        np.array(base_numeric, dtype=float),
        np.array(plus_numeric, dtype=float),
        categorical,
        np.array(labels, dtype=int),
        kept_indices,
    )


train_base_num, train_plus_num, train_cat, y_train, train_idx = extract(
    rel_train,
    s4_train,
    s5_train,
)

test_base_num, test_plus_num, test_cat, y_test, test_idx = extract(
    rel_test,
    s4_test,
    s5_test,
)

print(f"Train rows: {len(y_train)}")
print(f"Test rows : {len(y_test)}")
print(f"Train hallucination rate: {y_train.mean():.4f}")
print(f"Test hallucination rate : {y_test.mean():.4f}")


# ============================================================
# Metadata one-hot encoding
# ============================================================

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(train_cat)

train_cat_ohe = ohe.transform(train_cat)
test_cat_ohe = ohe.transform(test_cat)

X_train_base = np.hstack([train_base_num, train_cat_ohe])
X_test_base = np.hstack([test_base_num, test_cat_ohe])

X_train_plus = np.hstack([train_plus_num, train_cat_ohe])
X_test_plus = np.hstack([test_plus_num, test_cat_ohe])

cat_feature_names = ohe.get_feature_names_out(["task_type", "model"]).tolist()

base_feature_names = [
    "Relevance score",
    "Signal4 score",
] + cat_feature_names

plus_feature_names = [
    "Relevance score",
    "Signal4 score",
    "Corrected BERTScore Precision",
] + cat_feature_names


# ============================================================
# Training and evaluation
# ============================================================

def tune_threshold_by_train_f1(y_true, y_prob):
    best_f1 = -1.0
    best_threshold = 0.5

    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def fit_and_eval(name, X_train, X_test, feature_names):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    best_threshold, best_train_f1 = tune_threshold_by_train_f1(
        y_train,
        train_prob,
    )

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    metrics = {
        "method": name,
        "best_threshold": float(best_threshold),
        "train_f1": round(float(best_train_f1), 4),
        "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
        "test_precision": round(float(precision_score(y_test, test_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, test_pred, zero_division=0)), 4),
        "test_auroc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "test_auprc": round(float(average_precision_score(y_test, test_prob)), 4),
        "test_ece": round(float(compute_ece(test_prob, y_test, n_bins=10)), 4),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
        "coefficients": {
            name: round(float(coef), 4)
            for name, coef in zip(feature_names, clf.coef_[0])
        },
    }

    return metrics


base_results = fit_and_eval(
    name="Logistic Regression S2+S4",
    X_train=X_train_base,
    X_test=X_test_base,
    feature_names=base_feature_names,
)

plus_results = fit_and_eval(
    name="Logistic Regression S2+S4+Corrected S5",
    X_train=X_train_plus,
    X_test=X_test_plus,
    feature_names=plus_feature_names,
)


# ============================================================
# Print comparison
# ============================================================

print("\n" + "=" * 110)
print("Fusion comparison: S2+S4 vs S2+S4+Corrected S5")
print("=" * 110)

print(
    f"{'Method':<45} "
    f"{'F1':>8} "
    f"{'Prec':>8} "
    f"{'Recall':>8} "
    f"{'AUROC':>8} "
    f"{'AUPRC':>8} "
    f"{'ECE':>8} "
    f"{'Thr':>8}"
)

print("-" * 110)

for r in [base_results, plus_results]:
    print(
        f"{r['method']:<45} "
        f"{r['test_f1']:>8.4f} "
        f"{r['test_precision']:>8.4f} "
        f"{r['test_recall']:>8.4f} "
        f"{r['test_auroc']:>8.4f} "
        f"{r['test_auprc']:>8.4f} "
        f"{r['test_ece']:>8.4f} "
        f"{r['best_threshold']:>8.2f}"
    )

print("-" * 110)

print(
    f"{'Delta plus - base':<45} "
    f"{plus_results['test_f1'] - base_results['test_f1']:>+8.4f} "
    f"{plus_results['test_precision'] - base_results['test_precision']:>+8.4f} "
    f"{plus_results['test_recall'] - base_results['test_recall']:>+8.4f} "
    f"{plus_results['test_auroc'] - base_results['test_auroc']:>+8.4f} "
    f"{plus_results['test_auprc'] - base_results['test_auprc']:>+8.4f} "
    f"{plus_results['test_ece'] - base_results['test_ece']:>+8.4f} "
    f"{'':>8}"
)

print("\nBase confusion matrix:", base_results["confusion_matrix"])
print("Plus confusion matrix:", plus_results["confusion_matrix"])

print("\nTop coefficients for S2+S4+Corrected S5:")
for name, coef in sorted(
    plus_results["coefficients"].items(),
    key=lambda x: abs(x[1]),
    reverse=True,
)[:20]:
    print(f"  {name:40s}: {coef:+.4f}")


# ============================================================
# Save
# ============================================================

output = {
    "comparison": "S2+S4 vs S2+S4+corrected_S5",
    "notes": {
        "s4_train": "/workspace/signal4_results_train_oof.json",
        "s4_test": "/workspace/signal4_results_test.json",
        "s5_train": "/workspace/signal5_v2_precision_results_train_mean.json",
        "s5_test": "/workspace/signal5_v2_precision_results_test_mean.json",
        "s5_direction": "high support, low hallucination",
        "same_rows_for_both_models": True,
    },
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "train_indices": train_idx,
    "test_indices": test_idx,
    "base_s2_s4": base_results,
    "plus_s2_s4_s5": plus_results,
    "delta_plus_minus_base": {
        "test_f1": round(float(plus_results["test_f1"] - base_results["test_f1"]), 4),
        "test_precision": round(float(plus_results["test_precision"] - base_results["test_precision"]), 4),
        "test_recall": round(float(plus_results["test_recall"] - base_results["test_recall"]), 4),
        "test_auroc": round(float(plus_results["test_auroc"] - base_results["test_auroc"]), 4),
        "test_auprc": round(float(plus_results["test_auprc"] - base_results["test_auprc"]), 4),
        "test_ece": round(float(plus_results["test_ece"] - base_results["test_ece"]), 4),
    },
}

with open("/workspace/fusion_s2s4_vs_s2s4s5_v2_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to /workspace/fusion_s2s4_vs_s2s4s5_v2_results.json")
