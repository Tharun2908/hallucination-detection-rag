import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Coverage threshold — fraction of answer sentences that must be supported
SUPPORT_THRESHOLD = 0.5

def compute_coverage(sentence_scores, support_threshold):
    """
    Coverage = fraction of answer sentences with entailment > support_threshold
    Low coverage = many unsupported sentences = hallucination
    """
    if not sentence_scores:
        return 0.0
    supported = sum(1 for s in sentence_scores if s >= support_threshold)
    return round(supported / len(sentence_scores), 4)

with open('/workspace/nli_results_train_v2.json') as f:
    train_results = json.load(f)
with open('/workspace/nli_results_test_v2.json') as f:
    test_results = json.load(f)

def add_coverage(results, support_threshold):
    out = []
    for r in results:
        if r['nli_sentence_scores'] is None:
            continue
        out.append({
            "idx":                        r["idx"],
            "signal6_score":              compute_coverage(r["nli_sentence_scores"], support_threshold),
            "ground_truth_hallucination": r["ground_truth_hallucination"],
            "model":                      r["model"],
            "task_type":                  r["task_type"],
        })
    return out

train_cov = add_coverage(train_results, SUPPORT_THRESHOLD)
test_cov  = add_coverage(test_results,  SUPPORT_THRESHOLD)

print(f"Train: {len(train_cov)} | Test: {len(test_cov)}")

# Tune threshold on train
train_scores = np.array([r["signal6_score"] for r in train_cov])
train_labels = np.array([r["ground_truth_hallucination"] for r in train_cov])

best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (train_scores < t).astype(int)
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})")

test_scores = np.array([r["signal6_score"] for r in test_cov])
test_labels = np.array([r["ground_truth_hallucination"] for r in test_cov])

preds = (test_scores < best_threshold).astype(int)
print(f"\n--- Signal 6 (Retrieval Coverage) ---")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, 1 - test_scores):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(test_labels, preds))
print("(TN, FP)\n(FN, TP)")

# Save
with open('/workspace/signal6_results_train.json', 'w') as f:
    json.dump(train_cov, f, indent=2)
with open('/workspace/signal6_results_test.json', 'w') as f:
    json.dump(test_cov, f, indent=2)

metrics = {
    "support_threshold": SUPPORT_THRESHOLD,
    "best_threshold": best_threshold,
    "test_f1":        round(f1_score(test_labels, preds, zero_division=0), 4),
    "test_precision": round(precision_score(test_labels, preds, zero_division=0), 4),
    "test_recall":    round(recall_score(test_labels, preds, zero_division=0), 4),
    "test_auroc":     round(roc_auc_score(test_labels, 1 - test_scores), 4),
    "confusion_matrix": confusion_matrix(test_labels, preds).tolist()
}
with open('/workspace/signal6_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nSaved signal6_results_train/test.json and signal6_metrics.json")
