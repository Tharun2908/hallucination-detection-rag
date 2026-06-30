# /workspace/relevance_v2_min_analysis.py
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

with open("/workspace/relevance_results_train_v2.json") as f:
    train_results = json.load(f)
with open("/workspace/relevance_results_test_v2.json") as f:
    test_results = json.load(f)

# Use min scores
train_raw = [r["raw_min_relevance"] for r in train_results if r["raw_min_relevance"] is not None]
test_raw  = [r["raw_min_relevance"] for r in test_results  if r["raw_min_relevance"] is not None]

# Normalize using train min/max
train_min, train_max = min(train_raw), max(train_raw)
print(f"Train min raw range: [{train_min:.3f}, {train_max:.3f}]")

def minmax(scores, mn, mx):
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(scores)
    return [float(max(0.0, min(1.0, (s - mn) / rng))) for s in scores]

train_norm = minmax(train_raw, train_min, train_max)
test_norm  = minmax(test_raw,  train_min, train_max)

train_labels = np.array([r["ground_truth_hallucination"] for r in train_results if r["raw_min_relevance"] is not None])
test_labels  = np.array([r["ground_truth_hallucination"] for r in test_results  if r["raw_min_relevance"] is not None])

train_scores = np.array(train_norm)
test_scores  = np.array(test_norm)

best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]:
    preds = (train_scores < t).astype(int)
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})")

preds = (test_scores < best_threshold).astype(int)
print(f"\n--- Signal 2 v2 (min aggregation) ---")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, 1 - test_scores):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(test_labels, preds))
print("(TN, FP)\n(FN, TP)")
