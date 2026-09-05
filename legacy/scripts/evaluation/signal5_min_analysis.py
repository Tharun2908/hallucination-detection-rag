import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

with open('/workspace/signal5_results_train_mean.json') as f:
    train_results = json.load(f)
with open('/workspace/signal5_results_test_mean.json') as f:
    test_results = json.load(f)

# Use min_recall instead of mean_recall
train_scores = np.array([r["bertscore_min_recall"] for r in train_results if r["bertscore_min_recall"] is not None])
train_labels = np.array([r["ground_truth_hallucination"] for r in train_results if r["bertscore_min_recall"] is not None])

best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]:
    preds = (train_scores < t).astype(int)
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})")

test_scores = np.array([r["bertscore_min_recall"] for r in test_results if r["bertscore_min_recall"] is not None])
test_labels = np.array([r["ground_truth_hallucination"] for r in test_results if r["bertscore_min_recall"] is not None])

preds = (test_scores < best_threshold).astype(int)
print(f"\n--- Signal 5 (BERTScore min) ---")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, 1 - test_scores):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(test_labels, preds))
print("(TN, FP)\n(FN, TP)")
