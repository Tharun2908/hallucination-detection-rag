import json
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

with open("/workspace/nli_results_train.json") as f:
    results_train = json.load(f)
with open("/workspace/nli_results_test.json") as f:
    results_test = json.load(f)

dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
dataset_test = load_dataset("wandb/RAGTruth-processed", split="test")

def filter_baseless(results, dataset):
    filtered = []
    for r in results:
        if r["nli_score"] is None:
            continue
        idx = r["idx"]
        example = dataset[idx]
        labels = example["hallucination_labels_processed"]
        is_conflict = labels["evident_conflict"] > 0
        is_baseless = labels["baseless_info"] > 0

        # keep faithful + baseless-only (drop conflict)
        if is_conflict:
            continue

        filtered.append({
            "nli_score": r["nli_score"],
            "ground_truth_hallucination": is_baseless
        })
    return filtered

filtered_train = filter_baseless(results_train, dataset_train)
filtered_test  = filter_baseless(results_test, dataset_test)

print(f"Filtered train size: {len(filtered_train)}")
print(f"Filtered test size:  {len(filtered_test)}")

train_scores = np.array([x["nli_score"] for x in filtered_train])
train_labels = np.array([x["ground_truth_hallucination"] for x in filtered_train])

best_f1, best_threshold = 0, 0.9
for t in [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]:
    preds = (train_scores < t).astype(int)
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold on filtered train: {best_threshold:.2f} (F1={best_f1:.4f})")

test_scores = np.array([x["nli_score"] for x in filtered_test])
test_labels = np.array([x["ground_truth_hallucination"] for x in filtered_test])

preds = (test_scores < best_threshold).astype(int)

print(f"\n--- Baseless-Only Results (threshold={best_threshold:.2f}) ---")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, 1 - test_scores):.4f}")
print(f"Positives (baseless): {test_labels.sum()} / {len(test_labels)}")

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, preds)
print(cm)
print("(TN, FP)")
print("(FN, TP)")
