import json
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Load datasets FIRST
print("Loading datasets...", flush=True)
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
dataset_test  = load_dataset("wandb/RAGTruth-processed", split="test")
print("Datasets loaded.", flush=True)

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
TOP_K = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = model.config.id2label

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

def check_nli(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {id2label[i].lower(): float(probs[i].item()) for i in range(len(probs))}

def compute_contradiction_score(context, answer, top_k=TOP_K):
    sentences = split_into_sentences(context)
    if not sentences:
        return 0.0
    contradiction_scores = []
    for sentence in sentences:
        result = check_nli(sentence, answer)
        contradiction_scores.append(result.get("contradiction", 0.0))
    top_scores = [max(contradiction_scores)]
    return float(np.mean(top_scores))

# --- Load dataset ---

def filter_conflict(dataset):
    filtered = []
    for idx, example in enumerate(dataset):
        labels = example["hallucination_labels_processed"]
        is_conflict = labels["evident_conflict"] > 0
        is_baseless = labels["baseless_info"] > 0
        if is_baseless and not is_conflict:
            continue
        filtered.append({
            "idx": idx,
            "context": example["context"],
            "answer": example["output"],
            "ground_truth_hallucination": is_conflict
        })
    return filtered

print("Filtering datasets...", flush=True)
train_filtered = filter_conflict(dataset_train)
test_filtered  = filter_conflict(dataset_test)
print(f"Train: {len(train_filtered)} | Test: {len(test_filtered)}", flush=True)

def score_all(filtered, split_name):
    scores = []
    for i, ex in enumerate(filtered):
        s = compute_contradiction_score(ex["context"], ex["answer"])
        scores.append(s)
        if i % 100 == 0:
            print(f"{split_name}: {i}/{len(filtered)}", flush=True)
    return np.array(scores)

print("\nScoring train...", flush=True)
train_scores = score_all(train_filtered, "train")
train_labels = np.array([x["ground_truth_hallucination"] for x in train_filtered])

# Tune threshold on train
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.91, 0.05)]:
    preds = (train_scores >= t).astype(int)  # high contradiction = hallucination
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})", flush=True)

print("\nScoring test...", flush=True)
test_scores = score_all(test_filtered, "test")
test_labels = np.array([x["ground_truth_hallucination"] for x in test_filtered])

preds = (test_scores >= best_threshold).astype(int)

print(f"\n--- Contradiction Score: Conflict-Only (threshold={best_threshold:.2f}) ---", flush=True)
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}", flush=True)
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}", flush=True)
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}", flush=True)
print(f"AUROC:     {roc_auc_score(test_labels, test_scores):.4f}", flush=True)
print(f"Positives: {test_labels.sum()} / {len(test_labels)}", flush=True)

print("\nConfusion Matrix:", flush=True)
cm = confusion_matrix(test_labels, preds)
print(cm, flush=True)
print("(TN, FP)", flush=True)
print("(FN, TP)", flush=True)
