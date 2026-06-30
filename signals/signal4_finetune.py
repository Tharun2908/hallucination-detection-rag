"""
signal4_finetune.py
Signal 4: Finetuned DeBERTa hallucination detector on RAGTruth

Input:  answer [SEP] context (truncate context to fit 512 tokens)
Model:  cross-encoder/nli-deberta-v3-base
Loss:   weighted cross-entropy
Early stopping: patience=1 on validation F1
Output: /workspace/signal4_model/
        /workspace/signal4_results_test.json
        /workspace/signal4_metrics.json
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Config ---
MODEL_NAME     = "cross-encoder/nli-deberta-v3-base"
OUTPUT_DIR     = "/workspace/signal4_model"
MAX_LENGTH     = 512
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
MAX_EPOCHS     = 5
PATIENCE       = 2
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT      = 0.1   # 10% of train for validation

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# --- Load dataset ---
print("Loading dataset...", flush=True)
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
dataset_test  = load_dataset("wandb/RAGTruth-processed", split="test")
print(f"Train: {len(dataset_train)} | Test: {len(dataset_test)}")

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)

# --- Dataset class ---
class RAGTruthDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex      = self.examples[idx]
        answer  = ex["answer"]
        context = ex["context"]
        label   = ex["label"]

        encoding = self.tokenizer(
            answer,
            context,
            max_length=self.max_length,
            truncation=True,   # keep full answer, truncate context
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "idx":            ex["idx"],
        }

# --- Prepare data ---
def prepare_examples(dataset):
    examples = []
    for idx, ex in enumerate(dataset):
        examples.append({
            "idx":      idx,
            "answer":   ex["output"],
            "context":  ex["context"],
            "label":    is_hallucinated(ex),
            "model":    ex.get("model", "unknown"),
            "task_type": ex.get("task_type", "unknown"),
        })
    return examples

print("Preparing examples...", flush=True)
all_train = prepare_examples(dataset_train)
test_examples = prepare_examples(dataset_test)

# --- Train/val split ---
from sklearn.model_selection import train_test_split
all_labels = [ex["label"] for ex in all_train]
train_examples, val_examples = train_test_split(all_train, test_size=VAL_SPLIT, random_state=42, stratify=all_labels)

print(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}")

# --- Compute class weights ---
train_labels = [ex["label"] for ex in train_examples]
n_neg = train_labels.count(0)
n_pos = train_labels.count(1)
total = len(train_labels)
weight_neg = total / (2 * n_neg)
weight_pos = total / (2 * n_pos)
class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(DEVICE)
print(f"Class weights: neg={weight_neg:.3f}, pos={weight_pos:.3f}")

# --- Load model and tokenizer ---
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
model     = model.to(DEVICE)

# --- Dataloaders ---
train_dataset = RAGTruthDataset(train_examples, tokenizer, MAX_LENGTH)
val_dataset   = RAGTruthDataset(val_examples,   tokenizer, MAX_LENGTH)
test_dataset  = RAGTruthDataset(test_examples,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# --- Optimizer and scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# --- Evaluation function ---
def evaluate(loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].numpy()
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "auroc":     round(roc_auc_score(labels, probs), 4),
    }

# --- Training loop with early stopping ---
best_val_f1   = 0
best_epoch    = 0
patience_count = 0

print("\nStarting training...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss={total_loss/(step+1):.4f}", flush=True)

    avg_train_loss = total_loss / len(train_loader)
    # Validation
    val_labels, val_probs = evaluate(val_loader)
    val_metrics = compute_metrics(val_labels, val_probs)
    print(f"\nEpoch {epoch} | Train Loss={avg_train_loss:.4f} | Val F1={val_metrics['f1']} | Precision={val_metrics['precision']} | Recall={val_metrics['recall']} | AUROC={val_metrics['auroc']}")

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch  = epoch
        patience_count = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  → Best model saved (val F1={best_val_f1})")
    else:
        patience_count += 1
        print(f"  → No improvement (patience {patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nBest epoch: {best_epoch} | Best val F1: {best_val_f1}")

# --- Load best model and evaluate on test ---
print("\nLoading best model for test evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
model = model.to(DEVICE)

# Tune threshold on validation set
val_labels, val_probs = evaluate(val_loader)
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(val_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on val: {best_threshold:.2f} (F1={best_f1:.4f})")

# Test evaluation
test_labels, test_probs = evaluate(test_loader)
test_metrics = compute_metrics(test_labels, test_probs, threshold=best_threshold)

print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 4 (Finetuned DeBERTa)")
print(f"{'='*50}")
print(f"F1:        {test_metrics['f1']}")
print(f"Precision: {test_metrics['precision']}")
print(f"Recall:    {test_metrics['recall']}")
print(f"AUROC:     {test_metrics['auroc']}")
print(f"Threshold: {best_threshold}")
cm = confusion_matrix(test_labels, (test_probs >= best_threshold).astype(int))
print(f"Confusion Matrix:\n{cm}")
print("(TN, FP)\n(FN, TP)")

# --- Save results ---
# Rebuild test examples with idx preserved
test_idxs = [ex["idx"] for ex in test_examples]
test_gt   = [ex["label"] for ex in test_examples]
test_model_names = [ex["model"] for ex in test_examples]
test_task_types  = [ex["task_type"] for ex in test_examples]

test_results = []
for i in range(len(test_idxs)):
    test_results.append({
        "idx":                        test_idxs[i],
        "signal4_score":              round(float(test_probs[i]), 4),
        "ground_truth_hallucination": bool(test_gt[i]),
        "predicted_hallucination":    bool(test_probs[i] >= best_threshold),
        "model":                      test_model_names[i],
        "task_type":                  test_task_types[i],
    })

with open("/workspace/signal4_results_test.json", "w") as f:
    json.dump(test_results, f, indent=2)

metrics_output = {
    "best_epoch":     best_epoch,
    "best_val_f1":    best_val_f1,
    "best_threshold": best_threshold,
    "test_metrics":   test_metrics,
    "confusion_matrix": cm.tolist(),
}
with open("/workspace/signal4_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

print("\nSaved:")
print("  /workspace/signal4_results_test.json")
print("  /workspace/signal4_metrics.json")
print("  /workspace/signal4_model/")
