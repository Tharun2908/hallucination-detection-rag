"""
signal8_distillation.py
Signal 8: Knowledge Distillation from MiniCheck-7B

Teacher: MiniCheck-7B (soft probability targets)
Student: DeBERTa finetuned with MSE loss on teacher soft labels
Input: answer [SEP] context
Loss: MSE(student_score, 1 - minicheck_score)

Key difference from Signal 4:
- Signal 4: binary cross-entropy on hard RAGTruth labels
- Signal 8: MSE on soft MiniCheck-7B probability outputs
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MODEL_NAME    = "cross-encoder/nli-deberta-v3-base"
OUTPUT_DIR    = "/workspace/signal8_model"
MAX_LENGTH    = 512
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS    = 5
PATIENCE      = 2
VAL_SPLIT     = 0.1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}", flush=True)

# --- Load teacher scores ---
print("Loading teacher scores...", flush=True)
with open('/workspace/minicheck_results_train_7b.json') as f:
    teacher_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_test_7b.json') as f:
    teacher_test = {r['idx']: r for r in json.load(f)}

print(f"Teacher train: {len(teacher_train)} | test: {len(teacher_test)}", flush=True)

# --- Load dataset ---
print("Loading dataset...", flush=True)
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
dataset_test  = load_dataset("wandb/RAGTruth-processed", split="test")

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)

def prepare_examples(dataset, teacher_map):
    examples = []
    for idx, ex in enumerate(dataset):
        if idx not in teacher_map:
            continue
        t = teacher_map[idx]
        if t['minicheck_score'] is None:
            continue
        examples.append({
            "idx":             idx,
            "answer":          ex["output"],
            "context":         ex["context"],
            "label":           is_hallucinated(ex),
            "soft_target":     float(1 - t['minicheck_score']),  # invert: high = hallucination
            "model":           ex.get("model", "unknown"),
            "task_type":       ex.get("task_type", "unknown"),
        })
    return examples

print("Preparing examples...", flush=True)
all_train     = prepare_examples(dataset_train, teacher_train)
test_examples = prepare_examples(dataset_test,  teacher_test)

# Stratified train/val split
all_labels = [ex["label"] for ex in all_train]
train_examples, val_examples = train_test_split(
    all_train, test_size=VAL_SPLIT, random_state=42, stratify=all_labels
)
print(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}", flush=True)

# --- Dataset class ---
class DistillDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["answer"], ex["context"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "soft_target":    torch.tensor(ex["soft_target"], dtype=torch.float),
            "label":          torch.tensor(ex["label"], dtype=torch.long),
            "idx":            ex["idx"],
        }

# --- Load model ---
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True
)
model = model.to(DEVICE)

# --- Dataloaders ---
train_dataset = DistillDataset(train_examples, tokenizer, MAX_LENGTH)
val_dataset   = DistillDataset(val_examples,   tokenizer, MAX_LENGTH)
test_dataset  = DistillDataset(test_examples,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# --- Loss: MSE on soft targets ---
mse_loss  = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def evaluate(loader):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits         = model(input_ids=input_ids, attention_mask=attention_mask).logits
            scores         = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
    return np.array(all_scores), np.array(all_labels)

def compute_f1_at_threshold(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    return f1_score(labels, preds, zero_division=0)

# --- Training ---
best_val_auroc = 0
best_epoch     = 0
patience_count = 0

print("\nStarting training...", flush=True)
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        soft_targets   = batch["soft_target"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        scores = torch.sigmoid(logits).squeeze(-1)
        loss   = mse_loss(scores, soft_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss={total_loss/(step+1):.4f}", flush=True)

    avg_loss = total_loss / len(train_loader)
    val_scores, val_labels = evaluate(val_loader)
    val_auroc = roc_auc_score(val_labels, val_scores)
    print(f"\nEpoch {epoch} | Train Loss={avg_loss:.4f} | Val AUROC={val_auroc:.4f}", flush=True)

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        best_epoch     = epoch
        patience_count = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  → Best model saved (val AUROC={best_val_auroc:.4f})", flush=True)
    else:
        patience_count += 1
        print(f"  → No improvement (patience {patience_count}/{PATIENCE})", flush=True)
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

print(f"\nBest epoch: {best_epoch} | Best val AUROC: {best_val_auroc:.4f}", flush=True)

# --- Load best model and evaluate ---
print("\nLoading best model...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=1, ignore_mismatched_sizes=True)
model = model.to(DEVICE)

# Tune threshold on train
train_scores, train_labels = evaluate(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False))
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (train_scores >= t).astype(int)
    f1 = f1_score(train_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})", flush=True)

# Evaluate on test
test_scores, test_labels = evaluate(test_loader)
preds = (test_scores >= best_threshold).astype(int)

print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 8 (Distillation from MiniCheck-7B)")
print(f"{'='*50}")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, test_scores):.4f}")
print(f"Threshold: {best_threshold}")
cm = confusion_matrix(test_labels, preds)
print(f"Confusion Matrix:\n{cm}")
print("(TN, FP)\n(FN, TP)")

# Save results
train_results = []
for i, ex in enumerate(train_examples):
    train_results.append({
        "idx":                        ex["idx"],
        "signal8_score":              round(float(train_scores[i]), 4),
        "ground_truth_hallucination": bool(ex["label"]),
        "model":                      ex["model"],
        "task_type":                  ex["task_type"],
    })
with open("/workspace/signal8_results_train.json", "w") as f:
    json.dump(train_results, f, indent=2)

test_results = []
for i in range(len(test_labels)):
    test_results.append({
        "idx":                        test_examples[i]["idx"],
        "signal8_score":              round(float(test_scores[i]), 4),
        "ground_truth_hallucination": bool(test_labels[i]),
        "predicted_hallucination":    bool(preds[i]),
        "model":                      test_examples[i]["model"],
        "task_type":                  test_examples[i]["task_type"],
    })
with open("/workspace/signal8_results_test.json", "w") as f:
    json.dump(test_results, f, indent=2)

metrics = {
    "best_epoch":      best_epoch,
    "best_val_auroc":  best_val_auroc,
    "best_threshold":  best_threshold,
    "test_f1":         round(f1_score(test_labels, preds, zero_division=0), 4),
    "test_precision":  round(precision_score(test_labels, preds, zero_division=0), 4),
    "test_recall":     round(recall_score(test_labels, preds, zero_division=0), 4),
    "test_auroc":      round(roc_auc_score(test_labels, test_scores), 4),
    "confusion_matrix": cm.tolist()
}
with open("/workspace/signal8_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print("  /workspace/signal8_results_train.json")
print("  /workspace/signal8_results_test.json")
print("  /workspace/signal8_metrics.json")
print("  /workspace/signal8_model/")
