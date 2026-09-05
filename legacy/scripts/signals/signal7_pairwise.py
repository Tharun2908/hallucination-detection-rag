"""
signal7_pairwise.py
Signal 7: Pairwise Ranking Finetuned DeBERTa

Training objective: margin ranking loss
- For each query, sample one faithful + one hallucinated answer
- Train model: score(faithful) - score(hallucinated) > margin
- Inference: single scalar score per example (higher = more faithful)

Base model: cross-encoder/nli-deberta-v3-base
Input: answer [SEP] context
Loss: MarginRankingLoss(margin=0.5)
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from collections import defaultdict

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Config ---
MODEL_NAME    = "cross-encoder/nli-deberta-v3-base"
OUTPUT_DIR    = "/workspace/signal7_model"
MAX_LENGTH    = 512
BATCH_SIZE    = 8
LEARNING_RATE = 2e-5
MAX_EPOCHS    = 5
PATIENCE      = 2
MARGIN        = 0.5
VAL_SPLIT     = 0.1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}", flush=True)

# --- Load dataset ---
print("Loading dataset...", flush=True)
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
dataset_test  = load_dataset("wandb/RAGTruth-processed", split="test")
print(f"Train: {len(dataset_train)} | Test: {len(dataset_test)}", flush=True)

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)

def prepare_examples(dataset):
    examples = []
    for idx, ex in enumerate(dataset):
        examples.append({
            "idx":       idx,
            "query":     ex["query"],
            "answer":    ex["output"],
            "context":   ex["context"],
            "label":     is_hallucinated(ex),
            "model":     ex.get("model", "unknown"),
            "task_type": ex.get("task_type", "unknown"),
        })
    return examples

print("Preparing examples...", flush=True)
all_train    = prepare_examples(dataset_train)
test_examples = prepare_examples(dataset_test)

# --- Stratified train/val split ---
all_labels = [ex["label"] for ex in all_train]
train_examples, val_examples = train_test_split(
    all_train, test_size=VAL_SPLIT, random_state=42, stratify=all_labels
)
print(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}", flush=True)

# --- Build pairs per query ---
def build_pairs(examples):
    """Group by query, create faithful-hallucinated pairs."""
    query_groups = defaultdict(lambda: {"faithful": [], "hallucinated": []})
    for ex in examples:
        key = ex["query"][:100] + '|||' + ex["context"][:100]
        if ex["label"] == 0:
            query_groups[key]["faithful"].append(ex)
        else:
            query_groups[key]["hallucinated"].append(ex)

    pairs = []
    for key, group in query_groups.items():
        if group["faithful"] and group["hallucinated"]:
            pairs.append((group["faithful"], group["hallucinated"]))
    return pairs

train_pairs = build_pairs(train_examples)
val_pairs   = build_pairs(val_examples)
print(f"Train pairs (queries): {len(train_pairs)} | Val pairs: {len(val_pairs)}", flush=True)

# --- Dataset class ---
class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length, resample=True):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.resample   = resample
        self._sample()

    def _sample(self):
        """Sample one faithful + one hallucinated per query."""
        self.sampled = []
        for faithful_list, hall_list in self.pairs:
            f = random.choice(faithful_list)
            h = random.choice(hall_list)
            self.sampled.append((f, h))

    def __len__(self):
        return len(self.sampled)

    def _encode(self, example):
        return self.tokenizer(
            example["answer"],
            example["context"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        faithful, hallucinated = self.sampled[idx]
        enc_f = self._encode(faithful)
        enc_h = self._encode(hallucinated)
        return {
            "input_ids_f":      enc_f["input_ids"].squeeze(0),
            "attention_mask_f": enc_f["attention_mask"].squeeze(0),
            "input_ids_h":      enc_h["input_ids"].squeeze(0),
            "attention_mask_h": enc_h["attention_mask"].squeeze(0),
        }

# --- Scoring model ---
class ScoringModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        self.scorer  = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls     = outputs.last_hidden_state[:, 0, :]
        score   = self.scorer(cls).squeeze(-1)
        return score

# --- Load tokenizer and model ---
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = ScoringModel(MODEL_NAME).to(DEVICE)

# --- Inference dataset (single examples) ---
class SingleDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples  = examples
        self.tokenizer = tokenizer
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
            "label":          ex["label"],
            "idx":            ex["idx"],
        }

# --- Validation function ---
def validate(pairs, tokenizer):
    """Compute ranking accuracy on pairs."""
    model.eval()
    correct, total = 0, 0
    all_scores, all_labels_list = [], []

    # Also score all val examples for AUROC
    val_all = [ex for faithful_list, hall_list in pairs
               for ex in faithful_list + hall_list]
    val_dataset = SingleDataset(val_all, tokenizer, MAX_LENGTH)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            scores         = model(input_ids, attention_mask).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels_list.extend(batch["label"].numpy().tolist())

    # Ranking accuracy
    idx_map = {ex["idx"]: score for ex, score in zip(val_all, all_scores)}
    for faithful_list, hall_list in pairs:
        for f in faithful_list:
            for h in hall_list:
                if f["idx"] in idx_map and h["idx"] in idx_map:
                    if idx_map[f["idx"]] > idx_map[h["idx"]]:
                        correct += 1
                    total += 1

    ranking_acc = correct / total if total > 0 else 0

    # AUROC (higher score = faithful = label 0, so invert)
    try:
        auroc = roc_auc_score(all_labels_list, [-s for s in all_scores])
    except:
        auroc = 0.0

    return ranking_acc, auroc

# --- Training setup ---
train_dataset = PairDataset(train_pairs, tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

ranking_loss = nn.MarginRankingLoss(margin=MARGIN)
optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps  = len(train_loader) * MAX_EPOCHS
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# --- Training loop ---
best_val_metric = 0
best_epoch      = 0
patience_count  = 0

print("\nStarting training...", flush=True)
for epoch in range(1, MAX_EPOCHS + 1):
    # Resample pairs each epoch
    train_dataset._sample()

    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids_f      = batch["input_ids_f"].to(DEVICE)
        attention_mask_f = batch["attention_mask_f"].to(DEVICE)
        input_ids_h      = batch["input_ids_h"].to(DEVICE)
        attention_mask_h = batch["attention_mask_h"].to(DEVICE)

        score_f = model(input_ids_f, attention_mask_f)
        score_h = model(input_ids_h, attention_mask_h)

        # target=1 means score_f should be higher than score_h
        target = torch.ones(score_f.size(0)).to(DEVICE)
        loss   = ranking_loss(score_f, score_h, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss={total_loss/(step+1):.4f}", flush=True)

    avg_loss = total_loss / len(train_loader)
    ranking_acc, val_auroc = validate(val_pairs, tokenizer)
    print(f"\nEpoch {epoch} | Train Loss={avg_loss:.4f} | Val Ranking Acc={ranking_acc:.4f} | Val AUROC={val_auroc:.4f}", flush=True)

    if ranking_acc > best_val_metric:
        best_val_metric = ranking_acc
        best_epoch      = epoch
        patience_count  = 0
        model.encoder.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        torch.save(model.scorer.state_dict(), f"{OUTPUT_DIR}/scorer.pt")
        print(f"  → Best model saved (val ranking acc={best_val_metric:.4f})", flush=True)
    else:
        patience_count += 1
        print(f"  → No improvement (patience {patience_count}/{PATIENCE})", flush=True)
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

print(f"\nBest epoch: {best_epoch} | Best val ranking acc: {best_val_metric:.4f}", flush=True)

# --- Load best model ---
print("\nLoading best model for inference...", flush=True)
model.encoder = AutoModel.from_pretrained(OUTPUT_DIR)
model.scorer.load_state_dict(torch.load(f"{OUTPUT_DIR}/scorer.pt"))
model = model.to(DEVICE)
model.eval()

# --- Score train set first for normalization ---
print("Scoring train set for threshold tuning...", flush=True)
train_single  = SingleDataset(train_examples, tokenizer, MAX_LENGTH)
train_loader2 = DataLoader(train_single, batch_size=BATCH_SIZE, shuffle=False)

tr_scores, tr_labels = [], []
with torch.no_grad():
    for batch in train_loader2:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scores         = model(input_ids, attention_mask).cpu().numpy()
        tr_scores.extend(scores.tolist())
        tr_labels.extend(batch["label"].numpy().tolist())

tr_arr  = np.array(tr_scores)
tr_norm = np.clip((tr_arr - tr_arr.min()) / (tr_arr.max() - tr_arr.min() + 1e-8), 0, 1)
tr_hall = 1 - tr_norm

# --- Score test set ---
print("Scoring test set...", flush=True)
test_dataset = SingleDataset(test_examples, tokenizer, MAX_LENGTH)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_scores, all_labels_list, all_idxs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scores         = model(input_ids, attention_mask).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels_list.extend(batch["label"].numpy().tolist())
        all_idxs.extend(batch["idx"].numpy().tolist())

# Normalize scores to [0,1] using min-max
scores_arr = np.array(all_scores)
# Use train min/max for normalization — no leakage
norm_scores = np.clip((scores_arr - tr_arr.min()) / (tr_arr.max() - tr_arr.min() + 1e-8), 0, 1)

# Higher raw score = more faithful = lower hallucination probability
# Invert for threshold tuning: higher = more likely hallucination
hall_scores = 1 - norm_scores

# Tune threshold on train
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (tr_hall >= t).astype(int)
    f1 = f1_score(tr_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})", flush=True)

# Save train results
train_results_out = []
for i, ex in enumerate(train_examples):
    train_results_out.append({
        "idx":                        ex["idx"],
        "signal7_score":              round(float(tr_norm[i]), 4),
        "ground_truth_hallucination": bool(ex["label"]),
        "model":                      ex["model"],
        "task_type":                  ex["task_type"],
    })
with open("/workspace/signal7_results_train.json", "w") as f:
    json.dump(train_results_out, f, indent=2)

# Evaluate on test
preds = (hall_scores >= best_threshold).astype(int)
print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 7 (Pairwise Ranking)")
print(f"{'='*50}")
print(f"F1:        {f1_score(all_labels_list, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(all_labels_list, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(all_labels_list, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(all_labels_list, hall_scores):.4f}")
print(f"Threshold: {best_threshold}")
cm = confusion_matrix(all_labels_list, preds)
print(f"Confusion Matrix:\n{cm}")
print("(TN, FP)\n(FN, TP)")

# Save test results
test_results_out = []
for i in range(len(all_idxs)):
    test_results_out.append({
        "idx":                        all_idxs[i],
        "signal7_score":              round(float(norm_scores[i]), 4),
        "ground_truth_hallucination": bool(all_labels_list[i]),
        "predicted_hallucination":    bool(preds[i]),
        "model":                      test_examples[all_idxs[i]]["model"],
        "task_type":                  test_examples[all_idxs[i]]["task_type"],
    })
with open("/workspace/signal7_results_test.json", "w") as f:
    json.dump(test_results_out, f, indent=2)

metrics = {
    "best_epoch":     best_epoch,
    "best_val_ranking_acc": best_val_metric,
    "best_threshold": best_threshold,
    "test_f1":        round(f1_score(all_labels_list, preds, zero_division=0), 4),
    "test_precision": round(precision_score(all_labels_list, preds, zero_division=0), 4),
    "test_recall":    round(recall_score(all_labels_list, preds, zero_division=0), 4),
    "test_auroc":     round(roc_auc_score(all_labels_list, hall_scores), 4),
    "confusion_matrix": cm.tolist()
}
with open("/workspace/signal7_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print("  /workspace/signal7_results_train.json")
print("  /workspace/signal7_results_test.json")
print("  /workspace/signal7_metrics.json")
print("  /workspace/signal7_model/")
