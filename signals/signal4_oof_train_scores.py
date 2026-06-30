"""
signal4_oof_train_scores.py

Generate out-of-fold (OOF) S4 scores for the RAGTruth train split.

Why:
- The final fusion model should not train on S4 scores produced in-sample.
- Each train example is scored by an S4 model that did NOT train on that example.
- These OOF scores are used only for fusion/cascade training.
- The normal full-train S4 checkpoint is still used for test scoring.

Outputs:
- /workspace/signal4_results_train_oof.json
- /workspace/signal4_oof_metrics.json
- /workspace/signal4_oof_models/fold_*/
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# -------------------------
# Config
# -------------------------

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

OUTPUT_PATH = "/workspace/signal4_results_train_oof.json"
METRICS_PATH = "/workspace/signal4_oof_metrics.json"
OOF_MODEL_DIR = "/workspace/signal4_oof_models"

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2

N_SPLITS = 5
INNER_VAL_SPLIT = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

Path(OOF_MODEL_DIR).mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}", flush=True)
print(f"Output path: {OUTPUT_PATH}", flush=True)


# -------------------------
# Data helpers
# -------------------------

def is_hallucinated(example) -> int:
    labels = example["hallucination_labels_processed"]
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)


def prepare_examples(dataset):
    examples = []
    for idx, ex in enumerate(dataset):
        examples.append(
            {
                "idx": idx,
                "answer": ex["output"],
                "context": ex["context"],
                "label": is_hallucinated(ex),
                "model": ex.get("model", "unknown"),
                "task_type": ex.get("task_type", "unknown"),
            }
        )
    return examples


class RAGTruthDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["answer"],
            ex["context"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
            "idx": torch.tensor(ex["idx"], dtype=torch.long),
        }


def make_loader(examples, tokenizer, shuffle=False):
    dataset = RAGTruthDataset(examples, tokenizer, MAX_LENGTH)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def compute_class_weights(examples):
    labels = [ex["label"] for ex in examples]
    n_neg = labels.count(0)
    n_pos = labels.count(1)
    total = len(labels)

    weight_neg = total / (2 * n_neg)
    weight_pos = total / (2 * n_pos)

    return torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(DEVICE)


def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_idxs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_idxs.extend(batch["idx"].numpy().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_idxs)


def metrics_at_threshold(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "auroc": round(roc_auc_score(labels, probs), 4),
    }


# -------------------------
# Load RAGTruth train split
# -------------------------

print("Loading RAGTruth train split...", flush=True)
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
all_examples = prepare_examples(dataset_train)
all_labels = np.array([ex["label"] for ex in all_examples])

print(f"Total train examples: {len(all_examples)}", flush=True)
print(
    f"Labels: neg={(all_labels == 0).sum()} | pos={(all_labels == 1).sum()}",
    flush=True,
)


# -------------------------
# OOF training
# -------------------------

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

all_oof_results = []
fold_metrics = []

for fold, (train_val_idx, heldout_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
    print("\n" + "=" * 70, flush=True)
    print(f"Fold {fold}/{N_SPLITS}", flush=True)
    print("=" * 70, flush=True)

    set_seed(SEED + fold)

    fold_dir = os.path.join(OOF_MODEL_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_val_examples = [all_examples[i] for i in train_val_idx]
    heldout_examples = [all_examples[i] for i in heldout_idx]

    train_val_labels = [ex["label"] for ex in train_val_examples]

    inner_train_examples, inner_val_examples = train_test_split(
        train_val_examples,
        test_size=INNER_VAL_SPLIT,
        random_state=SEED + fold,
        stratify=train_val_labels,
    )

    print(
        f"Inner train: {len(inner_train_examples)} | "
        f"Inner val: {len(inner_val_examples)} | "
        f"OOF heldout: {len(heldout_examples)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model = model.to(DEVICE)

    train_loader = make_loader(inner_train_examples, tokenizer, shuffle=True)
    val_loader = make_loader(inner_val_examples, tokenizer, shuffle=False)
    heldout_loader = make_loader(heldout_examples, tokenizer, shuffle=False)

    class_weights = compute_class_weights(inner_train_examples)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Fold {fold} | Epoch {epoch} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss={total_loss / step:.4f}",
                    flush=True,
                )

        avg_loss = total_loss / len(train_loader)

        val_labels, val_probs, _ = evaluate(model, val_loader)
        val_metrics = metrics_at_threshold(val_labels, val_probs, threshold=0.5)

        print(
            f"Fold {fold} | Epoch {epoch} | "
            f"Train Loss={avg_loss:.4f} | "
            f"Val F1={val_metrics['f1']} | "
            f"P={val_metrics['precision']} | "
            f"R={val_metrics['recall']} | "
            f"AUROC={val_metrics['auroc']}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_count = 0
            model.save_pretrained(fold_dir)
            tokenizer.save_pretrained(fold_dir)
            print(f"  -> Saved fold {fold} best model", flush=True)
        else:
            patience_count += 1
            print(f"  -> No improvement ({patience_count}/{PATIENCE})", flush=True)

        if patience_count >= PATIENCE:
            print(f"Early stopping fold {fold} at epoch {epoch}", flush=True)
            break

    print(f"Loading best fold {fold} model from epoch {best_epoch}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir).to(DEVICE)

    heldout_labels, heldout_probs, heldout_idxs = evaluate(model, heldout_loader)
    heldout_metrics = metrics_at_threshold(heldout_labels, heldout_probs, threshold=0.5)

    print(
        f"Fold {fold} OOF heldout | "
        f"F1={heldout_metrics['f1']} | "
        f"P={heldout_metrics['precision']} | "
        f"R={heldout_metrics['recall']} | "
        f"AUROC={heldout_metrics['auroc']}",
        flush=True,
    )

    fold_metrics.append(
        {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_inner_val_f1": round(float(best_val_f1), 4),
            "heldout_metrics_at_0_5": heldout_metrics,
            "n_inner_train": len(inner_train_examples),
            "n_inner_val": len(inner_val_examples),
            "n_heldout": len(heldout_examples),
        }
    )

    heldout_by_idx = {ex["idx"]: ex for ex in heldout_examples}

    for idx, prob, label in zip(heldout_idxs.tolist(), heldout_probs.tolist(), heldout_labels.tolist()):
        ex = heldout_by_idx[int(idx)]
        all_oof_results.append(
            {
                "idx": int(idx),
                "signal4_score": round(float(prob), 4),
                "ground_truth_hallucination": bool(label),
                "predicted_hallucination": bool(prob >= 0.5),
                "model": ex["model"],
                "task_type": ex["task_type"],
                "fold": fold,
                "score_type": "out_of_fold",
            }
        )

    # free memory before next fold
    del model
    torch.cuda.empty_cache()


# -------------------------
# Save combined OOF output
# -------------------------

all_oof_results = sorted(all_oof_results, key=lambda x: x["idx"])

unique_idxs = {r["idx"] for r in all_oof_results}
if len(all_oof_results) != len(all_examples):
    raise RuntimeError(
        f"Expected {len(all_examples)} OOF results, got {len(all_oof_results)}"
    )
if len(unique_idxs) != len(all_examples):
    raise RuntimeError(
        f"Expected {len(all_examples)} unique idxs, got {len(unique_idxs)}"
    )

oof_labels = np.array([int(r["ground_truth_hallucination"]) for r in all_oof_results])
oof_probs = np.array([float(r["signal4_score"]) for r in all_oof_results])
overall_oof_metrics = metrics_at_threshold(oof_labels, oof_probs, threshold=0.5)

with open(OUTPUT_PATH, "w") as f:
    json.dump(all_oof_results, f, indent=2)

metrics_output = {
    "method": "Signal 4 out-of-fold train scoring",
    "n_splits": N_SPLITS,
    "inner_val_split": INNER_VAL_SPLIT,
    "seed": SEED,
    "output_path": OUTPUT_PATH,
    "overall_oof_metrics_at_0_5": overall_oof_metrics,
    "fold_metrics": fold_metrics,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_output, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("OOF Signal 4 train scoring complete", flush=True)
print("=" * 70, flush=True)
print(f"Saved: {OUTPUT_PATH}", flush=True)
print(f"Saved: {METRICS_PATH}", flush=True)
print(f"Overall OOF metrics @0.5: {overall_oof_metrics}", flush=True)