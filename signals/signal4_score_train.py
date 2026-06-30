import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/workspace/signal4_model"
BATCH_SIZE = 16
MAX_LENGTH = 512

print(f"Device: {DEVICE}")

print("Loading dataset...")
dataset_train = load_dataset("wandb/RAGTruth-processed", split="train")
print(f"Train: {len(dataset_train)}")

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return int(labels["evident_conflict"] > 0 or labels["baseless_info"] > 0)

def prepare_examples(dataset):
    examples = []
    for idx, ex in enumerate(dataset):
        examples.append({
            "idx":       idx,
            "answer":    ex["output"],
            "context":   ex["context"],
            "label":     is_hallucinated(ex),
            "model":     ex.get("model", "unknown"),
            "task_type": ex.get("task_type", "unknown"),
        })
    return examples

print("Preparing examples...")
train_examples = prepare_examples(dataset_train)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(DEVICE)
model.eval()

class RAGTruthDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples   = examples
        self.tokenizer  = tokenizer
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
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "idx":            ex["idx"],
        }

train_dataset = RAGTruthDataset(train_examples, tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Scoring train set...")
all_probs = []
all_idxs  = []

with torch.no_grad():
    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        probs          = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_idxs.extend(batch["idx"].tolist())
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{len(train_loader)}", flush=True)

results = []
for i, idx in enumerate(all_idxs):
    ex = train_examples[idx]
    results.append({
        "idx":                        idx,
        "signal4_score":              round(float(all_probs[i]), 4),
        "ground_truth_hallucination": bool(ex["label"]),
        "model":                      ex["model"],
        "task_type":                  ex["task_type"],
    })

with open("/workspace/signal4_results_train.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} examples to /workspace/signal4_results_train.json")
