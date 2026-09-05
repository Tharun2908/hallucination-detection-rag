import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import os

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MODEL_DIR     = '/workspace/signal4_model'
OUTPUT_DIR    = '/workspace/signal4_halubench_fewshot'
MAX_LENGTH    = 512
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS    = 3
PATIENCE      = 1
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}", flush=True)

print("Loading HaluBench...", flush=True)
ds = load_dataset('PatronusAI/HaluBench', split='test')
ds = ds.filter(lambda x: x['source_ds'] != 'RAGTruth')

examples = [{'context': ex['passage'], 'answer': ex['answer'],
              'label': 1 if ex['label'] == 'FAIL' else 0,
              'source': ex['source_ds']} for ex in ds]

labels  = np.array([e['label'] for e in examples])
indices = np.arange(len(examples))

cal_idx, test_idx = train_test_split(indices, test_size=0.9, random_state=42, stratify=labels)
cal_labels = labels[cal_idx]
train_cal_idx, val_cal_idx = train_test_split(cal_idx, test_size=0.2, random_state=42, stratify=cal_labels)

train_examples = [examples[i] for i in train_cal_idx]
val_examples   = [examples[i] for i in val_cal_idx]
test_examples  = [examples[i] for i in test_idx]

print(f"Train: {len(train_examples)} | Val: {len(val_examples)} | Test: {len(test_examples)}", flush=True)

class HaluDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex['answer'], ex['context'],
                             max_length=self.max_length, truncation=True,
                             padding='max_length', return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(ex['label'], dtype=torch.long)}

print("Loading Signal 4 model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

train_loader = DataLoader(HaluDataset(train_examples, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(HaluDataset(val_examples,   tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(HaluDataset(test_examples,  tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

n_pos = sum(e['label'] for e in train_examples)
n_neg = len(train_examples) - n_pos
class_weights = torch.tensor([len(train_examples)/(2*n_neg), len(train_examples)/(2*n_pos)], dtype=torch.float).to(DEVICE)
print(f"Class weights: neg={class_weights[0]:.3f}, pos={class_weights[1]:.3f}", flush=True)

loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

def evaluate(loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch['input_ids'].to(DEVICE),
                           attention_mask=batch['attention_mask'].to(DEVICE)).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)

best_val_auroc, best_epoch, patience_count = 0, 0, 0

print("\nStarting few-shot finetuning...", flush=True)
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(input_ids=batch['input_ids'].to(DEVICE),
                       attention_mask=batch['attention_mask'].to(DEVICE)).logits
        loss = loss_fn(logits, batch['label'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (step+1) % 10 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss={total_loss/(step+1):.4f}", flush=True)

    val_probs, val_labels = evaluate(val_loader)
    val_auroc = roc_auc_score(val_labels, val_probs)
    print(f"Epoch {epoch} | Loss={total_loss/len(train_loader):.4f} | Val AUROC={val_auroc:.4f}", flush=True)

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        best_epoch = epoch
        patience_count = 0
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  -> Best model saved (val AUROC={best_val_auroc:.4f})", flush=True)
    else:
        patience_count += 1
        print(f"  -> No improvement (patience {patience_count}/{PATIENCE})", flush=True)
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

print(f"\nBest epoch: {best_epoch} | Val AUROC: {best_val_auroc:.4f}", flush=True)

print("\nLoading best model...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)

val_probs, val_labels = evaluate(val_loader)
best_f1, best_t = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(val_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print(f"Best threshold on val: {best_t:.2f} (F1={best_f1:.4f})", flush=True)

test_probs, test_labels = evaluate(test_loader)
preds = (test_probs >= best_t).astype(int)

print(f"\n{'='*60}")
print("FINAL RESULTS - Few-shot Adaptation on HaluBench")
print(f"{'='*60}")
print(f"Train size: {len(train_examples)} (10% of HaluBench)")
print(f"F1:        {f1_score(test_labels, preds, zero_division=0):.4f}")
print(f"Precision: {precision_score(test_labels, preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_labels, preds, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(test_labels, test_probs):.4f}")
print(f"AUPRC:     {average_precision_score(test_labels, test_probs):.4f}")

metrics = {
    'train_size': len(train_examples), 'val_size': len(val_examples),
    'test_size': len(test_examples), 'best_epoch': best_epoch,
    'best_val_auroc': best_val_auroc, 'best_threshold': best_t,
    'test_f1':        round(f1_score(test_labels, preds, zero_division=0), 4),
    'test_precision': round(precision_score(test_labels, preds, zero_division=0), 4),
    'test_recall':    round(recall_score(test_labels, preds, zero_division=0), 4),
    'test_auroc':     round(roc_auc_score(test_labels, test_probs), 4),
    'test_auprc':     round(average_precision_score(test_labels, test_probs), 4),
}
with open('/workspace/halubench_fewshot_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved to /workspace/halubench_fewshot_results.json")
