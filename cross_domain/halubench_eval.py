"""
halubench_eval.py
Out-of-domain evaluation on HaluBench

Signals evaluated:
- S2 (Relevance-approx) — zero-shot, no retraining
- Signal 4 (Finetuned DeBERTa) — trained on RAGTruth, tested on HaluBench
- Fusion S2+S4 — simple average (no metadata, since generator/task unknown)

RAGTruth examples excluded to avoid overlap.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from collections import defaultdict

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 512
S2_MIN, S2_MAX = -11.430, 10.641

print(f"Device: {DEVICE}", flush=True)
print("Note: S2 uses full answer-context scoring (approx of sentence-level). Thresholds transferred from RAGTruth.", flush=True)

# --- Load HaluBench (exclude RAGTruth) ---
print("Loading HaluBench...", flush=True)
ds = load_dataset('PatronusAI/HaluBench', split='test')
ds = ds.filter(lambda x: x['source_ds'] != 'RAGTruth')
print(f"Examples: {len(ds)}", flush=True)

examples = []
for ex in ds:
    examples.append({
        'context':  ex['passage'],
        'answer':   ex['answer'],
        'label':    1 if ex['label'] == 'FAIL' else 0,  # 1 = hallucination
        'source':   ex['source_ds'],
    })

labels = np.array([e['label'] for e in examples])
print(f"Positive rate (hallucination): {labels.mean():.3f}", flush=True)

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return round(float(ece / len(probs)), 4)

def evaluate(scores, labels, threshold=None):
    if threshold is None:
        best_f1, threshold = 0, 0.5
        for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                threshold = t
    preds = (scores >= threshold).astype(int)
    return {
        'threshold': threshold,
        'f1':        round(f1_score(labels, preds, zero_division=0), 4),
        'precision': round(precision_score(labels, preds, zero_division=0), 4),
        'recall':    round(recall_score(labels, preds, zero_division=0), 4),
        'auroc':     round(roc_auc_score(labels, scores), 4),
        'auprc':     round(average_precision_score(labels, scores), 4),
        'ece':       compute_ece(scores, labels),
        'n':         len(labels),
        'pos_rate':  round(float(labels.mean()), 3),
    }

# --- Signal 2: Relevance scoring ---
print("\nScoring S2 (Relevance-approx)...", flush=True)
rel_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

s2_raw = []
for i, ex in enumerate(examples):
    score = float(rel_model.predict([(ex['answer'], ex['context'])])[0])
    s2_raw.append(score)
    if (i + 1) % 500 == 0:
        print(f"  S2: {i+1}/{len(examples)}", flush=True)

s2_raw = np.array(s2_raw)
# Normalize using RAGTruth train stats
s2_norm = np.clip((s2_raw - S2_MIN) / (S2_MAX - S2_MIN + 1e-8), 0, 1)
s2_hall = 1 - s2_norm  # higher = more likely hallucination
print(f"S2 raw range: [{s2_raw.min():.3f}, {s2_raw.max():.3f}]", flush=True)

# --- Signal 4: Finetuned DeBERTa ---
print("\nScoring Signal 4 (Finetuned DeBERTa)...", flush=True)

class HaluDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex['answer'], ex['context'],
            max_length=self.max_length,
            truncation=True, padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          ex['label'],
        }

tokenizer = AutoTokenizer.from_pretrained('/workspace/signal4_model')
s4_model  = AutoModelForSequenceClassification.from_pretrained('/workspace/signal4_model')
s4_model  = s4_model.to(DEVICE)
s4_model.eval()

halu_dataset = HaluDataset(examples, tokenizer, MAX_LENGTH)
halu_loader  = DataLoader(halu_dataset, batch_size=BATCH_SIZE, shuffle=False)

s4_scores = []
with torch.no_grad():
    for i, batch in enumerate(halu_loader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        logits         = s4_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs          = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        s4_scores.extend(probs.tolist())
        if (i + 1) % 100 == 0:
            print(f"  S4: {(i+1)*BATCH_SIZE}/{len(examples)}", flush=True)

s4_scores = np.array(s4_scores)

# --- Fusion S2+S4 (simple average, no metadata) ---
fusion_scores = (s2_hall + s4_scores) / 2

# --- Use RAGTruth thresholds ---
S2_THRESHOLD      = 0.45   # from RAGTruth experiments
S4_THRESHOLD      = 0.45
FUSION_THRESHOLD  = 0.50

# --- Overall evaluation ---
print(f"\n{'='*80}")
print("HALUBENCH OUT-OF-DOMAIN EVALUATION (RAGTruth excluded)")
print(f"{'='*80}")
print(f"\n{'Method':<25} {'F1':>6} {'P':>6} {'R':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'-'*80}")

results = {}

m_s2 = evaluate(s2_hall, labels, threshold=S2_THRESHOLD)
m_s4 = evaluate(s4_scores, labels, threshold=S4_THRESHOLD)
m_fu = evaluate(fusion_scores, labels, threshold=FUSION_THRESHOLD)

for name, m in [("S2 (Relevance-approx)", m_s2), ("Signal 4 (Finetuned)", m_s4), ("Fusion S2+S4", m_fu)]:
    print(f"{name:<25} {m['f1']:>6} {m['precision']:>6} {m['recall']:>6} {m['auroc']:>7} {m['auprc']:>7} {m['ece']:>7}")
    results[name] = m

# --- Per source domain ---
print(f"\n{'='*80}")
print("PER-DOMAIN BREAKDOWN (Signal 4)")
print(f"{'='*80}")
print(f"\n{'Domain':<15} {'n':>6} {'pos_rate':>9} {'F1':>6} {'AUROC':>7} {'AUPRC':>7}")
print(f"{'-'*80}")

domain_results = {}
sources = sorted(set(e['source'] for e in examples))
for source in sources:
    mask = np.array([e['source'] == source for e in examples])
    m = evaluate(s4_scores[mask], labels[mask], threshold=S4_THRESHOLD)
    print(f"{source:<15} {m['n']:>6} {m['pos_rate']:>9} {m['f1']:>6} {m['auroc']:>7} {m['auprc']:>7}")
    domain_results[source] = m

# --- Save ---
output = {
    'overall': results,
    'per_domain_s4': domain_results,
}
with open('/workspace/halubench_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /workspace/halubench_results.json")
