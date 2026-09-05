"""
halubench_scores.py
Save per-example scores for S2, S4, MiniCheck-7B on HaluBench
for cascaded verifier analysis.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from datasets import load_dataset

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LENGTH = 512
S2_MIN, S2_MAX = -11.430, 10.641

print(f"Device: {DEVICE}", flush=True)

# --- Load HaluBench ---
print("Loading HaluBench...", flush=True)
ds = load_dataset('PatronusAI/HaluBench', split='test')
ds = ds.filter(lambda x: x['source_ds'] != 'RAGTruth')

examples = [{
    'context': ex['passage'],
    'answer':  ex['answer'],
    'label':   1 if ex['label'] == 'FAIL' else 0,
    'source':  ex['source_ds'],
} for ex in ds]

print(f"Examples: {len(examples)}", flush=True)

# --- Signal 2: Relevance ---
print("\nScoring S2...", flush=True)
rel_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
s2_raw = []
for i, ex in enumerate(examples):
    score = float(rel_model.predict([(ex['answer'], ex['context'])])[0])
    s2_raw.append(score)
    if (i+1) % 1000 == 0:
        print(f"  S2: {i+1}/{len(examples)}", flush=True)
s2_norm = np.clip((np.array(s2_raw) - S2_MIN) / (S2_MAX - S2_MIN + 1e-8), 0, 1)
s2_hall = 1 - s2_norm
print(f"S2 done. Range: [{s2_hall.min():.3f}, {s2_hall.max():.3f}]", flush=True)

# --- Signal 4: Finetuned DeBERTa ---
print("\nScoring S4...", flush=True)

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
        }

tokenizer = AutoTokenizer.from_pretrained('/workspace/signal4_model')
s4_model  = AutoModelForSequenceClassification.from_pretrained('/workspace/signal4_model')
s4_model  = s4_model.to(DEVICE)
s4_model.eval()

halu_ds = HaluDataset(examples, tokenizer, MAX_LENGTH)
halu_dl = DataLoader(halu_ds, batch_size=BATCH_SIZE, shuffle=False)

s4_scores = []
with torch.no_grad():
    for i, batch in enumerate(halu_dl):
        logits = s4_model(
            input_ids=batch['input_ids'].to(DEVICE),
            attention_mask=batch['attention_mask'].to(DEVICE)
        ).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        s4_scores.extend(probs.tolist())
        if (i+1) % 100 == 0:
            print(f"  S4: {(i+1)*BATCH_SIZE}/{len(examples)}", flush=True)

s4_scores = np.array(s4_scores)
print(f"S4 done. Range: [{s4_scores.min():.3f}, {s4_scores.max():.3f}]", flush=True)

# Save S2+S4 intermediate
with open("/workspace/halubench_s2s4_scores.json", "w") as f:
    json.dump([{"idx": i, "label": examples[i]["label"], "source": examples[i]["source"], "s2_hall": round(float(s2_hall[i]), 4), "s4_score": round(float(s4_scores[i]), 4)} for i in range(len(examples))], f)
print("S2+S4 saved", flush=True)

# Save S2+S4 intermediate
intermediate = [{'idx': i, 'label': examples[i]['label'], 'source': examples[i]['source'],
                 's2_hall': round(float(s2_hall[i]), 4), 's4_score': round(float(s4_scores[i]), 4)}
                for i in range(len(examples))]
with open('/workspace/halubench_s2s4_scores.json', 'w') as f:
    json.dump(intermediate, f)
print("S2+S4 saved.", flush=True)

# --- MiniCheck-7B ---
print("\nScoring MiniCheck-7B...", flush=True)
from minicheck.minicheck import MiniCheck
import nltk
nltk.download('punkt_tab', quiet=True)

scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', batch_size=BATCH_SIZE)

mc_scores = []
for i in range(0, len(examples), BATCH_SIZE):
    batch  = examples[i:i+BATCH_SIZE]
    docs   = [e['context'] for e in batch]
    claims = [e['answer']  for e in batch]
    _, probs, _, _ = scorer.score(docs=docs, claims=claims)
    mc_scores.extend([1 - p for p in probs])
    if (i+BATCH_SIZE) % 1000 == 0:
        print(f"  MC: {min(i+BATCH_SIZE, len(examples))}/{len(examples)}", flush=True)

mc_scores = np.array(mc_scores)
print(f"MiniCheck done. Range: [{mc_scores.min():.3f}, {mc_scores.max():.3f}]", flush=True)

# --- Save per-example scores ---
results = []
for i, ex in enumerate(examples):
    results.append({
        'idx':     i,
        'label':   ex['label'],
        'source':  ex['source'],
        's2_hall': round(float(s2_hall[i]), 4),
        's4_score': round(float(s4_scores[i]), 4),
        'mc_hall': round(float(mc_scores[i]), 4),
    })

with open('/workspace/halubench_per_example_scores.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} examples to /workspace/halubench_per_example_scores.json")
