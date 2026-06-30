"""
ragtruth_plusplus_eval.py
Label-noise robustness evaluation on RAGTruth++

RAGTruth++ has re-annotated labels with more subtle hallucinations flagged.
We evaluate our saved signals on this re-annotated test set.
Key question: do conclusions hold under improved annotation quality?
"""
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

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

# --- Load RAGTruth++ ---
print("Loading RAGTruth++...", flush=True)
messages = pd.read_csv('hf://datasets/blue-guardrails/ragtruth-plus-plus/messages.csv')
spans    = pd.read_csv('hf://datasets/blue-guardrails/ragtruth-plus-plus/hallucination_spans.csv')

import json as json_lib
messages['meta_parsed'] = messages['meta'].apply(json_lib.loads)
messages['model']       = messages['meta_parsed'].apply(lambda x: x.get('model'))
assistant = messages[messages['role'] == 'assistant'].copy()
hall_ids  = set(spans['message_stable_id'].unique())
assistant['is_hallucinated'] = assistant['stable_id'].isin(hall_ids).astype(int)
print(f"RAGTruth++ examples: {len(assistant)} | pos_rate: {assistant['is_hallucinated'].mean():.3f}", flush=True)

# --- Load RAGTruth to get context and match signals ---
print("Loading RAGTruth for text matching...", flush=True)
ds_train = load_dataset('wandb/RAGTruth-processed', split='train')
ds_test  = load_dataset('wandb/RAGTruth-processed', split='test')

# Build lookup: text -> (split, idx)
text_to_info = {}
for idx, ex in enumerate(ds_train):
    text_to_info[ex['output'][:100]] = ('train', idx)
for idx, ex in enumerate(ds_test):
    text_to_info[ex['output'][:100]] = ('test', idx)

# --- Load saved signal scores ---
print("Loading signal scores...", flush=True)
with open('/workspace/nli_results_train_v2.json') as f:
    s1_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/nli_results_test_v2.json') as f:
    s1_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_train_v2.json') as f:
    s2_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    s2_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_train.json') as f:
    s4_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_train_7b.json') as f:
    mc_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_test_7b.json') as f:
    mc_test = {r['idx']: r for r in json.load(f)}

# --- Align RAGTruth++ with signal scores ---
aligned = []
for _, row in assistant.iterrows():
    key = row['text'][:100]
    if key not in text_to_info:
        continue
    split, idx = text_to_info[key]
    s1_map = s1_train if split == 'train' else s1_test
    s2_map = s2_train if split == 'train' else s2_test
    s4_map = s4_train if split == 'train' else s4_test
    mc_map = mc_train if split == 'train' else mc_test

    if idx not in s1_map or idx not in s2_map or idx not in s4_map or idx not in mc_map:
        continue

    r1, r2, r4, rmc = s1_map[idx], s2_map[idx], s4_map[idx], mc_map[idx]
    if any(x is None for x in [r1['nli_score'], r2['raw_min_relevance'], r4['signal4_score'], rmc['minicheck_score']]):
        continue

    aligned.append({
        'rt_label':    int(r1['ground_truth_hallucination']),  # original RAGTruth label
        'rtp_label':   int(row['is_hallucinated']),             # RAGTruth++ label
        's1_hall':     1 - r1['nli_score'],
        's2_hall':     1 - norm_s2(r2['raw_min_relevance']),
        's4_score':    r4['signal4_score'],
        'mc_hall':     1 - rmc['minicheck_score'],
        'model':       row['model'],
    })

print(f"Aligned examples: {len(aligned)}", flush=True)

rt_labels  = np.array([e['rt_label']  for e in aligned])
rtp_labels = np.array([e['rtp_label'] for e in aligned])
s1_scores  = np.array([e['s1_hall']   for e in aligned])
s2_scores  = np.array([e['s2_hall']   for e in aligned])
s4_scores  = np.array([e['s4_score']  for e in aligned])
mc_scores  = np.array([e['mc_hall']   for e in aligned])
fusion_scores = (s2_scores + s4_scores) / 2

print(f"\nLabel comparison:")
print(f"  Original RAGTruth pos_rate:  {rt_labels.mean():.3f}")
print(f"  RAGTruth++ pos_rate:         {rtp_labels.mean():.3f}")
print(f"  Label changes: {(rt_labels != rtp_labels).sum()} / {len(aligned)}")

# Use RAGTruth train thresholds
THRESHOLDS = {'S1': 0.35, 'S2': 0.45, 'S4': 0.45, 'MC': 0.80, 'Fusion': 0.50}

print(f"\n{'='*80}")
print("RESULTS ON ORIGINAL RAGTruth LABELS")
print(f"{'='*80}")
print(f"{'Method':<25} {'F1':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'-'*80}")
for name, scores, t in [
    ('Signal 1 (NLI)', s1_scores, THRESHOLDS['S1']),
    ('Signal 2 (Relevance)', s2_scores, THRESHOLDS['S2']),
    ('Signal 4 (Finetuned)', s4_scores, THRESHOLDS['S4']),
    ('MiniCheck-7B', mc_scores, THRESHOLDS['MC']),
    ('Fusion S2+S4', fusion_scores, THRESHOLDS['Fusion']),
]:
    m = evaluate(scores, rt_labels, threshold=t)
    print(f"{name:<25} {m['f1']:>6} {m['auroc']:>7} {m['auprc']:>7} {m['ece']:>7}")

print(f"\n{'='*80}")
print("RESULTS ON RAGTruth++ LABELS (re-annotated)")
print(f"{'='*80}")
print(f"{'Method':<25} {'F1':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'-'*80}")
results = {}
for name, scores, t in [
    ('Signal 1 (NLI)', s1_scores, THRESHOLDS['S1']),
    ('Signal 2 (Relevance)', s2_scores, THRESHOLDS['S2']),
    ('Signal 4 (Finetuned)', s4_scores, THRESHOLDS['S4']),
    ('MiniCheck-7B', mc_scores, THRESHOLDS['MC']),
    ('Fusion S2+S4', fusion_scores, THRESHOLDS['Fusion']),
]:
    m = evaluate(scores, rtp_labels, threshold=t)
    print(f"{name:<25} {m['f1']:>6} {m['auroc']:>7} {m['auprc']:>7} {m['ece']:>7}")
    results[name] = m

with open('/workspace/ragtruth_plusplus_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to /workspace/ragtruth_plusplus_results.json")
