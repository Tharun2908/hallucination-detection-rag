"""
halubench_calibration.py
Calibration-only adaptation on HaluBench

Experiment:
- Split HaluBench into 10% calibration + 90% test
- Retune threshold on calibration set (model unchanged)
- Compare to zero-shot threshold transfer from RAGTruth

Signals: S2+S4 fusion (no metadata, consistent with HaluBench evaluation)
"""

import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return round(float(ece / len(probs)), 4)

def evaluate(scores, labels, threshold):
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
    }

def best_threshold_on(scores, labels):
    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# --- Load HaluBench per-example scores ---
data = json.load(open('/workspace/halubench_per_example_scores.json'))

labels    = np.array([d['label']    for d in data])
s2_hall   = np.array([d['s2_hall']  for d in data])
s4_scores = np.array([d['s4_score'] for d in data])
sources   = [d['source'] for d in data]

fusion_scores = (s2_hall + s4_scores) / 2

print(f"Total HaluBench examples: {len(data)}")
print(f"Positive rate: {labels.mean():.3f}")

# --- Split: 10% calibration, 90% test ---
indices = np.arange(len(data))
cal_idx, test_idx = train_test_split(indices, test_size=0.9, random_state=42, stratify=labels)

cal_scores  = fusion_scores[cal_idx]
cal_labels  = labels[cal_idx]
test_scores = fusion_scores[test_idx]
test_labels = labels[test_idx]

print(f"\nCalibration set: {len(cal_idx)} | Test set: {len(test_idx)}")
print(f"Cal pos rate: {cal_labels.mean():.3f} | Test pos rate: {test_labels.mean():.3f}")

# --- Zero-shot threshold (from RAGTruth) ---
RAGTRUTH_THRESHOLD = 0.50
m_zeroshot = evaluate(test_scores, test_labels, RAGTRUTH_THRESHOLD)
print(f"\n--- Zero-shot (RAGTruth threshold={RAGTRUTH_THRESHOLD}) ---")
print(f"F1={m_zeroshot['f1']} | P={m_zeroshot['precision']} | R={m_zeroshot['recall']} | AUROC={m_zeroshot['auroc']} | AUPRC={m_zeroshot['auprc']} | ECE={m_zeroshot['ece']}")

# --- Calibrated threshold (tuned on 10% HaluBench) ---
cal_threshold, cal_f1 = best_threshold_on(cal_scores, cal_labels)
m_calibrated = evaluate(test_scores, test_labels, cal_threshold)
print(f"\n--- Calibrated (threshold={cal_threshold:.2f}, cal F1={cal_f1:.4f}) ---")
print(f"F1={m_calibrated['f1']} | P={m_calibrated['precision']} | R={m_calibrated['recall']} | AUROC={m_calibrated['auroc']} | AUPRC={m_calibrated['auprc']} | ECE={m_calibrated['ece']}")

# --- Per domain breakdown ---
print(f"\n--- Per Domain (Calibrated threshold={cal_threshold:.2f}) ---")
domain_results = {}
for source in sorted(set(sources)):
    mask = np.array([s == source for s in sources])
    test_mask = mask[test_idx]
    if test_mask.sum() == 0:
        continue
    m = evaluate(test_scores[test_mask], test_labels[test_mask], cal_threshold)
    print(f"  {source:<15} n={m['n']:>4} | F1={m['f1']} | AUROC={m['auroc']} | AUPRC={m['auprc']}")
    domain_results[source] = m

# --- Summary table ---
print(f"\n{'='*70}")
print("SUMMARY: Zero-shot vs Calibration-only Adaptation")
print(f"{'='*70}")
print(f"{'Setting':<35} {'F1':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'-'*70}")
print(f"{'Zero-shot (RAGTruth threshold)':<35} {m_zeroshot['f1']:>6} {m_zeroshot['auroc']:>7} {m_zeroshot['auprc']:>7} {m_zeroshot['ece']:>7}")
print(f"{'Calibrated (10% HaluBench)':<35} {m_calibrated['f1']:>6} {m_calibrated['auroc']:>7} {m_calibrated['auprc']:>7} {m_calibrated['ece']:>7}")

# Save
results = {
    'calibration_set_size': len(cal_idx),
    'test_set_size': len(test_idx),
    'ragtruth_threshold': RAGTRUTH_THRESHOLD,
    'calibrated_threshold': cal_threshold,
    'zero_shot': m_zeroshot,
    'calibrated': m_calibrated,
    'per_domain_calibrated': domain_results,
}
with open('/workspace/halubench_calibration_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to /workspace/halubench_calibration_results.json")
