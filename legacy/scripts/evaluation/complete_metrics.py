import json
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

# --- Load all signals ---
with open('/workspace/nli_results_test_v2.json') as f:
    s1 = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    s2 = {r['idx']: r for r in json.load(f)}
with open('/workspace/consistency_results_test.json') as f:
    s3 = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4 = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal5_results_test_mean.json') as f:
    s5 = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_test_roberta.json') as f:
    mc_r = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_test_7b.json') as f:
    mc_7b = {r['idx']: r for r in json.load(f)}

# Load fusion logreg S2+S4 scores
with open('/workspace/nli_results_train_v2.json') as f:
    s1_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_train_v2.json') as f:
    s2_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_train_oof.json') as f:
    s4_train = {r['idx']: r for r in json.load(f)}

# --- ECE computation ---
def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc  = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return round(float(ece / len(probs)), 4)

# --- Threshold sweep ---
def best_threshold(scores, labels):
    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

def compute_all_metrics(hall_scores, labels, threshold=None):
    """hall_scores: higher = more likely hallucination"""
    if threshold is None:
        threshold = best_threshold(hall_scores, labels)
    preds = (hall_scores >= threshold).astype(int)
    return {
        "threshold":  round(threshold, 2),
        "f1":         round(f1_score(labels, preds, zero_division=0), 4),
        "precision":  round(precision_score(labels, preds, zero_division=0), 4),
        "recall":     round(recall_score(labels, preds, zero_division=0), 4),
        "auroc":      round(roc_auc_score(labels, hall_scores), 4),
        "auprc":      round(average_precision_score(labels, hall_scores), 4),
        "ece":        compute_ece(hall_scores, labels),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

# --- Align all indices ---
common = sorted(s1.keys() & s2.keys() & s3.keys() & s4.keys() & s5.keys() & mc_r.keys() & mc_7b.keys())
print(f"Common examples: {len(common)}")

labels = np.array([s1[idx]['ground_truth_hallucination'] for idx in common])
print(f"Positive rate: {labels.mean():.3f}")

# --- Compute scores for each signal ---
# Direction: higher = more likely hallucination
s1_scores  = np.array([1 - s1[idx]['nli_score'] for idx in common])
s2_scores  = np.array([1 - norm_s2(s2[idx]['raw_min_relevance']) for idx in common])
s3_scores  = np.array([1 - s3[idx]['consistency_score'] for idx in common])
s4_scores  = np.array([s4[idx]['signal4_score'] for idx in common])
s5_scores  = np.array([1 - s5[idx]['signal5_score'] for idx in common])
mcr_scores = np.array([1 - mc_r[idx]['minicheck_score'] for idx in common])
mc7_scores = np.array([1 - mc_7b[idx]['minicheck_score'] for idx in common])

# --- Fusion S2+S4 (recompute using logreg) ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

def extract_fusion(rel_map, s4_map):
    common_f = sorted(rel_map.keys() & s4_map.keys())
    X, y, cats = [], [], []
    for idx in common_f:
        r2, r4 = rel_map[idx], s4_map[idx]
        if r2['raw_min_relevance'] is None or r4['signal4_score'] is None:
            continue
        X.append([norm_s2(r2['raw_min_relevance']), r4['signal4_score']])
        y.append(int(r2['ground_truth_hallucination']))
        cats.append([r2['task_type'], r2['model']])
    return np.array(X), np.array(y), cats

X_train, y_train, cats_train = extract_fusion(s2_train, s4_train)
X_test,  y_test,  cats_test  = extract_fusion(s2, s4)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(cats_train)
X_train_full = np.hstack([X_train, ohe.transform(cats_train)])
X_test_full  = np.hstack([X_test,  ohe.transform(cats_test)])

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_full, y_train)
fusion_scores = clf.predict_proba(X_test_full)[:, 1]

# Tune threshold on train
train_prob = clf.predict_proba(X_train_full)[:, 1]
fusion_threshold = best_threshold(train_prob, y_train)

# Align fusion scores with common indices
s4_test_list = [s4[idx] for idx in common]
fusion_aligned = np.array([
    fusion_scores[i] for i, idx in enumerate(sorted(s2.keys() & s4.keys())) if idx in set(common)
])

# --- Compute all metrics ---
signals = {
    "Signal 1 (NLI)":           s1_scores,
    "Signal 2 (Relevance)":     s2_scores,
    "Signal 3 (Consistency)":   s3_scores,
    "Signal 4 (Finetuned)":     s4_scores,
    "Signal 5 (BERTScore)":     s5_scores,
    "MiniCheck roberta-large":  mcr_scores,
    "MiniCheck 7B":             mc7_scores,
}

all_results = {}
print(f"\n{'='*80}")
print(f"{'Method':<30} {'F1':>6} {'P':>6} {'R':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'='*80}")

for name, scores in signals.items():
    m = compute_all_metrics(scores, labels)
    all_results[name] = m
    print(f"{name:<30} {m['f1']:>6} {m['precision']:>6} {m['recall']:>6} {m['auroc']:>7} {m['auprc']:>7} {m['ece']:>7}")

# Fusion
fusion_labels = np.array([s4[idx]['ground_truth_hallucination'] for idx in sorted(s2.keys() & s4.keys()) if idx in set(common)])
m = compute_all_metrics(fusion_scores, fusion_labels, threshold=fusion_threshold)
all_results["Fusion: Logreg S2+S4"] = m
print(f"{'Fusion: Logreg S2+S4':<30} {m['f1']:>6} {m['precision']:>6} {m['recall']:>6} {m['auroc']:>7} {m['auprc']:>7} {m['ece']:>7}")
print(f"{'='*80}")

with open('/workspace/complete_metrics_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to /workspace/complete_metrics_results.json")
