import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, roc_auc_score
from collections import Counter

# --- Load data ---
with open('/workspace/nli_results_test_v2.json') as f:
    s1 = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    s2 = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4 = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal5_results_test_mean.json') as f:
    s5 = {r['idx']: r for r in json.load(f)}

S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

common = sorted(set(s1.keys()) & set(s2.keys()) & set(s4.keys()) & set(s5.keys()))

examples = []
for idx in common:
    r1, r2, r4, r5 = s1[idx], s2[idx], s4[idx], s5[idx]
    if any(x is None for x in [r1['nli_score'], r2['raw_min_relevance'], r4['signal4_score'], r5['signal5_score']]):
        continue
    assert r1['ground_truth_hallucination'] == r2['ground_truth_hallucination'] == r4['ground_truth_hallucination'] == r5['ground_truth_hallucination']

    # Align direction: higher = more likely hallucination
    s1_h = 1 - r1['nli_score']
    s2_h = 1 - norm_s2(r2['raw_min_relevance'])
    s4_v = r4['signal4_score']           # already: higher = hallucination
    s5_h = 1 - r5['signal5_score']

    examples.append({
        'idx':       idx,
        'label':     r1['ground_truth_hallucination'],
        'task_type': r1['task_type'],
        'model':     r1['model'],
        's1_h':      s1_h,
        's2_h':      s2_h,
        's4':        s4_v,
        's5_h':      s5_h,
        # raw for fusion evaluation
        'nli_score':        r1['nli_score'],
        'relevance_score':  norm_s2(r2['raw_min_relevance']),
        'signal4_score':    r4['signal4_score'],
    })

print(f"Examples: {len(examples)}", flush=True)

X_raw = np.array([[e['s1_h'], e['s2_h'], e['s4'], e['s5_h']] for e in examples])
labels = np.array([e['label'] for e in examples])

# --- Normalize for clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- Sweep k=2,3,4 ---
print("\nSweeping k...", flush=True)
silhouette_scores = {}
best_k, best_sil = 2, -1
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores[k] = round(sil, 4)
    if sil > best_sil:
        best_sil = sil
        best_k = k
    print(f"k={k} | Silhouette={sil:.4f}", flush=True)
print(f"\nBest k by silhouette: {best_k} (score={best_sil:.4f})", flush=True)

# --- Use k=3 for interpretability ---
K = 3
km = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_assignments = km.fit_predict(X_scaled)

for i, ex in enumerate(examples):
    ex['cluster'] = int(cluster_assignments[i])

print(f"\n--- Cluster Analysis (k={K}) ---", flush=True)

results = {}

# Thresholds from previous experiments
S2_THRESHOLD = 0.45
S4_THRESHOLD = 0.45

def eval_signal(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    return {
        'f1':        round(f1_score(labels, preds, zero_division=0), 4),
        'precision': round(precision_score(labels, preds, zero_division=0), 4),
        'recall':    round(recall_score(labels, preds, zero_division=0), 4),
        'auroc':     round(roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5, 4),
    }

for c in range(K):
    mask = cluster_assignments == c
    cluster_examples = [e for e in examples if e['cluster'] == c]
    cluster_labels   = labels[mask]

    n          = len(cluster_examples)
    hall_rate  = float(cluster_labels.mean())
    task_dist  = Counter(e['task_type'] for e in cluster_examples)
    model_dist = Counter(e['model'] for e in cluster_examples)

    # Signal scores in this cluster
    s1_scores = np.array([e['s1_h'] for e in cluster_examples])
    s2_scores = np.array([e['s2_h'] for e in cluster_examples])
    s4_scores = np.array([e['s4']   for e in cluster_examples])
    s5_scores = np.array([e['s5_h'] for e in cluster_examples])

    # Fusion score: simple average S2+S4 (inverted back)
    fusion_scores = (s2_scores + s4_scores) / 2

    # Per-signal performance
    s2_metrics      = eval_signal(s2_scores, cluster_labels, S2_THRESHOLD)
    s4_metrics      = eval_signal(s4_scores, cluster_labels, S4_THRESHOLD)
    fusion_metrics  = eval_signal(fusion_scores, cluster_labels, 0.50)

    print(f"\n=== Cluster {c} (n={n}, hall_rate={hall_rate:.3f}) ===")
    print(f"  Task types:  {dict(task_dist)}")
    print(f"  Models:      {dict(model_dist)}")
    print(f"  Avg scores:  S1={s1_scores.mean():.3f} | S2={s2_scores.mean():.3f} | S4={s4_scores.mean():.3f} | S5={s5_scores.mean():.3f}")
    print(f"  S2 alone:    F1={s2_metrics['f1']} | P={s2_metrics['precision']} | R={s2_metrics['recall']} | AUROC={s2_metrics['auroc']}")
    print(f"  S4 alone:    F1={s4_metrics['f1']} | P={s4_metrics['precision']} | R={s4_metrics['recall']} | AUROC={s4_metrics['auroc']}")
    print(f"  S2+S4 fusion: F1={fusion_metrics['f1']} | P={fusion_metrics['precision']} | R={fusion_metrics['recall']} | AUROC={fusion_metrics['auroc']}")

    results[c] = {
        'n': n, 'hall_rate': round(hall_rate, 3),
        'task_dist': dict(task_dist), 'model_dist': dict(model_dist),
        'avg_s1': round(float(s1_scores.mean()), 3),
        'avg_s2': round(float(s2_scores.mean()), 3),
        'avg_s4': round(float(s4_scores.mean()), 3),
        'avg_s5': round(float(s5_scores.mean()), 3),
        's2_metrics': s2_metrics,
        's4_metrics': s4_metrics,
        'fusion_metrics': fusion_metrics,
    }

# Save results
output = {'silhouette_scores': silhouette_scores, 'best_k_silhouette': best_k, 'chosen_k': K, 'clusters': results}
with open('/workspace/clustering_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /workspace/clustering_results.json", flush=True)
