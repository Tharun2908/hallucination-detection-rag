import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

# --- Load data ---
with open('/workspace/relevance_results_train_v2.json') as f:
    rel_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_train_oof.json') as f:
    s4_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    rel_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/minicheck_results_test_7b.json') as f:
    mc7b_test = {r['idx']: r for r in json.load(f)}

# --- Build features ---
def extract(rel_map, s4_map):
    common = sorted(rel_map.keys() & s4_map.keys())
    X, y, cats, idxs = [], [], [], []
    for idx in common:
        r2, r4 = rel_map[idx], s4_map[idx]
        if r2['raw_min_relevance'] is None or r4['signal4_score'] is None:
            continue
        X.append([norm_s2(r2['raw_min_relevance']), r4['signal4_score']])
        y.append(int(r2['ground_truth_hallucination']))
        cats.append([r2['task_type'], r2['model']])
        idxs.append(idx)
    return np.array(X), np.array(y), cats, idxs

X_train, y_train, cats_train, _ = extract(rel_train, s4_train)
X_test,  y_test,  cats_test,  test_idxs = extract(rel_test, s4_test)

# --- Train logistic regression ---
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(cats_train)
X_train_full = np.hstack([X_train, ohe.transform(cats_train)])
X_test_full  = np.hstack([X_test,  ohe.transform(cats_test)])

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_full, y_train)

# Tune threshold on train
train_prob = clf.predict_proba(X_train_full)[:, 1]
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (train_prob >= t).astype(int)
    f1 = f1_score(y_train, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Lightweight threshold: {best_threshold:.2f} (train F1={best_f1:.4f})")

# Test probabilities
test_prob = clf.predict_proba(X_test_full)[:, 1]

# --- MiniCheck-7B scores ---
mc_scores = np.array([
    1 - mc7b_test[idx]['minicheck_score']
    if idx in mc7b_test and mc7b_test[idx]['minicheck_score'] is not None
    else None
    for idx in test_idxs
])

# Filter valid examples (both scores available)
valid_mask = np.array([s is not None for s in mc_scores])
test_prob_v  = test_prob[valid_mask]
y_test_v     = y_test[valid_mask]
mc_scores_v  = mc_scores[valid_mask].astype(float)

print(f"Valid examples: {valid_mask.sum()}")

# --- Cascaded evaluation ---
# Confidence = distance from 0.5
confidence = np.abs(test_prob_v - 0.5)

# MiniCheck threshold (from RAGTruth experiments)
MC_THRESHOLD = 0.8  # 1 - 0.2

escalation_rates = [0, 5, 10, 20, 30, 50, 75, 100]
results = []

for esc_rate in escalation_rates:
    n_escalate = int(len(confidence) * esc_rate / 100)

    if esc_rate == 0:
        # Lightweight only
        preds = (test_prob_v >= best_threshold).astype(int)
        cost = 1.0
    elif esc_rate == 100:
        # MiniCheck only
        preds = (mc_scores_v >= MC_THRESHOLD).astype(int)
        cost = 11.0  # approximate relative cost
    else:
        # Escalate least confident examples to MiniCheck (exact count)
        escalate_idx = np.argsort(confidence)[:n_escalate]
        escalate_mask = np.zeros(len(confidence), dtype=bool)
        escalate_mask[escalate_idx] = True

        preds = (test_prob_v >= best_threshold).astype(int)
        # Override with MiniCheck for escalated examples
        preds[escalate_mask] = (mc_scores_v[escalate_mask] >= MC_THRESHOLD).astype(int)
        cost = 1 + (esc_rate / 100) * 10  # relative cost

    f1  = round(f1_score(y_test_v, preds, zero_division=0), 4)
    pre = round(precision_score(y_test_v, preds, zero_division=0), 4)
    rec = round(recall_score(y_test_v, preds, zero_division=0), 4)

    results.append({
        'escalation_rate': esc_rate,
        'f1': f1,
        'precision': pre,
        'recall': rec,
        'cost': cost,
        'n_escalated': n_escalate,
    })

    print(f"Escalation {esc_rate:3d}% | F1={f1} | P={pre} | R={rec} | Cost={cost:.1f}x")

# --- Plot ---
esc_rates = [r['escalation_rate'] for r in results]
f1_scores  = [r['f1'] for r in results]
costs      = [r['cost'] for r in results]

lightweight_f1 = results[0]['f1']
minicheck_f1   = results[-1]['f1']

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(esc_rates, f1_scores, 'b-o', linewidth=2, markersize=6)
ax.axhline(y=lightweight_f1, color='green', linestyle='--', label=f'Lightweight only (F1={lightweight_f1})')
ax.axhline(y=minicheck_f1,   color='orange', linestyle='--', label=f'MiniCheck-7B only (F1={minicheck_f1})')

for r in results:
    ax.annotate(f"{r['cost']:.0f}x", (r['escalation_rate'], r['f1']),
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

ax.set_xlabel('Escalation rate to MiniCheck-7B (%)')
ax.set_ylabel('Final F1 score')
ax.set_title('Performance–Cost Trade-off for Cascaded Verification')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/cascaded_verifier_plot.png', dpi=150)
print("\nPlot saved to /workspace/cascaded_verifier_plot.png")

# Save results
with open('/workspace/cascaded_verifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to /workspace/cascaded_verifier_results.json")
