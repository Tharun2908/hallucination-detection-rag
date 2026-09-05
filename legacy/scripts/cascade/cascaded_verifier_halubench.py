import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Load per-example scores ---
data = json.load(open('/workspace/halubench_per_example_scores.json'))

labels    = np.array([d['label']    for d in data])
s2_hall   = np.array([d['s2_hall']  for d in data])
s4_scores = np.array([d['s4_score'] for d in data])
mc_hall   = np.array([d['mc_hall']  for d in data])

print(f"Examples: {len(data)} | Pos rate: {labels.mean():.3f}")

# --- Fusion S2+S4 (simple average, no metadata since HaluBench has no generator info) ---
# Use RAGTruth-tuned threshold
FUSION_THRESHOLD = 0.50
MC_THRESHOLD     = 0.80

fusion_scores = (s2_hall + s4_scores) / 2
confidence    = np.abs(fusion_scores - 0.5)

escalation_rates = [0, 5, 10, 20, 30, 50, 75, 100]
results = []

print(f"\n{'Escalation':>12} {'F1':>6} {'P':>6} {'R':>6} {'Cost':>6}")
print('-' * 45)

for esc_rate in escalation_rates:
    n_escalate = int(len(confidence) * esc_rate / 100)

    if esc_rate == 0:
        preds = (fusion_scores >= FUSION_THRESHOLD).astype(int)
        cost  = 1.0
    elif esc_rate == 100:
        preds = (mc_hall >= MC_THRESHOLD).astype(int)
        cost  = 11.0
    else:
        escalate_idx  = np.argsort(confidence)[:n_escalate]
        escalate_mask = np.zeros(len(confidence), dtype=bool)
        escalate_mask[escalate_idx] = True

        preds = (fusion_scores >= FUSION_THRESHOLD).astype(int)
        preds[escalate_mask] = (mc_hall[escalate_mask] >= MC_THRESHOLD).astype(int)
        cost = 1 + (esc_rate / 100) * 10

    f1  = round(f1_score(labels, preds, zero_division=0), 4)
    pre = round(precision_score(labels, preds, zero_division=0), 4)
    rec = round(recall_score(labels, preds, zero_division=0), 4)

    results.append({'escalation_rate': esc_rate, 'f1': f1, 'precision': pre, 'recall': rec, 'cost': cost})
    print(f"{esc_rate:>11}% {f1:>6} {pre:>6} {rec:>6} {cost:>5.1f}x")

# --- Plot ---
esc_rates = [r['escalation_rate'] for r in results]
f1_scores = [r['f1'] for r in results]

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
ax.set_title('Performance–Cost Trade-off (HaluBench, out-of-domain)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/cascaded_verifier_halubench_plot.png', dpi=150)
print("\nPlot saved to /workspace/cascaded_verifier_halubench_plot.png")

with open('/workspace/cascaded_verifier_halubench_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved.")
