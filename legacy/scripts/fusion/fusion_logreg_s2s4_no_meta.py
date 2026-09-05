import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

with open('/workspace/relevance_results_train_v2.json') as f:
    rel_train = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_train_oof.json') as f:
    s4_train = {ex['idx']: ex for ex in json.load(f)}

with open('/workspace/relevance_results_test_v2.json') as f:
    rel_test = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {ex['idx']: ex for ex in json.load(f)}

S2_MIN, S2_MAX = -11.430, 10.641

def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

def extract(rel_map, s4_map):
    common = sorted(rel_map.keys() & s4_map.keys())
    X, labels = [], []
    for idx in common:
        r2, r4 = rel_map[idx], s4_map[idx]
        if any(x is None for x in [r2['raw_min_relevance'], r4['signal4_score']]):
            continue
        labels.append(int(r2['ground_truth_hallucination']))
        X.append([norm_s2(r2['raw_min_relevance']), r4['signal4_score']])
    return np.array(X), np.array(labels)

X_train, y_train = extract(rel_train, s4_train)
X_test,  y_test  = extract(rel_test,  s4_test)

print(f"Train: {len(y_train)} | Test: {len(y_test)}")
print(f"Features: S2, S4 only (no metadata)")

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

train_prob = clf.predict_proba(X_train)[:, 1]
best_f1, best_threshold = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (train_prob >= t).astype(int)
    f1 = f1_score(y_train, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold on train: {best_threshold:.2f} (F1={best_f1:.4f})")

y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

print(f"\n--- Logistic Regression S2+S4 (no metadata) ---")
print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("(TN, FP)\n(FN, TP)")
