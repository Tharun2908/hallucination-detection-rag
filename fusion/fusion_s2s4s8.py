import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

with open('/workspace/relevance_results_train_v2.json') as f:
    rel_train = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_train.json') as f:
    s4_train = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal8_results_train.json') as f:
    s8_train = {ex['idx']: ex for ex in json.load(f)}

with open('/workspace/relevance_results_test_v2.json') as f:
    rel_test = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal8_results_test.json') as f:
    s8_test = {ex['idx']: ex for ex in json.load(f)}

S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

def extract(rel_map, s4_map, s8_map):
    common = sorted(rel_map.keys() & s4_map.keys() & s8_map.keys())
    numeric, categorical, labels = [], [], []
    for idx in common:
        r2, r4, r8 = rel_map[idx], s4_map[idx], s8_map[idx]
        if any(x is None for x in [r2['raw_min_relevance'], r4['signal4_score'], r8['signal8_score']]):
            continue
        assert r2['ground_truth_hallucination'] == r4['ground_truth_hallucination'] == r8['ground_truth_hallucination']
        labels.append(int(r2['ground_truth_hallucination']))
        numeric.append([
            norm_s2(r2['raw_min_relevance']),
            r4['signal4_score'],
            r8['signal8_score'],
        ])
        categorical.append([r2['task_type'], r2['model']])
    return np.array(numeric), categorical, np.array(labels)

train_num, train_cat, y_train = extract(rel_train, s4_train, s8_train)
test_num,  test_cat,  y_test  = extract(rel_test,  s4_test,  s8_test)

print(f"Train: {len(y_train)} | Test: {len(y_test)}")

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_cat)

X_train = np.hstack([train_num, ohe.transform(train_cat)])
X_test  = np.hstack([test_num,  ohe.transform(test_cat)])

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

print(f"\n--- Logistic Regression S2+S4+S8 (threshold={best_threshold:.2f}) ---")
print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("(TN, FP)\n(FN, TP)")
