import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

with open('/workspace/nli_results_train_v2.json') as f:
    nli_train = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/relevance_results_train_v2.json') as f:
    rel_train = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_train.json') as f:
    s4_train = {ex['idx']: ex for ex in json.load(f)}

with open('/workspace/nli_results_test_v2.json') as f:
    nli_test = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    rel_test = {ex['idx']: ex for ex in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {ex['idx']: ex for ex in json.load(f)}

S2_MIN, S2_MAX = -11.430, 10.641

def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

def extract(nli_map, rel_map, s4_map):
    common = sorted(nli_map.keys() & rel_map.keys() & s4_map.keys())
    numeric, categorical, labels = [], [], []
    for idx in common:
        r1, r2, r4 = nli_map[idx], rel_map[idx], s4_map[idx]
        if any(x is None for x in [r1['nli_score'], r2['raw_min_relevance'], r4['signal4_score']]):
            continue
        assert r1['ground_truth_hallucination'] == r2['ground_truth_hallucination'] == r4['ground_truth_hallucination']
        labels.append(int(r1['ground_truth_hallucination']))
        numeric.append([
            r1['nli_score'],
            norm_s2(r2['raw_min_relevance']),
            r4['signal4_score'],
        ])
        categorical.append([r1['task_type'], r1['model']])
    return np.array(numeric), categorical, np.array(labels)

train_num, train_cat, y_train = extract(nli_train, rel_train, s4_train)
test_num,  test_cat,  y_test  = extract(nli_test,  rel_test,  s4_test)

print(f"Train: {len(y_train)} | Test: {len(y_test)}")

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_cat)

X_train = np.hstack([train_num, ohe.transform(train_cat)])
X_test  = np.hstack([test_num,  ohe.transform(test_cat)])

cat_feature_names = ohe.get_feature_names_out(['task_type', 'model']).tolist()
feature_names = ['NLI score', 'Relevance score', 'Signal4 score'] + cat_feature_names

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

print(f"\n--- Logistic Regression Fusion S1+S2+S4 (no S3, threshold={best_threshold:.2f}) ---")
print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"AUROC:     {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("(TN, FP)\n(FN, TP)")

print(f"\n--- Learned Coefficients ---")
for name, coef in sorted(zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:35s}: {coef:+.4f}")

results = {
    "method": "Logistic Regression S1+S2+S4 (no S3)",
    "best_threshold": best_threshold,
    "train_f1": round(best_f1, 4),
    "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    "test_auroc":     round(roc_auc_score(y_test, y_prob), 4),
    "coefficients":   {name: round(float(coef), 4) for name, coef in zip(feature_names, clf.coef_[0])},
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}
with open('/workspace/fusion_logreg_no_s3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to /workspace/fusion_logreg_no_s3_results.json")
