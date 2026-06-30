import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

S1_NAME = "NLI mean"
S2_NAME = "Relevance min"
S3_NAME = "Consistency mean"
S4_NAME = "Finetuned DeBERTa"

print(f"Signals: {S1_NAME} | {S2_NAME} | {S3_NAME} | {S4_NAME}")

with open('/workspace/nli_results_train_v2.json') as f:
    s1_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_train_v2.json') as f:
    s2_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/consistency_results_train.json') as f:
    s3_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_train.json') as f:
    s4_train = {r['idx']: r for r in json.load(f)}

with open('/workspace/nli_results_test_v2.json') as f:
    s1_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    s2_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/consistency_results_test.json') as f:
    s3_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {r['idx']: r for r in json.load(f)}

S2_MIN, S2_MAX = -11.430, 10.641

def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

def align(s1, s2, s3, s4):
    common = sorted(s1.keys() & s2.keys() & s3.keys() & s4.keys())
    labels, sc1, sc2, sc3, sc4 = [], [], [], [], []
    for idx in common:
        r1, r2, r3, r4 = s1[idx], s2[idx], s3[idx], s4[idx]
        if any(x is None for x in [r1['nli_score'], r2['raw_min_relevance'], r3['consistency_score'], r4['signal4_score']]):
            continue
        assert r1['ground_truth_hallucination'] == r2['ground_truth_hallucination'] == r3['ground_truth_hallucination'] == r4['ground_truth_hallucination']
        labels.append(r1['ground_truth_hallucination'])
        sc1.append(r1['nli_score'])
        sc2.append(norm_s2(r2['raw_min_relevance']))
        sc3.append(r3['consistency_score'])
        sc4.append(r4['signal4_score'])
    return np.array(labels), np.array(sc1), np.array(sc2), np.array(sc3), np.array(sc4)

train_labels, tr_s1, tr_s2, tr_s3, tr_s4 = align(s1_train, s2_train, s3_train, s4_train)
test_labels,  te_s1, te_s2, te_s3, te_s4 = align(s1_test,  s2_test,  s3_test,  s4_test)

print(f"Train: {len(train_labels)} | Test: {len(test_labels)}")

tr_s1h, tr_s2h, tr_s3h = 1-tr_s1, 1-tr_s2, 1-tr_s3
te_s1h, te_s2h, te_s3h = 1-te_s1, 1-te_s2, 1-te_s3

all_results = []

def tune_and_evaluate(train_scores, test_scores, train_labels, test_labels, name):
    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (train_scores >= t).astype(int)
        f1 = f1_score(train_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    preds = (test_scores >= best_t).astype(int)
    test_f1  = round(f1_score(test_labels, preds, zero_division=0), 4)
    test_pre = round(precision_score(test_labels, preds, zero_division=0), 4)
    test_rec = round(recall_score(test_labels, preds, zero_division=0), 4)
    test_auc = round(roc_auc_score(test_labels, test_scores), 4)
    cm = confusion_matrix(test_labels, preds).tolist()
    print(f"\n--- {name} (threshold={best_t:.2f}, train F1={best_f1:.4f}) ---")
    print(f"F1:        {test_f1}")
    print(f"Precision: {test_pre}")
    print(f"Recall:    {test_rec}")
    print(f"AUROC:     {test_auc}")
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, preds)}")
    print("(TN, FP)\n(FN, TP)")
    all_results.append({
        "name": name, "threshold": best_t, "train_f1": round(best_f1, 4),
        "test_f1": test_f1, "test_precision": test_pre,
        "test_recall": test_rec, "test_auroc": test_auc,
        "confusion_matrix": cm
    })

tune_and_evaluate(tr_s1h,  te_s1h,  train_labels, test_labels, f"Signal 1 ({S1_NAME})")
tune_and_evaluate(tr_s2h,  te_s2h,  train_labels, test_labels, f"Signal 2 ({S2_NAME})")
tune_and_evaluate(tr_s3h,  te_s3h,  train_labels, test_labels, f"Signal 3 ({S3_NAME})")
tune_and_evaluate(tr_s4,   te_s4,   train_labels, test_labels, f"Signal 4 ({S4_NAME})")
tune_and_evaluate((tr_s1h+tr_s2h+tr_s3h+tr_s4)/4, (te_s1h+te_s2h+te_s3h+te_s4)/4, train_labels, test_labels, "Fusion: Simple Average (S1+S2+S3+S4)")
tune_and_evaluate(0.15*tr_s1h+0.25*tr_s2h+0.10*tr_s3h+0.50*tr_s4, 0.15*te_s1h+0.25*te_s2h+0.10*te_s3h+0.50*te_s4, train_labels, test_labels, "Fusion: Weighted S4-heavy (0.15/0.25/0.10/0.50)")
tune_and_evaluate((tr_s2h+tr_s4)/2, (te_s2h+te_s4)/2, train_labels, test_labels, "Fusion: S2+S4")
tune_and_evaluate((tr_s1h+tr_s4)/2, (te_s1h+te_s4)/2, train_labels, test_labels, "Fusion: S1+S4")
tune_and_evaluate((tr_s3h+tr_s4)/2, (te_s3h+te_s4)/2, train_labels, test_labels, "Fusion: S3+S4")
tune_and_evaluate(0.25*tr_s1h+0.50*tr_s2h+0.25*tr_s3h, 0.25*te_s1h+0.50*te_s2h+0.25*te_s3h, train_labels, test_labels, "Fusion: Weighted S1+S2+S3 (0.25/0.50/0.25)")

with open("/workspace/fusion_results_s4.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to /workspace/fusion_results_s4.json")
