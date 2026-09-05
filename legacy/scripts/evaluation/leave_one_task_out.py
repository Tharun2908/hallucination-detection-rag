import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

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

# --- Load all data ---
with open('/workspace/nli_results_train_v2.json') as f:
    s1_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_train_v2.json') as f:
    s2_train = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_train_oof.json') as f:
    s4_train = {r['idx']: r for r in json.load(f)}

with open('/workspace/nli_results_test_v2.json') as f:
    s1_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/relevance_results_test_v2.json') as f:
    s2_test = {r['idx']: r for r in json.load(f)}
with open('/workspace/signal4_results_test.json') as f:
    s4_test = {r['idx']: r for r in json.load(f)}

# --- Build full aligned dataset (train + test combined) ---
def build_examples(s1_map, s2_map, s4_map):
    common = sorted(s1_map.keys() & s2_map.keys() & s4_map.keys())
    examples = []
    for idx in common:
        r1, r2, r4 = s1_map[idx], s2_map[idx], s4_map[idx]
        if any(x is None for x in [r1['nli_score'], r2['raw_min_relevance'], r4['signal4_score']]):
            continue
        examples.append({
            'idx':       idx,
            'label':     int(r1['ground_truth_hallucination']),
            'task_type': r1['task_type'],
            'model':     r1['model'],
            's2':        norm_s2(r2['raw_min_relevance']),
            's4':        r4['signal4_score'],
        })
    return examples

train_examples = build_examples(s1_train, s2_train, s4_train)
test_examples  = build_examples(s1_test,  s2_test,  s4_test)
all_examples   = train_examples + test_examples

print(f"Total examples: {len(all_examples)} (train={len(train_examples)}, test={len(test_examples)})")

task_types = ['Summary', 'QA', 'Data2txt']

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

results = {}

print(f"\n{'='*90}")
print(f"LEAVE-ONE-TASK-TYPE-OUT EVALUATION")
print(f"{'='*90}")
print(f"\n{'Held-out Task':<15} {'Train Tasks':<25} {'n_test':>7} {'pos_rate':>9} {'F1':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7}")
print(f"{'-'*90}")

for held_out in task_types:
    train_tasks = [t for t in task_types if t != held_out]

    # Split
    loto_train = [e for e in all_examples if e['task_type'] != held_out]
    loto_test  = [e for e in all_examples if e['task_type'] == held_out]

    if len(loto_test) == 0:
        continue

    # Features
    X_tr = np.array([[e['s2'], e['s4']] for e in loto_train])
    y_tr = np.array([e['label'] for e in loto_train])
    cats_tr = [[e['task_type'], e['model']] for e in loto_train]

    X_te = np.array([[e['s2'], e['s4']] for e in loto_test])
    y_te = np.array([e['label'] for e in loto_test])
    cats_te = [[e['task_type'], e['model']] for e in loto_test]

    # Without metadata
    clf_no_meta = LogisticRegression(max_iter=1000, random_state=42)
    clf_no_meta.fit(X_tr, y_tr)
    tr_prob_nm = clf_no_meta.predict_proba(X_tr)[:, 1]
    te_prob_nm = clf_no_meta.predict_proba(X_te)[:, 1]

    # Tune threshold on train
    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (tr_prob_nm >= t).astype(int)
        f1 = f1_score(y_tr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    m_no_meta = evaluate(te_prob_nm, y_te, threshold=best_t)

    # With metadata
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(cats_tr)
    X_tr_meta = np.hstack([X_tr, ohe.transform(cats_tr)])
    X_te_meta = np.hstack([X_te, ohe.transform(cats_te)])

    clf_meta = LogisticRegression(max_iter=1000, random_state=42)
    clf_meta.fit(X_tr_meta, y_tr)
    tr_prob_m = clf_meta.predict_proba(X_tr_meta)[:, 1]
    te_prob_m = clf_meta.predict_proba(X_te_meta)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (tr_prob_m >= t).astype(int)
        f1 = f1_score(y_tr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    m_meta = evaluate(te_prob_m, y_te, threshold=best_t)

    train_str = '+'.join(train_tasks)
    print(f"{held_out:<15} {train_str:<25} {m_no_meta['n']:>7} {m_no_meta['pos_rate']:>9} {m_no_meta['f1']:>6} {m_no_meta['auroc']:>7} {m_no_meta['auprc']:>7} {m_no_meta['ece']:>7}  (no meta)")
    print(f"{'':<15} {'':<25} {m_meta['n']:>7} {m_meta['pos_rate']:>9} {m_meta['f1']:>6} {m_meta['auroc']:>7} {m_meta['auprc']:>7} {m_meta['ece']:>7}  (with meta)")
    print()

    results[held_out] = {
        'no_metadata': m_no_meta,
        'with_metadata': m_meta,
        'n_train': len(loto_train),
        'train_tasks': train_tasks,
    }

# --- In-domain baseline for comparison ---
print(f"\n--- In-domain baseline (train on RAGTruth train, test on RAGTruth test) ---")
X_tr = np.array([[e['s2'], e['s4']] for e in train_examples])
y_tr = np.array([e['label'] for e in train_examples])
cats_tr = [[e['task_type'], e['model']] for e in train_examples]
X_te = np.array([[e['s2'], e['s4']] for e in test_examples])
y_te = np.array([e['label'] for e in test_examples])
cats_te = [[e['task_type'], e['model']] for e in test_examples]

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(cats_tr)
X_tr_full = np.hstack([X_tr, ohe.transform(cats_tr)])
X_te_full = np.hstack([X_te, ohe.transform(cats_te)])

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr_full, y_tr)
tr_prob = clf.predict_proba(X_tr_full)[:, 1]
te_prob = clf.predict_proba(X_te_full)[:, 1]

best_f1, best_t = 0, 0.5
for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
    preds = (tr_prob >= t).astype(int)
    f1 = f1_score(y_tr, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

m_in = evaluate(te_prob, y_te, threshold=best_t)
print(f"F1={m_in['f1']} | AUROC={m_in['auroc']} | AUPRC={m_in['auprc']} | ECE={m_in['ece']}")
results['in_domain'] = m_in

with open('/workspace/leave_one_task_out_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to /workspace/leave_one_task_out_results.json")
