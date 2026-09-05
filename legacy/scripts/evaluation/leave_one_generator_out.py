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

generators = sorted(set(e['model'] for e in all_examples))
print(f"Generators: {generators}")
print(f"Total examples: {len(all_examples)}")

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

print(f"\n{'='*95}")
print(f"LEAVE-ONE-GENERATOR-OUT EVALUATION")
print(f"{'='*95}")
print(f"\n{'Held-out Model':<25} {'n_test':>7} {'pos_rate':>9} {'F1':>6} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7} {'Setting'}")
print(f"{'-'*95}")

for held_out in generators:
    logo_train = [e for e in all_examples if e['model'] != held_out]
    logo_test  = [e for e in all_examples if e['model'] == held_out]

    if len(logo_test) == 0:
        continue

    X_tr = np.array([[e['s2'], e['s4']] for e in logo_train])
    y_tr = np.array([e['label'] for e in logo_train])
    cats_tr = [[e['task_type'], e['model']] for e in logo_train]

    X_te = np.array([[e['s2'], e['s4']] for e in logo_test])
    y_te = np.array([e['label'] for e in logo_test])
    cats_te = [[e['task_type'], e['model']] for e in logo_test]

    # Without metadata
    clf_nm = LogisticRegression(max_iter=1000, random_state=42)
    clf_nm.fit(X_tr, y_tr)
    tr_prob_nm = clf_nm.predict_proba(X_tr)[:, 1]
    te_prob_nm = clf_nm.predict_proba(X_te)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (tr_prob_nm >= t).astype(int)
        f1 = f1_score(y_tr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    m_nm = evaluate(te_prob_nm, y_te, threshold=best_t)

    # With metadata
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(cats_tr)
    X_tr_m = np.hstack([X_tr, ohe.transform(cats_tr)])
    X_te_m = np.hstack([X_te, ohe.transform(cats_te)])

    clf_m = LogisticRegression(max_iter=1000, random_state=42)
    clf_m.fit(X_tr_m, y_tr)
    tr_prob_m = clf_m.predict_proba(X_tr_m)[:, 1]
    te_prob_m = clf_m.predict_proba(X_te_m)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        preds = (tr_prob_m >= t).astype(int)
        f1 = f1_score(y_tr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    m_m = evaluate(te_prob_m, y_te, threshold=best_t)

    print(f"{held_out:<25} {m_nm['n']:>7} {m_nm['pos_rate']:>9} {m_nm['f1']:>6} {m_nm['auroc']:>7} {m_nm['auprc']:>7} {m_nm['ece']:>7}  no meta")
    print(f"{'':<25} {m_m['n']:>7} {m_m['pos_rate']:>9} {m_m['f1']:>6} {m_m['auroc']:>7} {m_m['auprc']:>7} {m_m['ece']:>7}  with meta")
    print()

    results[held_out] = {
        'no_metadata':   m_nm,
        'with_metadata': m_m,
        'n_train':       len(logo_train),
    }

with open('/workspace/leave_one_generator_out_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to /workspace/leave_one_generator_out_results.json")
