"""
halubench_minicheck.py
MiniCheck-7B evaluation on HaluBench (out-of-domain)
"""
import json
import numpy as np
from datasets import load_dataset
from minicheck.minicheck import MiniCheck
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BATCH_SIZE = 16

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return round(float(ece / len(probs)), 4)

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

if __name__ == '__main__':
    print("Loading HaluBench...", flush=True)
    ds = load_dataset('PatronusAI/HaluBench', split='test')
    ds = ds.filter(lambda x: x['source_ds'] != 'RAGTruth')
    print(f"Examples: {len(ds)}", flush=True)

    examples = [{
        'context': ex['passage'],
        'answer':  ex['answer'],
        'label':   1 if ex['label'] == 'FAIL' else 0,
        'source':  ex['source_ds'],
    } for ex in ds]

    labels = np.array([e['label'] for e in examples])

    print("Loading MiniCheck-7B...", flush=True)
    scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', batch_size=BATCH_SIZE)
    print("MiniCheck loaded.", flush=True)

    print("Scoring...", flush=True)
    all_scores = []
    for i in range(0, len(examples), BATCH_SIZE):
        batch = examples[i:i+BATCH_SIZE]
        docs   = [e['context'] for e in batch]
        claims = [e['answer']  for e in batch]
        _, probs, _, _ = scorer.score(docs=docs, claims=claims)
        # MiniCheck returns support prob — invert for hallucination score
        all_scores.extend([1 - p for p in probs])
        if (i + BATCH_SIZE) % 1000 == 0:
            print(f"  {min(i+BATCH_SIZE, len(examples))}/{len(examples)}", flush=True)

    scores = np.array(all_scores)

    # Use RAGTruth threshold (0.2 was best for MiniCheck-7B)
    m = evaluate(scores, labels, threshold=0.8)  # 1-0.2

    print(f"\n{'='*70}")
    print("MiniCheck-7B on HaluBench (out-of-domain)")
    print(f"{'='*70}")
    print(f"F1:        {m['f1']}")
    print(f"Precision: {m['precision']}")
    print(f"Recall:    {m['recall']}")
    print(f"AUROC:     {m['auroc']}")
    print(f"AUPRC:     {m['auprc']}")
    print(f"ECE:       {m['ece']}")

    # Per domain
    print(f"\n--- Per Domain ---")
    sources = sorted(set(e['source'] for e in examples))
    domain_results = {}
    for source in sources:
        mask = np.array([e['source'] == source for e in examples])
        dm = evaluate(scores[mask], labels[mask], threshold=0.8)
        print(f"{source:<15} F1={dm['f1']} AUROC={dm['auroc']} AUPRC={dm['auprc']}")
        domain_results[source] = dm

    results = {'overall': m, 'per_domain': domain_results}
    with open('/workspace/halubench_minicheck_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/halubench_minicheck_results.json")
