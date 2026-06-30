import json
import numpy as np
from datasets import load_dataset
from minicheck.minicheck import MiniCheck
import nltk
nltk.download('punkt_tab', quiet=True)

BATCH_SIZE = 16

if __name__ == '__main__':
    print("Loading HaluBench...", flush=True)
    ds = load_dataset('PatronusAI/HaluBench', split='test')
    ds = ds.filter(lambda x: x['source_ds'] != 'RAGTruth')

    examples = [{
        'context': ex['passage'],
        'answer':  ex['answer'],
        'label':   1 if ex['label'] == 'FAIL' else 0,
        'source':  ex['source_ds'],
    } for ex in ds]

    print(f"Examples: {len(examples)}", flush=True)

    print("Loading MiniCheck-7B...", flush=True)
    scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', batch_size=BATCH_SIZE, max_model_len=4096)
    print("MiniCheck loaded.", flush=True)

    mc_scores = []
    for i in range(0, len(examples), BATCH_SIZE):
        batch  = examples[i:i+BATCH_SIZE]
        docs   = [e['context'] for e in batch]
        claims = [e['answer']  for e in batch]
        _, probs, _, _ = scorer.score(docs=docs, claims=claims)
        mc_scores.extend([1 - p for p in probs])
        if (i+BATCH_SIZE) % 1000 == 0:
            print(f"  MC: {min(i+BATCH_SIZE, len(examples))}/{len(examples)}", flush=True)

    # Load S2+S4 scores
    s2s4 = json.load(open('/workspace/halubench_s2s4_scores.json'))

    # Merge
    results = []
    for i, ex in enumerate(examples):
        results.append({
            'idx':      i,
            'label':    ex['label'],
            'source':   ex['source'],
            's2_hall':  s2s4[i]['s2_hall'],
            's4_score': s2s4[i]['s4_score'],
            'mc_hall':  round(float(mc_scores[i]), 4),
        })

    with open('/workspace/halubench_per_example_scores.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} examples.", flush=True)
