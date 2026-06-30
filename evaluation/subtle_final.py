from datasets import load_dataset
from collections import Counter
import json

ds = load_dataset("wandb/RAGTruth-processed", split="train")

def parse(x):
    x = (x or "").strip()
    if not x or x == "[]": return []
    try: return json.loads(x)
    except Exception: return []

your_pos = lambda lab: (lab.get("evident_conflict",0)>0) or (lab.get("baseless_info",0)>0)

# 1. Real distribution of the four types
type_counts = Counter()
for ex in ds:
    for s in parse(ex["hallucination_labels"]):
        type_counts[s.get("label_type")] += 1
print("Raw span types present:", dict(type_counts))

# 2. Map raw types -> which processed field they land in.
#    For each example, compare its raw type set against your binary call.
sc_only = 0          # only hallucination is Subtle Conflict
sc_only_faithful = 0 # ...and your function calls it faithful
leaked = 0           # ANY example with spans but binary=faithful
leaked_types = Counter()

for ex in ds:
    spans = parse(ex["hallucination_labels"])
    if not spans:
        continue
    types = {s.get("label_type") for s in spans}
    pos = your_pos(ex["hallucination_labels_processed"])
    if not pos:
        leaked += 1
        for t in types: leaked_types[t] += 1
    if types == {"Subtle Conflict"}:
        sc_only += 1
        if not pos:
            sc_only_faithful += 1

print(f"\nExamples with spans but binary=FAITHFUL (leak): {leaked}")
print(f"  leaked examples by type present: {dict(leaked_types)}")
print(f"\nSubtle-Conflict-only examples: {sc_only}")
print(f"  ...labeled FAITHFUL by your function: {sc_only_faithful}")

# 3. Does 'baseless_info' actually aggregate BOTH baseless subtypes? Spot-check:
#    examples whose ONLY type is 'Subtle Baseless...' but processed baseless_info>0
sb_names = [t for t in type_counts if t and "Baseless" in t and "Subtle" in t]
print(f"\nSubtle-baseless type string(s) seen: {sb_names}")
