from datasets import load_dataset
from collections import Counter
import ast

ds = load_dataset("wandb/RAGTruth-processed", split="train")

def parse(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(x) if x else []
    except Exception: return []

# 1. What raw label_type values actually exist, and how often?
type_counts = Counter()
for ex in ds:
    for span in parse(ex["hallucination_labels"]):
        t = span.get("label_type") or span.get("type") or span.get("label")
        type_counts[t] += 1
print("Raw span types present:", dict(type_counts))

# 2. THE decisive cross-tab: for each example, does its set of raw types
#    survive into your binary label?
your_pos = lambda lab: (lab.get("evident_conflict",0)>0) or (lab.get("baseless_info",0)>0)

# examples whose ONLY hallucination type is Subtle Conflict
subtle_conflict_only = 0
sc_only_labeled_faithful = 0
for ex in ds:
    spans = parse(ex["hallucination_labels"])
    types = {(s.get("label_type") or s.get("type") or s.get("label")) for s in spans}
    types.discard(None)
    if types == {"Subtle Conflict"}:
        subtle_conflict_only += 1
        if not your_pos(ex["hallucination_labels_processed"]):
            sc_only_labeled_faithful += 1

print(f"\nExamples whose ONLY hallucination is Subtle Conflict: {subtle_conflict_only}")
print(f"  ...of those, labeled FAITHFUL by your function: {sc_only_labeled_faithful}")

# 3. General leak: any example with >=1 raw span but binary=faithful?
leaked = 0
for ex in ds:
    spans = parse(ex["hallucination_labels"])
    if len(spans) > 0 and not your_pos(ex["hallucination_labels_processed"]):
        leaked += 1
print(f"\nExamples with raw hallucination spans but binary=FAITHFUL: {leaked}")
