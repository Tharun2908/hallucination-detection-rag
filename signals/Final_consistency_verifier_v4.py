"""
Final_consistency_verifier_v4.py
Signal 3: Consistency-based hallucination detection

Design:
- Given: query + context + candidate answer (from RAGTruth output field)
- Generate multiple alternative answers using Mistral 7B
- Compare candidate answer to alternatives via semantic similarity
- Low agreement = candidate answer deviates from model's natural answer distribution
                = heuristic evidence of hallucination (stability-based signal)

Note: High agreement does not guarantee faithfulness — shared abstention or
style overlap can also produce high scores. Present as a heuristic signal.

Methodology:
- Threshold tuned on train split, applied to test (no leakage)
- Threshold swept from 0.50 to 0.95, best F1 selected
- normalize_embeddings=True for cleaner cosine similarity
- Bad generation filtering: skip outputs under 3 words
- Seeds set for reproducibility (random, numpy, torch, cuda)
- Tokenizer truncation: max_length=3072 (leaves 1024 for generation)
- Output files saved to /workspace (persistent storage — survives pod restarts)

Model  : mistralai/Mistral-7B-Instruct-v0.2
Dataset: wandb/RAGTruth-processed (train + test splits)
Output : /workspace/consistency_results_train.json
         /workspace/consistency_results_test.json
         /workspace/consistency_metrics.json
         /workspace/consistency_run.log  (nohup log)
"""

import json
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME       = "mistralai/Mistral-7B-Instruct-v0.2"
TEMPERATURES     = [0.5, 0.7, 0.9, 1.1, 1.3]
MAX_NEW_TOKENS   = 150
MIN_ANSWER_WORDS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# Output directory — persistent storage, survives pod restarts
OUTPUT_DIR = "/workspace"

# ─── Debug mode — set DEBUG=True for quick 10-example test ───────────────────
DEBUG      = False   # ← set to True for quick test, False for full run
DEBUG_SIZE = 10

# ─── Load models ──────────────────────────────────────────────────────────────
print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")

print("Loading Mistral 7B tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pad token safety
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
print("Mistral 7B loaded.")

# Safe device resolution for tokenized inputs (works with sharded/multi-GPU)
_model_device = next(model.parameters()).device

print("Loading embedding model...")
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE,
)
print("Embedder loaded.")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_hallucinated(example: dict) -> bool:
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0


def build_prompt(query: str, context: str) -> str:
    return (
        "[INST] You are a helpful assistant. "
        "Answer the question based only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer: [/INST]"
    )


def generate_alternatives(query: str, context: str) -> list[str]:
    """
    Generate multiple alternative answers using stochastic decoding
    across a small temperature range [0.5, 1.3].
    Filters out degenerate outputs shorter than MIN_ANSWER_WORDS.
    """
    prompt = build_prompt(query, context)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3072,    # Mistral context window is 4096; leave 1024 for generation
    )
    inputs = {k: v.to(_model_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    answers = []
    for temp in TEMPERATURES:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temp,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()

        if len(generated.split()) >= MIN_ANSWER_WORDS:
            answers.append(generated)

    return answers


def compute_consistency_score(given_answer: str, alternatives: list[str]) -> float | None:
    """
    Compare the candidate answer (from dataset) to alternative generations.

    Embeds candidate + alternatives with L2-normalized embeddings.
    Cosine similarity = dot product after normalization.

    High score → candidate aligns with model's natural answer distribution → faithful
    Low score  → candidate deviates from alternatives → possible hallucination

    Returns None if too few valid alternatives.

    Limitation: shared abstention or style overlap can inflate this score.
    Present as a heuristic stability signal, not a precise factual verifier.
    """
    if len(alternatives) < 2:
        return None

    all_texts  = [given_answer] + alternatives
    embeddings = embedder.encode(all_texts, normalize_embeddings=True)

    given_emb = embeddings[0]
    alt_embs  = embeddings[1:]

    similarities = [float(np.dot(given_emb, alt)) for alt in alt_embs]
    return float(np.mean(similarities))


def compute_metrics(results: list[dict], threshold: float) -> dict:
    """Compute evaluation metrics at a given threshold."""
    valid    = [r for r in results if r["consistency_score"] is not None]
    y_true   = [r["ground_truth_hallucination"] for r in valid]
    y_pred   = [r["consistency_score"] < threshold for r in valid]
    y_scores = [1 - r["consistency_score"] for r in valid]

    acc              = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold"        : threshold,
        "accuracy"         : round(acc,   4),
        "precision"        : round(prec,  4),
        "recall"           : round(rec,   4),
        "f1"               : round(f1,    4),
        "auroc"            : round(auroc, 4) if auroc is not None else None,
        "confusion_matrix" : cm,
        "n_examples"       : len(valid),
        "n_errors"         : len(results) - len(valid),
    }


def sweep_threshold(results: list[dict]) -> dict:
    """Sweep thresholds 0.50 → 0.95, return metrics at best F1."""
    thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]
    best = {"f1": -1}
    for t in thresholds:
        m = compute_metrics(results, t)
        if m["f1"] > best["f1"]:
            best = m
    return best


def run_split(split: str) -> list[dict]:
    """Score all examples in a dataset split. Returns results list."""
    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    dataset     = load_dataset("wandb/RAGTruth-processed", split=split)
    if DEBUG:
        dataset = dataset.select(range(DEBUG_SIZE))
        print(f"DEBUG MODE: using {DEBUG_SIZE} examples only")

    output_file = f"{OUTPUT_DIR}/consistency_results_{split}.json"
    # Resume from checkpoint if it exists
    done_indices = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        done_indices = {r["idx"] for r in results}
        print(f"Resuming from checkpoint: {len(results)} examples already done.")
    else:
        results = []

    print(f"Loaded {len(dataset)} examples.")

    for idx, example in enumerate(dataset):
        if idx in done_indices:
            continue

        query        = example["query"]
        context      = example["context"]
        given_answer = example["output"]        # candidate answer to evaluate
        ground_truth = is_hallucinated(example)

        print(f"[{idx+1}/{len(dataset)}] Processing... ", end="", flush=True)

        try:
            alternatives = generate_alternatives(query, context)
            score        = compute_consistency_score(given_answer, alternatives)

            # Build score string separately — can't use :.3f inside f-string conditional
            score_str = f"{score:.3f}" if score is not None else "N/A"
            print(
                f"score={score_str} | "
                f"alts={len(alternatives)} | "
                f"gt={'HALL' if ground_truth else 'FAITH'}"
            )

            results.append({
                "idx"                       : idx,
                "query"                     : query[:100],
                "consistency_score"         : round(score, 4) if score is not None else None,
                "ground_truth_hallucination": ground_truth,
                "model"                     : example.get("model",     "unknown"),
                "task_type"                 : example.get("task_type", "unknown"),
                "n_valid_alternatives"      : len(alternatives),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "idx"                       : idx,
                "query"                     : query[:100],
                "consistency_score"         : None,
                "ground_truth_hallucination": ground_truth,
                "model"                     : example.get("model",     "unknown"),
                "task_type"                 : example.get("task_type", "unknown"),
                "n_valid_alternatives"      : 0,
                "error"                     : str(e),
            })

        # Checkpoint every 50 examples — saved to persistent storage
        if (idx + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint saved ({idx+1}/{len(dataset)}) → {output_file}")

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_file}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
# Step 1: Score train split
train_results = run_split("train")

# Step 2: Find best threshold on train split — no test leakage
print("\nSweeping thresholds on train split...")
best_train_metrics = sweep_threshold(train_results)
best_threshold     = best_train_metrics["threshold"]
print(f"Best threshold from train: {best_threshold}  (F1={best_train_metrics['f1']})")

# Step 3: Score test split
test_results = run_split("test")

# Step 4: Apply train-selected threshold to test
print(f"\nEvaluating test split at train-selected threshold ({best_threshold})...")
test_metrics = compute_metrics(test_results, best_threshold)

# Add thresholded predictions to test results for error analysis
for r in test_results:
    if r["consistency_score"] is not None:
        r["predicted_hallucination"] = bool(r["consistency_score"] < best_threshold)
    else:
        r["predicted_hallucination"] = None

with open(f"{OUTPUT_DIR}/consistency_results_test.json", "w") as f:
    json.dump(test_results, f, indent=2)

# ─── Save metrics ─────────────────────────────────────────────────────────────
metrics_output = {
    "train_threshold_sweep" : best_train_metrics,
    "best_threshold"        : best_threshold,
    "test_metrics"          : test_metrics,
}
with open(f"{OUTPUT_DIR}/consistency_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

# ─── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"Best threshold (from train) : {best_threshold}")
print(f"Test F1                     : {test_metrics['f1']}")
print(f"Test Precision              : {test_metrics['precision']}")
print(f"Test Recall                 : {test_metrics['recall']}")
print(f"Test AUROC                  : {test_metrics['auroc']}")
print(f"Test Accuracy               : {test_metrics['accuracy']}")
print(f"Confusion matrix            : {test_metrics['confusion_matrix']}")
print(f"{'='*50}")
print("All done! Files saved to /workspace:")
print("  consistency_results_train.json")
print("  consistency_results_test.json")
print("  consistency_metrics.json")
