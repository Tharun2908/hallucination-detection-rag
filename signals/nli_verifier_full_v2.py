"""
nli_verifier_full_v2.py
Signal 1: NLI-based hallucination detection on full RAGTruth dataset

Corrected methodology:
- Split context into sentences
- Split answer into sentences
- For each answer sentence, run NLI against every context sentence
- Keep the BEST entailment score for that answer sentence
- Aggregate best entailment scores across answer sentences

Main score used for evaluation:
- mean entailment across answer sentences

Also stored for analysis:
- min entailment across answer sentences
- per-sentence best entailment scores

Interpretation:
- High score = answer sentences are supported by context
- Low score  = one or more answer sentences are not supported

Methodology:
- Threshold tuned on train split, applied to test (no leakage)
- Sentence-level answer-to-context support is more appropriate than
  context-sentence -> full-answer scoring
- Label order read from model config (not hardcoded)
- Checkpoint every 50 examples → safe to kill and resume

Limitation:
- Sentence-level entailment is still a heuristic proxy, not full logical verification
- Multi-hop / compositional evidence across multiple context sentences
  may still be under-captured
- Regex sentence splitting is lightweight and imperfect

Model  : cross-encoder/nli-deberta-v3-base
Dataset: wandb/RAGTruth-processed (train + test splits)
Output : /workspace/nli_results_train_v2.json
         /workspace/nli_results_test_v2.json
         /workspace/nli_metrics_v2.json
"""

import json
import os
import re
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
OUTPUT_DIR = "/workspace"
DEBUG = False
DEBUG_SIZE = 10
CHECKPOINT_EVERY = 50

# Main score used for evaluation:
# "mean" = average support over answer sentences
# "min"  = weakest-supported answer sentence
AGGREGATION_MODE = "mean"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load model ───────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("Loading NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nli_model = nli_model.to(DEVICE)
nli_model.eval()
print("NLI model loaded.")

ID2LABEL = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
print("Model labels:", ID2LABEL)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_hallucinated(example: dict) -> bool:
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0


def split_into_sentences(text: str) -> list[str]:
    """
    Lightweight regex-based sentence splitting.
    Acceptable for thesis use, though imperfect for abbreviations, decimals, etc.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def check_nli_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """
    Run NLI on a batch of (premise, hypothesis) pairs.
    Returns list of dicts with probabilities for each label.

    Each pair:
    premise    = context sentence
    hypothesis = answer sentence
    """
    if not pairs:
        return []

    premises = [p for p, h in pairs]
    hypotheses = [h for p, h in pairs]

    inputs = nli_tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = nli_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

    results = []
    for row in probs:
        result = {
            ID2LABEL[i]: float(row[i])
            for i in range(len(row))
        }
        results.append(result)

    return results


def compute_nli_score(context: str, answer: str) -> dict:
    """
    Corrected sentence-level NLI support scoring.

    Method:
    - Split context into sentences
    - Split answer into sentences
    - For each answer sentence, compare against all context sentences
    - Keep the BEST entailment score for that answer sentence
    - Aggregate best entailment scores across answer sentences

    Returns:
    - mean_entailment: average best support over answer sentences
    - min_entailment : weakest-supported answer sentence
    - per_sentence_scores: best entailment score per answer sentence
    - n_answer_sentences
    - n_context_sentences
    """
    context_sentences = split_into_sentences(context)
    answer_sentences = split_into_sentences(answer)

    if not context_sentences or not answer_sentences:
        return {
            "mean_entailment": 0.0,
            "min_entailment": 0.0,
            "per_sentence_scores": [],
            "n_answer_sentences": len(answer_sentences),
            "n_context_sentences": len(context_sentences),
        }

    best_scores = []

    for answer_sent in answer_sentences:
        pairs = [(context_sent, answer_sent) for context_sent in context_sentences]
        nli_results = check_nli_batch(pairs)

        entailment_scores = [
            r.get("entailment", 0.0)
            for r in nli_results
        ]

        best_entailment = max(entailment_scores) if entailment_scores else 0.0
        best_scores.append(float(best_entailment))

    return {
        "mean_entailment": float(np.mean(best_scores)),
        "min_entailment": float(np.min(best_scores)),
        "per_sentence_scores": [round(s, 4) for s in best_scores],
        "n_answer_sentences": len(answer_sentences),
        "n_context_sentences": len(context_sentences),
    }


def get_main_score(score_dict: dict) -> float:
    """Choose the main evaluation score based on aggregation mode."""
    if AGGREGATION_MODE == "mean":
        return score_dict["mean_entailment"]
    elif AGGREGATION_MODE == "min":
        return score_dict["min_entailment"]
    else:
        raise ValueError(f"Unsupported AGGREGATION_MODE: {AGGREGATION_MODE}")


def compute_metrics(results: list[dict], threshold: float) -> dict:
    """Compute evaluation metrics at a given threshold."""
    valid = [r for r in results if r["nli_score"] is not None]
    y_true = [r["ground_truth_hallucination"] for r in valid]
    y_pred = [r["nli_score"] < threshold for r in valid]
    y_scores = [1 - r["nli_score"] for r in valid]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
        "confusion_matrix": cm,
        "n_examples": len(valid),
        "n_errors": len(results) - len(valid),
    }


def sweep_threshold(results: list[dict]) -> dict:
    """Sweep thresholds 0.10 → 0.90, return metrics at best F1."""
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]
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

    dataset = load_dataset("wandb/RAGTruth-processed", split=split)
    if DEBUG:
        dataset = dataset.select(range(DEBUG_SIZE))
        print(f"DEBUG MODE: {DEBUG_SIZE} examples only")

    output_file = f"{OUTPUT_DIR}/nli_results_{split}_v2.json"

    # ── Resume from checkpoint ────────────────────────────────────────────────
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

        query = example["query"]
        context = example["context"]
        answer = example["output"]
        ground_truth = is_hallucinated(example)

        print(f"[{idx+1}/{len(dataset)}] Processing... ", end="", flush=True)

        try:
            scores = compute_nli_score(context, answer)
            score = get_main_score(scores)

            print(
                f"score={score:.3f} | "
                f"mean={scores['mean_entailment']:.3f} | "
                f"min={scores['min_entailment']:.3f} | "
                f"ans_sents={scores['n_answer_sentences']} | "
                f"ctx_sents={scores['n_context_sentences']} | "
                f"gt={'HALL' if ground_truth else 'FAITH'}"
            )

            results.append({
                "idx": idx,
                "query": query[:100],
                "nli_score": round(score, 4),
                "nli_mean_entailment": round(scores["mean_entailment"], 4),
                "nli_min_entailment": round(scores["min_entailment"], 4),
                "nli_sentence_scores": scores["per_sentence_scores"],
                "n_answer_sentences": scores["n_answer_sentences"],
                "n_context_sentences": scores["n_context_sentences"],
                "ground_truth_hallucination": ground_truth,
                "model": example.get("model", "unknown"),
                "task_type": example.get("task_type", "unknown"),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "idx": idx,
                "query": query[:100],
                "nli_score": None,
                "nli_mean_entailment": None,
                "nli_min_entailment": None,
                "nli_sentence_scores": None,
                "n_answer_sentences": None,
                "n_context_sentences": None,
                "ground_truth_hallucination": ground_truth,
                "model": example.get("model", "unknown"),
                "task_type": example.get("task_type", "unknown"),
                "error": str(e),
            })

        # ── Checkpoint every N examples ───────────────────────────────────────
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint saved ({idx+1}/{len(dataset)})")

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_file}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
print(f"\nRunning Signal 1 with AGGREGATION_MODE = '{AGGREGATION_MODE}'")

# Step 1: Score train split
train_results = run_split("train")

# Step 2: Find best threshold on train — no test leakage
print("\nSweeping thresholds on train split...")
best_train_metrics = sweep_threshold(train_results)
best_threshold = best_train_metrics["threshold"]
print(f"Best threshold: {best_threshold}  (F1={best_train_metrics['f1']})")

# Step 3: Score test split
test_results = run_split("test")

# Step 4: Apply train threshold to test
print(f"\nEvaluating test at threshold {best_threshold}...")
test_metrics = compute_metrics(test_results, best_threshold)

# Add predictions to test results
for r in test_results:
    if r["nli_score"] is not None:
        r["predicted_hallucination"] = bool(r["nli_score"] < best_threshold)
    else:
        r["predicted_hallucination"] = None

with open(f"{OUTPUT_DIR}/nli_results_test_v2.json", "w") as f:
    json.dump(test_results, f, indent=2)

# ─── Save metrics ─────────────────────────────────────────────────────────────
metrics_output = {
    "aggregation_mode": AGGREGATION_MODE,
    "train_threshold_sweep": best_train_metrics,
    "best_threshold": best_threshold,
    "test_metrics": test_metrics,
}
with open(f"{OUTPUT_DIR}/nli_metrics_v2.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

# ─── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 1 (NLI, corrected)")
print(f"{'='*50}")
print(f"Aggregation mode            : {AGGREGATION_MODE}")
print(f"Best threshold (from train) : {best_threshold}")
print(f"Test F1                     : {test_metrics['f1']}")
print(f"Test Precision              : {test_metrics['precision']}")
print(f"Test Recall                 : {test_metrics['recall']}")
print(f"Test AUROC                  : {test_metrics['auroc']}")
print(f"Test Accuracy               : {test_metrics['accuracy']}")
print(f"Confusion matrix            : {test_metrics['confusion_matrix']}")
print(f"{'='*50}")
print("All done! Files saved to /workspace:")
print("  nli_results_train_v2.json")
print("  nli_results_test_v2.json")
print("  nli_metrics_v2.json")
