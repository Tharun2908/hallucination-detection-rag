"""
relevance_verifier_full_v2.py
Signal 2: Semantic support / relevance-based hallucination detection on full RAGTruth dataset

Corrected methodology:
- Split context into sentences
- Split answer into sentences
- For each answer sentence, score semantic relevance against every context sentence
- Keep the BEST relevance score for that answer sentence
- Aggregate best relevance scores across answer sentences

Main score used for evaluation:
- mean relevance across answer sentences

Also stored for analysis:
- min relevance across answer sentences
- per-sentence best relevance scores

Interpretation:
- High score = answer sentences are semantically grounded in context
- Low score  = one or more answer sentences are weakly supported / off-context

Methodology:
- Threshold tuned on train split, applied to test (no leakage)
- Raw logits aggregated before normalization
- Min-max normalization uses train split statistics only
- Sentence-level answer-to-context semantic support is more appropriate
  than query-based relevance for hallucination detection

Limitation:
- Relevance / semantic support != entailment
- A semantically related answer can still hallucinate unsupported details
- Regex sentence splitting is lightweight and imperfect

Model  : cross-encoder/ms-marco-MiniLM-L-6-v2
Dataset: wandb/RAGTruth-processed (train + test splits)
Output : /workspace/relevance_results_train_v2.json
         /workspace/relevance_results_test_v2.json
         /workspace/relevance_metrics_v2.json
"""

import json
import os
import re
import random
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import CrossEncoder
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
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = "/workspace"
DEBUG = False
DEBUG_SIZE = 10
CHECKPOINT_EVERY = 50

# Main score used for evaluation:
# "mean" = average semantic support over answer sentences
# "min"  = weakest-supported answer sentence
AGGREGATION_MODE = "mean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading relevance model...")
relevance_model = CrossEncoder(MODEL_NAME)
print("Relevance model loaded.")


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


def compute_relevance_score(context: str, answer: str) -> dict:
    """
    Corrected sentence-level semantic support scoring.

    Method:
    - Split context into sentences
    - Split answer into sentences
    - For each answer sentence, score it against all context sentences
    - Keep the BEST relevance score for that answer sentence
    - Aggregate best relevance scores across answer sentences

    Returns:
    - mean_relevance: average best support over answer sentences
    - min_relevance : weakest-supported answer sentence
    - per_sentence_scores: best relevance score per answer sentence
    - n_answer_sentences
    - n_context_sentences
    """
    context_sentences = split_into_sentences(context)
    answer_sentences = split_into_sentences(answer)

    if not context_sentences or not answer_sentences:
        return {
            "mean_relevance": 0.0,
            "min_relevance": 0.0,
            "per_sentence_scores": [],
            "n_answer_sentences": len(answer_sentences),
            "n_context_sentences": len(context_sentences),
        }

    best_scores = []

    for answer_sent in answer_sentences:
        # Treat answer sentence as query-like text, context sentence as passage-like text
        pairs = [(answer_sent, context_sent) for context_sent in context_sentences]
        scores = relevance_model.predict(pairs)

        best_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        best_scores.append(best_score)

    return {
        "mean_relevance": float(np.mean(best_scores)),
        "min_relevance": float(np.min(best_scores)),
        "per_sentence_scores": [round(s, 4) for s in best_scores],
        "n_answer_sentences": len(answer_sentences),
        "n_context_sentences": len(context_sentences),
    }


def get_main_score(score_dict: dict) -> float:
    """Choose the main evaluation score based on aggregation mode."""
    if AGGREGATION_MODE == "mean":
        return score_dict["mean_relevance"]
    elif AGGREGATION_MODE == "min":
        return score_dict["min_relevance"]
    else:
        raise ValueError(f"Unsupported AGGREGATION_MODE: {AGGREGATION_MODE}")


def minmax_normalize(scores: list[float], train_min: float, train_max: float) -> list[float]:
    """
    Normalize scores using train split min/max.
    Clipped to [0, 1] to handle test scores outside train range.
    Same train_min/train_max applied to both train and test — no leakage.
    """
    rng = train_max - train_min
    if rng == 0:
        return [0.5] * len(scores)

    return [
        float(max(0.0, min(1.0, (s - train_min) / rng)))
        for s in scores
    ]


def compute_metrics(results: list[dict], threshold: float) -> dict:
    """Compute evaluation metrics at a given threshold on normalized scores."""
    valid = [r for r in results if r["relevance_score"] is not None]
    y_true = [r["ground_truth_hallucination"] for r in valid]
    y_pred = [r["relevance_score"] < threshold for r in valid]
    y_scores = [1 - r["relevance_score"] for r in valid]

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
    """Score all examples in a dataset split. Returns results with raw scores."""
    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    dataset = load_dataset("wandb/RAGTruth-processed", split=split)
    if DEBUG:
        dataset = dataset.select(range(DEBUG_SIZE))
        print(f"DEBUG MODE: {DEBUG_SIZE} examples only")

    output_file = f"{OUTPUT_DIR}/relevance_results_{split}_v2.json"

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
            scores = compute_relevance_score(context, answer)
            raw_score = get_main_score(scores)

            print(
                f"raw={raw_score:.3f} | "
                f"mean={scores['mean_relevance']:.3f} | "
                f"min={scores['min_relevance']:.3f} | "
                f"ans_sents={scores['n_answer_sentences']} | "
                f"ctx_sents={scores['n_context_sentences']} | "
                f"gt={'HALL' if ground_truth else 'FAITH'}"
            )

            results.append({
                "idx": idx,
                "query": query[:100],
                "raw_relevance_score": round(raw_score, 4),
                "raw_mean_relevance": round(scores["mean_relevance"], 4),
                "raw_min_relevance": round(scores["min_relevance"], 4),
                "relevance_sentence_scores": scores["per_sentence_scores"],
                "relevance_score": None,  # filled after normalization
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
                "raw_relevance_score": None,
                "raw_mean_relevance": None,
                "raw_min_relevance": None,
                "relevance_sentence_scores": None,
                "relevance_score": None,
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

    # Final save of raw scores
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved → {output_file}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
print(f"\nRunning Signal 2 with AGGREGATION_MODE = '{AGGREGATION_MODE}'")

# Step 1: Score train split (raw scores only)
train_results = run_split("train")

# Step 2: Compute train min/max for normalization — no test leakage
train_raw = [
    r["raw_relevance_score"]
    for r in train_results
    if r["raw_relevance_score"] is not None
]
train_min = float(min(train_raw))
train_max = float(max(train_raw))
print(f"\nTrain raw score range: min={train_min:.3f}, max={train_max:.3f}")

# Step 3: Normalize train scores using train min/max
train_norm = minmax_normalize(train_raw, train_min, train_max)
norm_iter = iter(train_norm)
for r in train_results:
    if r["raw_relevance_score"] is not None:
        r["relevance_score"] = round(next(norm_iter), 4)

with open(f"{OUTPUT_DIR}/relevance_results_train_v2.json", "w") as f:
    json.dump(train_results, f, indent=2)
print("Train scores normalized and saved.")

# Step 4: Find best threshold on normalized train scores
print("\nSweeping thresholds on train split...")
best_train_metrics = sweep_threshold(train_results)
best_threshold = best_train_metrics["threshold"]
print(f"Best threshold: {best_threshold}  (F1={best_train_metrics['f1']})")

# Step 5: Score test split (raw scores only)
test_results = run_split("test")

# Step 6: Normalize test using TRAIN min/max — no leakage
test_raw = [
    r["raw_relevance_score"]
    for r in test_results
    if r["raw_relevance_score"] is not None
]
test_norm = minmax_normalize(test_raw, train_min, train_max)
norm_iter = iter(test_norm)
for r in test_results:
    if r["raw_relevance_score"] is not None:
        r["relevance_score"] = round(next(norm_iter), 4)

# Step 7: Apply train threshold to normalized test scores
print(f"\nEvaluating test at threshold {best_threshold}...")
test_metrics = compute_metrics(test_results, best_threshold)

# Add predictions to test results
for r in test_results:
    if r["relevance_score"] is not None:
        r["predicted_hallucination"] = bool(r["relevance_score"] < best_threshold)
    else:
        r["predicted_hallucination"] = None

with open(f"{OUTPUT_DIR}/relevance_results_test_v2.json", "w") as f:
    json.dump(test_results, f, indent=2)

# ─── Save metrics ─────────────────────────────────────────────────────────────
metrics_output = {
    "aggregation_mode": AGGREGATION_MODE,
    "train_min": train_min,
    "train_max": train_max,
    "train_threshold_sweep": best_train_metrics,
    "best_threshold": best_threshold,
    "test_metrics": test_metrics,
}
with open(f"{OUTPUT_DIR}/relevance_metrics_v2.json", "w") as f:
    json.dump(metrics_output, f, indent=2)

# ─── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 2 (Relevance, corrected)")
print(f"{'='*50}")
print(f"Aggregation mode            : {AGGREGATION_MODE}")
print(f"Train raw range             : [{train_min:.3f}, {train_max:.3f}]")
print(f"Best threshold (from train) : {best_threshold}")
print(f"Test F1                     : {test_metrics['f1']}")
print(f"Test Precision              : {test_metrics['precision']}")
print(f"Test Recall                 : {test_metrics['recall']}")
print(f"Test AUROC                  : {test_metrics['auroc']}")
print(f"Test Accuracy               : {test_metrics['accuracy']}")
print(f"Confusion matrix            : {test_metrics['confusion_matrix']}")
print(f"{'='*50}")
print("All done! Files saved to /workspace:")
print("  relevance_results_train_v2.json")
print("  relevance_results_test_v2.json")
print("  relevance_metrics_v2.json")
