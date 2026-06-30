"""
signal5_bertscore.py
Signal 5: BERTScore-based hallucination detection on RAGTruth

Methodology:
- Split answer into sentences, split context into sentences
- For each answer sentence, compute BERTScore recall against every context sentence
- Keep MAX recall score for that answer sentence (best context support)
- Aggregate across answer sentences: mean (main) and min (also stored)

Score interpretation:
- High recall = answer sentence well covered by some context sentence = faithful
- Low recall  = answer sentence not covered = hallucination

Main score: mean of per-sentence max recall
Score direction: low = hallucination → invert as 1 - score for AUROC

Model: roberta-large (default for BERTScore)
Output: /workspace/signal5_results_train.json
        /workspace/signal5_results_test.json
        /workspace/signal5_metrics.json
"""

import json
import os
import re
import numpy as np
import torch
from datasets import load_dataset
from bert_score import BERTScorer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

OUTPUT_DIR = "/workspace"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGGREGATION_MODE = "mean"

print(f"Device: {DEVICE}", flush=True)
scorer = BERTScorer(lang="en", device=DEVICE, rescale_with_baseline=False)

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0

def split_into_sentences(text):
    if not text or not isinstance(text, str):
        return []
    text = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]

def compute_bertscore_signal(answer, context):
    """
    For each answer sentence, compute BERTScore recall against all context sentences.
    Keep max recall per answer sentence.
    Aggregate: mean and min across answer sentences.
    """
    answer_sents  = split_into_sentences(answer)
    context_sents = split_into_sentences(context)

    if not answer_sents or not context_sents:
        return {
            "mean_recall": 0.0,
            "min_recall":  0.0,
            "per_sentence_scores": [],
            "n_answer_sentences":  len(answer_sents),
            "n_context_sentences": len(context_sents),
        }

    best_scores = []

    for ans_sent in answer_sents:
        # Compare this answer sentence against all context sentences
        candidates = [ans_sent] * len(context_sents)
        references = context_sents

        _, R, _ = scorer.score(candidates, references, verbose=False, batch_size=min(32, len(context_sents)))

        max_recall = float(R.max().item())
        best_scores.append(max_recall)

    return {
        "mean_recall":           round(float(np.mean(best_scores)), 4),
        "min_recall":            round(float(np.min(best_scores)), 4),
        "per_sentence_scores":   [round(s, 4) for s in best_scores],
        "n_answer_sentences":    len(answer_sents),
        "n_context_sentences":   len(context_sents),
    }

def get_main_score(score_dict):
    if AGGREGATION_MODE == "mean":
        return score_dict["mean_recall"]
    elif AGGREGATION_MODE == "min":
        return score_dict["min_recall"]

def compute_metrics(results, threshold):
    valid    = [r for r in results if r["signal5_score"] is not None]
    y_true   = [r["ground_truth_hallucination"] for r in valid]
    y_pred   = [r["signal5_score"] < threshold for r in valid]
    y_scores = [1 - r["signal5_score"] for r in valid]
    return {
        "threshold":        threshold,
        "f1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "auroc":            round(roc_auc_score(y_true, y_scores), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def sweep_threshold(results):
    best = {"f1": -1}
    for t in [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]:
        m = compute_metrics(results, t)
        if m["f1"] > best["f1"]:
            best = m
    return best

def run_split(split):
    print(f"\n{'='*60}")
    print(f"Processing split: {split}", flush=True)
    print(f"{'='*60}")

    dataset = load_dataset("wandb/RAGTruth-processed", split=split)
    output_file = f"{OUTPUT_DIR}/signal5_results_{split}_{AGGREGATION_MODE}.json"

    done_indices = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        done_indices = {r["idx"] for r in results}
        print(f"Resuming: {len(results)} examples already done.", flush=True)
    else:
        results = []

    print(f"Loaded {len(dataset)} examples.", flush=True)

    for idx, example in enumerate(dataset):
        if idx in done_indices:
            continue

        answer  = example["output"]
        context = example["context"]
        label   = is_hallucinated(example)

        print(f"[{idx+1}/{len(dataset)}] Processing...", end=" ", flush=True)

        try:
            scores = compute_bertscore_signal(answer, context)
            main_score = get_main_score(scores)

            print(
                f"score={main_score:.3f} | "
                f"mean={scores['mean_recall']:.3f} | "
                f"min={scores['min_recall']:.3f} | "
                f"ans_sents={scores['n_answer_sentences']} | "
                f"ctx_sents={scores['n_context_sentences']} | "
                f"gt={'HALL' if label else 'FAITH'}",
                flush=True
            )

            results.append({
                "idx":                        idx,
                "signal5_score":              main_score,
                "bertscore_mean_recall":      scores["mean_recall"],
                "bertscore_min_recall":       scores["min_recall"],
                "bertscore_sentence_scores":  scores["per_sentence_scores"],
                "n_answer_sentences":         scores["n_answer_sentences"],
                "n_context_sentences":        scores["n_context_sentences"],
                "ground_truth_hallucination": bool(label),
                "model":                      example.get("model", "unknown"),
                "task_type":                  example.get("task_type", "unknown"),
            })

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results.append({
                "idx":                        idx,
                "signal5_score":              None,
                "bertscore_mean_recall":      None,
                "bertscore_min_recall":       None,
                "bertscore_sentence_scores":  None,
                "n_answer_sentences":         None,
                "n_context_sentences":        None,
                "ground_truth_hallucination": bool(label),
                "model":                      example.get("model", "unknown"),
                "task_type":                  example.get("task_type", "unknown"),
                "error":                      str(e),
            })

        if (idx + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint saved ({idx+1}/{len(dataset)})", flush=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {output_file}", flush=True)

    return results

# --- Main ---
print(f"Starting Signal 5 (BERTScore) | AGGREGATION_MODE={AGGREGATION_MODE}", flush=True)

train_results = run_split("train")

print("\nSweeping thresholds on train...", flush=True)
best_train = sweep_threshold(train_results)
best_threshold = best_train["threshold"]
print(f"Best threshold: {best_threshold} (F1={best_train['f1']})", flush=True)

test_results = run_split("test")
test_metrics = compute_metrics(test_results, best_threshold)

print(f"\n{'='*50}")
print("FINAL RESULTS — Signal 5 (BERTScore)")
print(f"{'='*50}")
print(f"Aggregation mode : {AGGREGATION_MODE}")
print(f"F1:        {test_metrics['f1']}")
print(f"Precision: {test_metrics['precision']}")
print(f"Recall:    {test_metrics['recall']}")
print(f"AUROC:     {test_metrics['auroc']}")
print(f"Threshold: {best_threshold}")
print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

with open(f"{OUTPUT_DIR}/signal5_metrics_{AGGREGATION_MODE}.json", "w") as f:
    json.dump({
        "aggregation_mode": AGGREGATION_MODE,
        "best_threshold":   best_threshold,
        "train_metrics":    best_train,
        "test_metrics":     test_metrics,
    }, f, indent=2)

print("\nSaved:")
print("  /workspace/signal5_results_train.json")
print("  /workspace/signal5_results_test.json")
print("  /workspace/signal5_metrics.json")
