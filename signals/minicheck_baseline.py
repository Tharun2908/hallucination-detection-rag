"""
minicheck_baseline.py
External Baseline: MiniCheck (deberta-v3-large) hallucination detection on RAGTruth

Methodology:
- Uses MiniCheck deberta-v3-large as a pretrained faithfulness model
- Input: doc=context, claim=answer
- Output: support probability (high = supported = faithful)
- Positioned as external SOTA baseline, not part of fusion system

Output: /workspace/minicheck_results_test.json
        /workspace/minicheck_results_train.json
        /workspace/minicheck_metrics_7b.json
"""

import json
import os
import numpy as np
from datasets import load_dataset
from minicheck.minicheck import MiniCheck
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

OUTPUT_DIR  = "/workspace"
MODEL_NAME  = "Bespoke-MiniCheck-7B"
BATCH_SIZE  = 16
CHECKPOINT_EVERY = 50

def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0

def compute_metrics(results, threshold):
    valid    = [r for r in results if r["minicheck_score"] is not None]
    y_true   = [r["ground_truth_hallucination"] for r in valid]
    y_pred   = [r["minicheck_score"] < threshold for r in valid]
    y_scores = [1 - r["minicheck_score"] for r in valid]
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

def run_split(split, scorer):
    print(f"\n{'='*60}")
    print(f"Processing split: {split}", flush=True)
    print(f"{'='*60}")

    dataset = load_dataset("wandb/RAGTruth-processed", split=split)
    output_file = f"{OUTPUT_DIR}/minicheck_results_{split}_7b.json"

    done_indices = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        done_indices = {r["idx"] for r in results}
        print(f"Resuming: {len(results)} examples already done.", flush=True)
    else:
        results = []

    print(f"Loaded {len(dataset)} examples.", flush=True)

    pending = []
    for idx, example in enumerate(dataset):
        if idx in done_indices:
            continue
        pending.append({
            "idx":       idx,
            "answer":    example["output"],
            "context":   example["context"],
            "label":     is_hallucinated(example),
            "model":     example.get("model", "unknown"),
            "task_type": example.get("task_type", "unknown"),
        })

    print(f"Pending: {len(pending)} examples", flush=True)

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        docs   = [ex["context"] for ex in batch]
        claims = [ex["answer"]  for ex in batch]

        try:
            pred_labels, max_support_probs, _, _ = scorer.score(docs=docs, claims=claims)

            for i, ex in enumerate(batch):
                results.append({
                    "idx":                        ex["idx"],
                    "minicheck_score":            round(float(max_support_probs[i]), 4),
                    "minicheck_label":            int(pred_labels[i]),
                    "ground_truth_hallucination": bool(ex["label"]),
                    "model":                      ex["model"],
                    "task_type":                  ex["task_type"],
                })

        except Exception as e:
            print(f"ERROR at batch {batch_start}: {e}", flush=True)
            for ex in batch:
                results.append({
                    "idx":                        ex["idx"],
                    "minicheck_score":            None,
                    "minicheck_label":            None,
                    "ground_truth_hallucination": bool(ex["label"]),
                    "model":                      ex["model"],
                    "task_type":                  ex["task_type"],
                    "error":                      str(e),
                })

        done = min(batch_start + BATCH_SIZE, len(pending))
        print(f"  Processed {done}/{len(pending)}", flush=True)

        if (batch_start // BATCH_SIZE + 1) % (CHECKPOINT_EVERY // BATCH_SIZE) == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint saved", flush=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {output_file}", flush=True)

    return results

if __name__ == "__main__":
    # --- Main ---
    print(f"Loading MiniCheck ({MODEL_NAME})...", flush=True)
    scorer = MiniCheck(model_name=MODEL_NAME, batch_size=BATCH_SIZE)
    print("MiniCheck loaded.", flush=True)

    train_results = run_split("train", scorer)

    print("\nSweeping thresholds on train...", flush=True)
    best_train = sweep_threshold(train_results)
    best_threshold = best_train["threshold"]
    print(f"Best threshold: {best_threshold} (F1={best_train['f1']})", flush=True)

    test_results = run_split("test", scorer)
    test_metrics = compute_metrics(test_results, best_threshold)

    print(f"\n{'='*50}")
    print("FINAL RESULTS — MiniCheck Baseline")
    print(f"{'='*50}")
    print(f"Model:     {MODEL_NAME}")
    print(f"F1:        {test_metrics['f1']}")
    print(f"Precision: {test_metrics['precision']}")
    print(f"Recall:    {test_metrics['recall']}")
    print(f"AUROC:     {test_metrics['auroc']}")
    print(f"Threshold: {best_threshold}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    with open(f"{OUTPUT_DIR}/minicheck_metrics_7b.json", "w") as f:
        json.dump({
            "model":           MODEL_NAME,
            "best_threshold":  best_threshold,
            "train_metrics":   best_train,
            "test_metrics":    test_metrics,
        }, f, indent=2)

    print("\nSaved:")
    print("  /workspace/minicheck_results_train.json")
    print("  /workspace/minicheck_results_test.json")
    print("  /workspace/minicheck_metrics_7b.json")
