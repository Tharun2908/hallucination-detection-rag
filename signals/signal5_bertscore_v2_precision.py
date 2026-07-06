"""
signal5_bertscore_v2_precision.py

Signal 5: BERTScore Precision-based hallucination detection on RAGTruth

Corrected methodology:
- Split answer into sentences.
- Split context into sentences.
- For each answer sentence, compute BERTScore PRECISION against every context sentence.
- Keep MAX precision for that answer sentence.
- Aggregate across answer sentences:
    - mean precision = main score by default
    - min precision also stored

Why Precision?
- Candidate = answer sentence
- Reference = context sentence
- BERTScore Precision measures how much of the answer sentence is covered by the context.
- This matches faithfulness / support detection.

Score interpretation:
- High score = answer is supported / faithful
- Low score  = answer is weakly supported / hallucination risk

Metric direction:
- signal5_score = support score
- For hallucination AUROC, use 1 - signal5_score

Outputs:
- /workspace/signal5_v2_precision_results_train_mean.json
- /workspace/signal5_v2_precision_results_test_mean.json
- /workspace/signal5_v2_precision_metrics_mean.json

Smoke test:
    python signal5_bertscore_v2_precision.py --limit 50 --overwrite

Full run:
    python signal5_bertscore_v2_precision.py --overwrite

Optional min aggregation:
    python signal5_bertscore_v2_precision.py --aggregation min --overwrite
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from bert_score import BERTScorer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


# -----------------------------
# General utilities
# -----------------------------

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def flatten_to_text(obj: Any) -> str:
    """
    Safely converts answer/context fields into plain text.

    Handles:
    - string
    - list of strings
    - list of dicts
    - dicts with text/content/context fields
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, (list, tuple)):
        parts = [flatten_to_text(x) for x in obj]
        return " ".join(p for p in parts if p.strip())

    if isinstance(obj, dict):
        preferred_keys = [
            "text",
            "content",
            "context",
            "passage",
            "document",
            "body",
            "sentence",
        ]

        parts = []

        for key in preferred_keys:
            if key in obj:
                parts.append(flatten_to_text(obj[key]))

        for key, value in obj.items():
            if key not in preferred_keys:
                if not isinstance(value, (int, float, bool, type(None))):
                    parts.append(flatten_to_text(value))

        return " ".join(p for p in parts if p.strip())

    return str(obj)


def split_into_sentences(text: Any, min_chars: int = 10) -> List[str]:
    """
    Robust sentence splitter.

    Keeps things simple, but avoids the old issue where non-string context
    became an empty list.
    """
    text = flatten_to_text(text)

    if not text:
        return []

    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_chars]

    return sentences


def is_hallucinated(example: Dict[str, Any]) -> bool:
    """
    RAGTruth binary hallucination label.

    Hallucinated if either:
    - evident_conflict > 0
    - baseless_info > 0
    """
    labels = example["hallucination_labels_processed"]

    return (
        labels.get("evident_conflict", 0) > 0
        or labels.get("baseless_info", 0) > 0
    )


def safe_get(example: Dict[str, Any], key: str, default: str = "unknown") -> str:
    value = example.get(key, default)
    if value is None:
        return default
    return str(value)


# -----------------------------
# Signal 5 computation
# -----------------------------

def compute_bertscore_signal(
    answer: Any,
    context: Any,
    scorer: BERTScorer,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    For each answer sentence:
    - compare it against all context sentences
    - use BERTScore Precision
    - take max precision as best support

    Then aggregate across answer sentences.
    """
    answer_sents = split_into_sentences(answer)
    context_sents = split_into_sentences(context)

    if not answer_sents or not context_sents:
        return {
            "mean_precision": 0.0,
            "min_precision": 0.0,
            "per_sentence_scores": [],
            "best_context_sentence_indices": [],
            "n_answer_sentences": len(answer_sents),
            "n_context_sentences": len(context_sents),
            "empty_reason": (
                "no_answer_sentences"
                if not answer_sents
                else "no_context_sentences"
            ),
        }

    candidates = []
    references = []

    for ans_sent in answer_sents:
        for ctx_sent in context_sents:
            candidates.append(ans_sent)
            references.append(ctx_sent)

    with torch.no_grad():
        P, R, F1 = scorer.score(
            candidates,
            references,
            verbose=False,
            batch_size=batch_size,
        )

    precision_matrix = P.detach().cpu().numpy().reshape(
        len(answer_sents),
        len(context_sents),
    )

    best_precision_per_answer = precision_matrix.max(axis=1)
    best_context_indices = precision_matrix.argmax(axis=1)

    return {
        "mean_precision": round(float(np.mean(best_precision_per_answer)), 4),
        "min_precision": round(float(np.min(best_precision_per_answer)), 4),
        "per_sentence_scores": [
            round(float(x), 4) for x in best_precision_per_answer
        ],
        "best_context_sentence_indices": [
            int(x) for x in best_context_indices
        ],
        "n_answer_sentences": len(answer_sents),
        "n_context_sentences": len(context_sents),
    }


def get_main_score(score_dict: Dict[str, Any], aggregation: str) -> float:
    if aggregation == "mean":
        return score_dict["mean_precision"]

    if aggregation == "min":
        return score_dict["min_precision"]

    raise ValueError(f"Unknown aggregation mode: {aggregation}")


# -----------------------------
# Metrics
# -----------------------------

def get_valid_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []

    for r in results:
        score = r.get("signal5_score")

        if score is None:
            continue

        try:
            score = float(score)
        except Exception:
            continue

        if np.isfinite(score):
            valid.append(r)

    return valid


def compute_metrics(results: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    valid = get_valid_results(results)

    y_true = np.array(
        [int(r["ground_truth_hallucination"]) for r in valid],
        dtype=int,
    )

    support_scores = np.array(
        [float(r["signal5_score"]) for r in valid],
        dtype=float,
    )

    # Low support score means hallucination.
    y_pred = (support_scores < threshold).astype(int)

    # For AUROC, high score should mean hallucination.
    hallucination_scores = 1.0 - support_scores

    metrics = {
        "n": int(len(valid)),
        "threshold": round(float(threshold), 4),
        "positive_rate_hallucination": round(float(y_true.mean()), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "confusion_matrix_order": "[[TN, FP], [FN, TP]]",
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = round(float(roc_auc_score(y_true, hallucination_scores)), 4)
    else:
        metrics["auroc"] = None

    return metrics


def sweep_threshold(
    results: List[Dict[str, Any]],
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
) -> Dict[str, Any]:
    best = None

    thresholds = np.arange(
        threshold_min,
        threshold_max + 1e-9,
        threshold_step,
    )

    for t in thresholds:
        t = round(float(t), 4)
        metrics = compute_metrics(results, threshold=t)

        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return best


# -----------------------------
# Running train/test split
# -----------------------------

def build_output_file(
    output_dir: str,
    split: str,
    aggregation: str,
    limit: Optional[int],
) -> str:
    limit_suffix = "" if limit is None else f"_limit{limit}"
    filename = f"signal5_v2_precision_results_{split}_{aggregation}{limit_suffix}.json"
    return os.path.join(output_dir, filename)


def run_split(
    split: str,
    scorer: BERTScorer,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 70, flush=True)
    print(f"Processing split: {split}", flush=True)
    print("=" * 70, flush=True)

    dataset = load_dataset(args.dataset_name, split=split)

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_file = build_output_file(
        output_dir=args.output_dir,
        split=split,
        aggregation=args.aggregation,
        limit=args.limit,
    )

    results = []
    done_indices = set()

    if os.path.exists(output_file) and not args.overwrite:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        done_indices = {int(r["idx"]) for r in results if "idx" in r}

        print(f"Resuming from: {output_file}", flush=True)
        print(f"Already completed: {len(done_indices)} examples", flush=True)

    elif os.path.exists(output_file) and args.overwrite:
        print(f"Overwrite enabled. Recomputing: {output_file}", flush=True)

    print(f"Loaded {len(dataset)} examples.", flush=True)

    for idx, example in enumerate(dataset):
        if idx in done_indices:
            continue

        answer = example["output"]
        context = example["context"]
        label = is_hallucinated(example)

        print(
            f"[{idx + 1}/{len(dataset)}] Processing...",
            end=" ",
            flush=True,
        )

        try:
            scores = compute_bertscore_signal(
                answer=answer,
                context=context,
                scorer=scorer,
                batch_size=args.batch_size,
            )

            main_score = get_main_score(scores, args.aggregation)

            print(
                f"support_score={main_score:.4f} | "
                f"meanP={scores['mean_precision']:.4f} | "
                f"minP={scores['min_precision']:.4f} | "
                f"ans_sents={scores['n_answer_sentences']} | "
                f"ctx_sents={scores['n_context_sentences']} | "
                f"gt={'HALL' if label else 'FAITH'}",
                flush=True,
            )

            results.append({
                "idx": idx,
                "signal_name": "signal5_bertscore_v2_precision",
                "signal5_score": float(main_score),
                "score_direction": "high_support_low_hallucination",

                "bertscore_mean_precision": scores["mean_precision"],
                "bertscore_min_precision": scores["min_precision"],
                "bertscore_sentence_scores": scores["per_sentence_scores"],
                "best_context_sentence_indices": scores["best_context_sentence_indices"],

                "n_answer_sentences": scores["n_answer_sentences"],
                "n_context_sentences": scores["n_context_sentences"],

                "ground_truth_hallucination": bool(label),
                "model": safe_get(example, "model"),
                "task_type": safe_get(example, "task_type"),
            })

        except Exception as e:
            print(f"ERROR: {e}", flush=True)

            results.append({
                "idx": idx,
                "signal_name": "signal5_bertscore_v2_precision",
                "signal5_score": None,
                "score_direction": "high_support_low_hallucination",

                "bertscore_mean_precision": None,
                "bertscore_min_precision": None,
                "bertscore_sentence_scores": None,
                "best_context_sentence_indices": None,

                "n_answer_sentences": None,
                "n_context_sentences": None,

                "ground_truth_hallucination": bool(label),
                "model": safe_get(example, "model"),
                "task_type": safe_get(example, "task_type"),
                "error": str(e),
            })

        if (idx + 1) % args.checkpoint_every == 0:
            save_json(results, output_file)
            print(
                f"  → Checkpoint saved: {output_file} "
                f"({idx + 1}/{len(dataset)})",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_json(results, output_file)
    print(f"Saved split results → {output_file}", flush=True)

    return results


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wandb/RAGTruth-processed",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "min"],
        default="mean",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="roberta-large",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use for smoke test, e.g. --limit 50",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing v2 result files instead of resuming.",
    )

    parser.add_argument(
        "--threshold_min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--threshold_max",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70, flush=True)
    print("Starting Signal 5 v2 — BERTScore Precision", flush=True)
    print("=" * 70, flush=True)
    print(f"Device       : {device}", flush=True)
    print(f"Model        : {args.model_type}", flush=True)
    print(f"Aggregation  : {args.aggregation}", flush=True)
    print(f"Output dir   : {args.output_dir}", flush=True)
    print(f"Limit        : {args.limit}", flush=True)

    scorer = BERTScorer(
        model_type=args.model_type,
        lang="en",
        device=device,
        rescale_with_baseline=False,
    )

    # 1. Compute train scores
    train_results = run_split(
        split="train",
        scorer=scorer,
        args=args,
    )

    # 2. Select threshold on train
    print("\n" + "=" * 70, flush=True)
    print("Sweeping threshold on train", flush=True)
    print("=" * 70, flush=True)

    best_train = sweep_threshold(
        train_results,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )

    best_threshold = float(best_train["threshold"])

    print(
        f"Best train threshold: {best_threshold:.4f} | "
        f"Train F1={best_train['f1']} | "
        f"Train Precision={best_train['precision']} | "
        f"Train Recall={best_train['recall']} | "
        f"Train AUROC={best_train['auroc']}",
        flush=True,
    )

    # 3. Compute test scores
    test_results = run_split(
        split="test",
        scorer=scorer,
        args=args,
    )

    # 4. Apply train-selected threshold to test
    test_metrics = compute_metrics(
        test_results,
        threshold=best_threshold,
    )

    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS — Signal 5 v2 BERTScore Precision", flush=True)
    print("=" * 70, flush=True)
    print(f"Aggregation mode : {args.aggregation}")
    print(f"Best threshold   : {best_threshold:.4f}")
    print(f"Test F1          : {test_metrics['f1']}")
    print(f"Test Precision   : {test_metrics['precision']}")
    print(f"Test Recall      : {test_metrics['recall']}")
    print(f"Test AUROC       : {test_metrics['auroc']}")
    print(f"Test Confusion   : {test_metrics['confusion_matrix']}")
    print(f"CM order         : {test_metrics['confusion_matrix_order']}")

    # 5. Save metrics
    limit_suffix = "" if args.limit is None else f"_limit{args.limit}"

    metrics_file = os.path.join(
        args.output_dir,
        f"signal5_v2_precision_metrics_{args.aggregation}{limit_suffix}.json",
    )

    train_file = build_output_file(
        output_dir=args.output_dir,
        split="train",
        aggregation=args.aggregation,
        limit=args.limit,
    )

    test_file = build_output_file(
        output_dir=args.output_dir,
        split="test",
        aggregation=args.aggregation,
        limit=args.limit,
    )

    save_json(
        {
            "signal_name": "signal5_bertscore_v2_precision",
            "model_type": args.model_type,
            "aggregation_mode": args.aggregation,
            "score_direction": "high_support_low_hallucination",

            "candidate_reference_direction": {
                "candidate": "answer_sentence",
                "reference": "context_sentence",
                "bertscore_component_used": "precision",
                "reason": (
                    "BERTScore Precision measures how much of the answer sentence "
                    "is covered by the context sentence."
                ),
            },

            "threshold_selection": {
                "method": "train_f1_sweep",
                "best_threshold_from_train": best_threshold,
                "threshold_min": args.threshold_min,
                "threshold_max": args.threshold_max,
                "threshold_step": args.threshold_step,
            },

            "train_metrics_at_best_threshold": best_train,
            "test_metrics": test_metrics,

            "output_files": {
                "train_results": train_file,
                "test_results": test_file,
                "metrics": metrics_file,
            },
        },
        metrics_file,
    )

    print("\nSaved:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    print(f"  {metrics_file}")


if __name__ == "__main__":
    main()
