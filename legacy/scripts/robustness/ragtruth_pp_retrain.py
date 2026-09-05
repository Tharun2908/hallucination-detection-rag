#!/usr/bin/env python
"""
RAGTruth++ retraining experiment for thesis chapter on label-noise robustness.

Question: Does the AUROC gap of supervised signals on RAGTruth++ reflect noisy
original RAGTruth labels (fixable by retraining), or something the model
architecture cannot capture from labels alone?

Three conditions, paired across 5-fold stratified CV on matched RAGTruth++:
  A. Baseline: original S4 (RAGTruth-trained) — no retraining
  B. Retrain on ++ labels: finetune S4 on train fold with RAGTruth++ labels
  C. Retrain on original labels: finetune S4 on train fold with the *original*
     RAGTruth labels for the same examples

Two subsets:
  full       — all matched examples (~408). Exploratory; original S4 may have
               seen some of these during its own training (leakage in Cond. A).
  clean_test — matched examples whose source was the original RAGTruth *test*
               split. No leakage in Cond. A. Smaller sample → noisier folds.

Per fold, per condition:
  - Inner train/val (90/10 stratified) for early stopping (B, C) and
    threshold tuning (A, B, C)
  - Threshold chosen on val (label_pp) to maximize F1, applied to test
  - AUROC/AUPRC are threshold-free; F1/precision/recall are threshold-dependent

Paired comparisons across folds:
  ΔAUROC(B − A), ΔAUROC(C − A), ΔAUROC(B − C), and same for AUPRC.

Usage:
    # Smoke test
    python /workspace/ragtruth_pp_retrain.py --smoke

    # Full run (~30-60 min)
    nohup python /workspace/ragtruth_pp_retrain.py \\
        > /workspace/ragtruth_pp_retrain.log 2>&1 &

Outputs:
    /workspace/ragtruth_pp_retrain/
        matched_data.json
        full/        results.json fold_predictions.json fold_splits.json summary.txt
        clean_test/  results.json fold_predictions.json fold_splits.json summary.txt
"""

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# Config
# =============================================================================
S4_MODEL_DIR = "/workspace/signal4_model"
OUT_DIR = Path("/workspace/ragtruth_pp_retrain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED = 42
MIN_CLEAN_SIZE = 60   # below this, skip clean_test run

# Training hyperparameters — match original S4
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2
VAL_SPLIT = 0.1


# =============================================================================
# Data loading + matching
# =============================================================================
def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0


def load_and_match():
    print("Loading RAGTruth++ from HF...")
    msgs = pd.read_csv("hf://datasets/blue-guardrails/ragtruth-plus-plus/messages.csv")
    spans = pd.read_csv("hf://datasets/blue-guardrails/ragtruth-plus-plus/hallucination_spans.csv")
    print(f"  messages: {len(msgs)} rows")
    print(f"  spans:    {len(spans)} rows")

    assistant = msgs[msgs["role"] == "assistant"].copy()
    print(f"  assistant turns: {len(assistant)}")

    hall_ids = set(spans["message_stable_id"].unique())
    assistant["pp_label"] = assistant["stable_id"].isin(hall_ids).astype(int)

    print("\nLoading original RAGTruth (train + test)...")
    ds_train = load_dataset("wandb/RAGTruth-processed", split="train")
    ds_test = load_dataset("wandb/RAGTruth-processed", split="test")

    # Build text-prefix lookup. Track duplicates explicitly.
    all_keys = []
    text_to_info = {}
    for split_name, ds in [("train", ds_train), ("test", ds_test)]:
        for idx in range(len(ds)):
            ex = ds[idx]
            key = ex["output"][:100]
            all_keys.append(key)
            if key not in text_to_info:
                text_to_info[key] = (split_name, idx, ex)

    dupes = sum(1 for _, c in Counter(all_keys).items() if c > 1)
    print(f"  text lookup: {len(text_to_info)} unique keys "
          f"({dupes} duplicates collapsed — first occurrence wins)")

    # Match
    matched = []
    unmatched = 0
    prefix_match_failures = 0
    for _, row in assistant.iterrows():
        content = str(row["text"]) if pd.notna(row["text"]) else ""
        key = content[:100]
        info = text_to_info.get(key)
        if info is None:
            unmatched += 1
            continue
        split, idx, rt_ex = info
        # Stronger sanity check on a longer prefix
        exact_200 = (content[:200] == rt_ex["output"][:200])
        if not exact_200:
            prefix_match_failures += 1
        matched.append({
            "stable_id": str(row["stable_id"]),
            "rt_split": split,
            "rt_idx": int(idx),
            "answer": rt_ex["output"],
            "context": rt_ex["context"],
            "query": rt_ex.get("query", ""),
            "task_type": rt_ex.get("task_type", ""),
            "model": rt_ex.get("model", ""),
            "label_pp": int(row["pp_label"]),
            "label_orig": int(is_hallucinated(rt_ex)),
            "exact_200_match": bool(exact_200),
            "pp_content_prefix": content[:200],
        })

    n = len(matched)
    print(f"\nMatched: {n} / {len(assistant)}  (unmatched: {unmatched})")
    print(f"  200-char prefix mismatches: {prefix_match_failures} "
          f"(may indicate slight content drift between sources)")
    n_pp_pos = sum(m["label_pp"] for m in matched)
    n_orig_pos = sum(m["label_orig"] for m in matched)
    n_flipped = sum(1 for m in matched if m["label_pp"] != m["label_orig"])
    print(f"  RAGTruth++ pos rate: {n_pp_pos / n:.3f}  ({n_pp_pos}/{n})")
    print(f"  Original   pos rate: {n_orig_pos / n:.3f}  ({n_orig_pos}/{n})")
    print(f"  Flipped labels:      {n_flipped} ({n_flipped / n:.1%})")

    n_from_train = sum(1 for m in matched if m["rt_split"] == "train")
    n_from_test = sum(1 for m in matched if m["rt_split"] == "test")
    print(f"\n  Source split composition:")
    print(f"    matched from original train: {n_from_train}  (S4 LEAKAGE risk in Cond. A)")
    print(f"    matched from original test:  {n_from_test}  (clean for Cond. A)")

    return matched


# =============================================================================
# Scoring + thresholding
# =============================================================================
def score_examples(model, tokenizer, examples, batch_size=16):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            inputs = tokenizer(
                [ex["answer"] for ex in batch],
                [ex["context"] for ex in batch],
                return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
            ).to("cuda")
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return scores


def find_best_threshold(labels, scores):
    """Sweep thresholds on val set; return (best_t, best_f1)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = [int(s >= t) for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def stratified_split(examples, label_field, val_split, rng):
    """Returns (inner_train, inner_val) preserving class balance."""
    pos_idx = [i for i, ex in enumerate(examples) if ex[label_field] == 1]
    neg_idx = [i for i, ex in enumerate(examples) if ex[label_field] == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    val_pos = max(int(len(pos_idx) * val_split), 1)
    val_neg = max(int(len(neg_idx) * val_split), 1)
    val_idx = pos_idx[:val_pos] + neg_idx[:val_neg]
    tr_idx = pos_idx[val_pos:] + neg_idx[val_neg:]
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return [examples[i] for i in tr_idx], [examples[i] for i in val_idx]


# =============================================================================
# Training loop
# =============================================================================
def train_on_fold(inner_train, inner_val, label_field, n_epochs, smoke=False):
    """Finetune from S4 checkpoint. Early stop on val F1 measured against
    label_field (the labels we're training on)."""
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    tokenizer = AutoTokenizer.from_pretrained(S4_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to("cuda")

    n_pos = sum(ex[label_field] for ex in inner_train)
    n_neg = len(inner_train) - n_pos
    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)
    w_neg = len(inner_train) / (2 * n_neg)
    w_pos = len(inner_train) / (2 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    n_steps = max((len(inner_train) // BATCH_SIZE + 1) * n_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    rng = np.random.RandomState(SEED)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        order = np.arange(len(inner_train))
        rng.shuffle(order)
        train_shuffled = [inner_train[i] for i in order]

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_shuffled), BATCH_SIZE):
            batch = train_shuffled[i:i + BATCH_SIZE]
            inputs = tokenizer(
                [ex["answer"] for ex in batch],
                [ex["context"] for ex in batch],
                return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
            ).to("cuda")
            labels_t = torch.tensor([ex[label_field] for ex in batch]).to("cuda")

            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            if smoke and n_batches >= 5:
                break

        val_scores = score_examples(model, tokenizer, inner_val, batch_size=16)
        val_preds = [int(s >= 0.5) for s in val_scores]
        val_labels = [ex[label_field] for ex in inner_val]
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"      epoch {epoch + 1}/{n_epochs}  loss={epoch_loss / max(n_batches, 1):.4f}  "
              f"val_f1@0.5={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"      early stop")
                break

        if smoke:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer


# =============================================================================
# Metrics + paired deltas
# =============================================================================
def compute_metrics(labels, scores, threshold):
    preds = [int(s >= threshold) for s in scores]
    out = {
        "n": len(labels),
        "pos_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if len(set(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        out["brier"] = float(brier_score_loss(labels, scores))
    else:
        out["auroc"] = None
        out["auprc"] = None
        out["brier"] = None
    return out


def aggregate(per_fold):
    keys = [k for k in per_fold[0] if isinstance(per_fold[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [f[k] for f in per_fold if f.get(k) is not None]
        if not vals:
            agg[k] = None
            continue
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        }
    return agg


def paired_deltas(per_fold_a, per_fold_b, metric):
    """Per-fold (b - a) for a metric, plus mean/std."""
    a_vals = [f.get(metric) for f in per_fold_a]
    b_vals = [f.get(metric) for f in per_fold_b]
    deltas = [b - a for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    if not deltas:
        return None
    return {
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
        "values": [float(d) for d in deltas],
    }


# =============================================================================
# Run experiment on a subset
# =============================================================================
def run_experiment(examples, name, n_epochs, smoke=False):
    print("\n" + "#" * 70)
    print(f"# Subset: {name}  (n={len(examples)})")
    print("#" * 70)

    sub_dir = OUT_DIR / name
    sub_dir.mkdir(parents=True, exist_ok=True)

    n_from_train = sum(1 for ex in examples if ex["rt_split"] == "train")
    n_from_test = sum(1 for ex in examples if ex["rt_split"] == "test")
    pp_labels_global = np.array([ex["label_pp"] for ex in examples])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Pre-load A model once
    print("Loading original S4 (Condition A)...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok_A = AutoTokenizer.from_pretrained(S4_MODEL_DIR)
    model_A = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to("cuda").eval()

    fold_results = {"A_baseline": [], "B_pp_retrain": [], "C_orig_retrain": []}
    fold_predictions = []
    fold_splits = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.arange(len(examples)), pp_labels_global)):
        train_set = [examples[i] for i in tr_idx]
        test_set = [examples[i] for i in te_idx]
        print(f"\n=== Fold {fold_i}/{N_FOLDS - 1}: train={len(train_set)} test={len(test_set)} ===")

        # Inner train/val split — stratified on label_pp (the eval target)
        rng = np.random.RandomState(SEED + fold_i)
        inner_tr_pp, inner_val = stratified_split(train_set, "label_pp", VAL_SPLIT, rng)

        fold_splits.append({
            "fold": fold_i,
            "train_ids": [ex["stable_id"] for ex in inner_tr_pp],
            "val_ids": [ex["stable_id"] for ex in inner_val],
            "test_ids": [ex["stable_id"] for ex in test_set],
        })

        val_labels_pp = [ex["label_pp"] for ex in inner_val]
        test_labels_pp = [ex["label_pp"] for ex in test_set]

        # ---- Condition A ----
        print("  [A] baseline")
        val_A = score_examples(model_A, tok_A, inner_val, batch_size=16)
        test_A = score_examples(model_A, tok_A, test_set, batch_size=16)
        thr_A, _ = find_best_threshold(val_labels_pp, val_A)
        m_A = compute_metrics(test_labels_pp, test_A, threshold=thr_A)
        m_A["fold"] = fold_i
        fold_results["A_baseline"].append(m_A)
        print(f"      thr={thr_A:.2f}  F1={m_A['f1']:.4f}  AUROC={m_A['auroc']:.4f}  AUPRC={m_A['auprc']:.4f}")

        # ---- Condition B: retrain on label_pp ----
        # Use same inner_tr_pp / inner_val (already stratified on label_pp)
        print("  [B] retrain on label_pp")
        model_B, tok_B = train_on_fold(inner_tr_pp, inner_val, "label_pp", n_epochs, smoke=smoke)
        val_B = score_examples(model_B, tok_B, inner_val, batch_size=16)
        test_B = score_examples(model_B, tok_B, test_set, batch_size=16)
        thr_B, _ = find_best_threshold(val_labels_pp, val_B)
        m_B = compute_metrics(test_labels_pp, test_B, threshold=thr_B)
        m_B["fold"] = fold_i
        fold_results["B_pp_retrain"].append(m_B)
        print(f"      thr={thr_B:.2f}  F1={m_B['f1']:.4f}  AUROC={m_B['auroc']:.4f}  AUPRC={m_B['auprc']:.4f}")
        del model_B, tok_B
        gc.collect(); torch.cuda.empty_cache()

        # ---- Condition C: retrain on label_orig ----
        # Re-stratify train_set on label_orig for early stopping fairness
        inner_tr_orig, inner_val_orig = stratified_split(
            train_set, "label_orig", VAL_SPLIT, np.random.RandomState(SEED + fold_i + 1000)
        )
        print("  [C] retrain on label_orig")
        model_C, tok_C = train_on_fold(inner_tr_orig, inner_val_orig, "label_orig", n_epochs, smoke=smoke)
        # Evaluate against label_pp as always
        val_C = score_examples(model_C, tok_C, inner_val, batch_size=16)
        test_C = score_examples(model_C, tok_C, test_set, batch_size=16)
        thr_C, _ = find_best_threshold(val_labels_pp, val_C)
        m_C = compute_metrics(test_labels_pp, test_C, threshold=thr_C)
        m_C["fold"] = fold_i
        fold_results["C_orig_retrain"].append(m_C)
        print(f"      thr={thr_C:.2f}  F1={m_C['f1']:.4f}  AUROC={m_C['auroc']:.4f}  AUPRC={m_C['auprc']:.4f}")
        del model_C, tok_C
        gc.collect(); torch.cuda.empty_cache()

        # Per-example predictions
        for j, ex in enumerate(test_set):
            fold_predictions.append({
                "fold": fold_i,
                "stable_id": ex["stable_id"],
                "rt_split": ex["rt_split"],
                "label_pp": ex["label_pp"],
                "label_orig": ex["label_orig"],
                "task_type": ex["task_type"],
                "model": ex["model"],
                "score_A": test_A[j],
                "score_B": test_B[j],
                "score_C": test_C[j],
            })

    del model_A, tok_A
    gc.collect(); torch.cuda.empty_cache()

    # Aggregate + paired deltas
    aggregated = {cond: aggregate(folds) for cond, folds in fold_results.items()}
    paired = {}
    for metric in ["auroc", "auprc", "f1"]:
        paired[f"B_minus_A_{metric}"] = paired_deltas(
            fold_results["A_baseline"], fold_results["B_pp_retrain"], metric)
        paired[f"C_minus_A_{metric}"] = paired_deltas(
            fold_results["A_baseline"], fold_results["C_orig_retrain"], metric)
        paired[f"B_minus_C_{metric}"] = paired_deltas(
            fold_results["C_orig_retrain"], fold_results["B_pp_retrain"], metric)

    results = {
        "name": name,
        "config": {
            "n_examples": len(examples),
            "n_from_train_split": n_from_train,
            "n_from_test_split": n_from_test,
            "n_folds": N_FOLDS,
            "n_epochs": n_epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "patience": PATIENCE,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "smoke": smoke,
        },
        "per_fold": fold_results,
        "aggregated": aggregated,
        "paired_deltas": paired,
    }

    with open(sub_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(sub_dir / "fold_predictions.json", "w") as f:
        json.dump(fold_predictions, f, indent=2)
    with open(sub_dir / "fold_splits.json", "w") as f:
        json.dump(fold_splits, f, indent=2)

    # Pretty summary
    lines = []
    lines.append("=" * 78)
    lines.append(f"RAGTruth++ retraining — subset: {name}")
    lines.append("=" * 78)
    lines.append(f"n_examples={len(examples)}  n_folds={N_FOLDS}  n_epochs={n_epochs}")
    lines.append(f"composition: {n_from_train} from original train, {n_from_test} from original test")
    if name == "full" and n_from_train > 0:
        lines.append("WARNING: Cond. A may be biased upward by S4 having seen original-train examples.")
    lines.append("")
    lines.append(f"{'Condition':<20}{'F1':>18}{'AUROC':>18}{'AUPRC':>18}")
    lines.append("-" * 74)
    for cond in ["A_baseline", "B_pp_retrain", "C_orig_retrain"]:
        agg = aggregated[cond]
        f1 = agg.get("f1") or {}
        au = agg.get("auroc") or {}
        ap = agg.get("auprc") or {}
        f1s = f"{f1['mean']:.4f}\u00b1{f1['std']:.4f}" if f1 else "—"
        aus = f"{au['mean']:.4f}\u00b1{au['std']:.4f}" if au else "—"
        aps = f"{ap['mean']:.4f}\u00b1{ap['std']:.4f}" if ap else "—"
        lines.append(f"{cond:<20}{f1s:>18}{aus:>18}{aps:>18}")

    lines.append("")
    lines.append("Paired deltas (per-fold, mean \u00b1 std):")
    for metric in ["auroc", "auprc", "f1"]:
        for d_name in [f"B_minus_A_{metric}", f"C_minus_A_{metric}", f"B_minus_C_{metric}"]:
            d = paired.get(d_name)
            if d:
                lines.append(f"  {d_name:<22} {d['mean']:+.4f} \u00b1 {d['std']:.4f}")

    lines.append("")
    lines.append("Interpretation guide (paired AUROC):")
    lines.append("  B-A >> 0 and C-A ~ 0   ->  RAGTruth++ labels add useful supervision")
    lines.append("  B-A ~ 0  and C-A ~ 0   ->  no clear evidence simple retraining recovers")
    lines.append("                              RAGTruth++ distinctions (small data, calibration,")
    lines.append("                              or labels need different modeling)")
    lines.append("  B-A >> 0 and C-A >> 0  ->  matched-subset adaptation helps,")
    lines.append("                              not only label correction")
    lines.append("Primary metric: AUROC/AUPRC (threshold-free). F1 reported with per-fold")
    lines.append("threshold tuned on val (label_pp) for each condition independently.")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(sub_dir / "summary.txt", "w") as f:
        f.write(summary)


# =============================================================================
# Main
# =============================================================================
def stratified_random_subset(matched, n, label_field, seed):
    """Stratified random sample of n examples, preserving class balance."""
    rng = np.random.RandomState(seed)
    pos = [m for m in matched if m[label_field] == 1]
    neg = [m for m in matched if m[label_field] == 0]
    target_pos = max(int(n * len(pos) / len(matched)), 2)
    target_neg = max(n - target_pos, 2)
    target_pos = min(target_pos, len(pos))
    target_neg = min(target_neg, len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:target_pos] + neg[:target_neg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 80 examples, 1 epoch")
    parser.add_argument("--subset", choices=["full", "clean_test", "both"],
                        default="both", help="Which subset(s) to run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    matched_path = OUT_DIR / "matched_data.json"
    if matched_path.exists():
        print(f"Loading cached matched data from {matched_path}")
        with open(matched_path) as f:
            matched = json.load(f)
    else:
        matched = load_and_match()
        with open(matched_path, "w") as f:
            json.dump(matched, f, indent=2)
        print(f"Saved {matched_path}")

    n_epochs_use = 1 if args.smoke else MAX_EPOCHS

    if args.smoke:
        matched = stratified_random_subset(matched, 80, "label_pp", SEED)
        print(f"\nSMOKE MODE: stratified random sample, n={len(matched)}, "
              f"pos rate={np.mean([m['label_pp'] for m in matched]):.3f}")

    if args.subset in ("full", "both"):
        run_experiment(matched, "full", n_epochs_use, smoke=args.smoke)

    if args.subset in ("clean_test", "both"):
        clean = [m for m in matched if m["rt_split"] == "test"]
        if len(clean) < MIN_CLEAN_SIZE:
            print(f"\nSkipping clean_test: only {len(clean)} examples "
                  f"(< {MIN_CLEAN_SIZE} threshold)")
        else:
            run_experiment(clean, "clean_test", n_epochs_use, smoke=args.smoke)


if __name__ == "__main__":
    main()
