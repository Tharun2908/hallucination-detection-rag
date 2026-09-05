#!/usr/bin/env python
"""
HaluBench adaptation curve — sample-efficiency experiment for the thesis
chapter on domain adaptation.

Question: how many HaluBench training examples are needed for the supervised
verifier (S4) to close the cross-domain gap, and where does the curve cross
MiniCheck-7B (the strong domain-robust baseline)?

Design
======
- HaluBench filtered to remove RAGTruth source -> 14,000 examples
- Fixed 8000-example test set with PROPORTIONAL stratification by source x
  label, held out once (preserves natural HaluBench distribution)
- Train pool = remaining 6000 examples
- Train sizes: 112 / 280 / 560 / 1120 / 2240 (Option 3 from design discussion)
- Train/val split: 80/20 within each size, source x label stratified
  Sizes: 112/28, 280/70, 560/140, 1120/280, 2240/560
- 3 seeds per training size -> 15 runs total
- Same training logic everywhere (max 5 epochs, save best by val AUROC):
  small-N val is noisy and we note this as a limitation
- Initialise from RAGTruth-finetuned S4 (NOT from DeBERTa base)
- Train sampling uses BALANCED source-label sampling so all sources are
  represented at every N (not proportional — proportional sampling at N=112
  would miss whole minority sources)
- Do not save model weights to disk (15 x 700MB would be wasteful);
  save predictions on the fixed test set per run, with original HaluBench
  indices for traceability

Outputs
=======
    /workspace/halubench_curve/
        config.json                  - run config + design choices
        results.json                 - per-run metrics + per-N aggregates
        results_incremental.json     - updated after each run
        per_run_predictions/*.json   - test predictions per run (with indices)
        summary.txt                  - readable summary
        curve_plot.png               - AUROC vs N with seed shading

Reference points to plot alongside the curve:
    - zero-shot S4 on the fixed test set (computed in this script)
    - MiniCheck-7B AUROC on HaluBench (already known: 0.7959 from notes)

Usage
=====
    # Smoke test (single small run, ~2 min)
    python /workspace/halubench_curve.py --smoke

    # Full run (~3-7 hours on V100)
    nohup python -u /workspace/halubench_curve.py \\
        > /workspace/halubench_curve.log 2>&1 &
    tail -f /workspace/halubench_curve.log
"""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# Config
# =============================================================================
S4_MODEL_DIR = "/workspace/signal4_model"
OUT_DIR = Path("/workspace/halubench_curve")
PRED_DIR = OUT_DIR / "per_run_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 8000
TRAIN_SIZES = [112, 280, 560, 1120, 2240]
SEEDS = [42, 123, 2024]

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MINICHECK_AUROC_REF = 0.7959


# =============================================================================
# Data loading
# =============================================================================
def load_halubench():
    """Load HaluBench, filter RAGTruth, return list of dicts with idx."""
    print("Loading HaluBench...", flush=True)
    ds = load_dataset("PatronusAI/HaluBench", split="test")
    ds = ds.filter(lambda x: x["source_ds"] != "RAGTruth")
    examples = []
    for i, ex in enumerate(ds):
        examples.append({
            "idx": i,  # preserved for traceability in predictions
            "context": ex["passage"],
            "answer": ex["answer"],
            "label": 1 if ex["label"] == "FAIL" else 0,
            "source": ex["source_ds"],
        })
    print(f"  total: {len(examples)}", flush=True)
    src_counter = Counter((e["source"], e["label"]) for e in examples)
    print("  by source x label:", flush=True)
    for (src, lbl), n in sorted(src_counter.items()):
        print(f"    {src}/{lbl}: {n}", flush=True)
    return examples


def hold_out_fixed_test(examples, test_size, seed=42):
    """PROPORTIONAL stratified train_pool / test split by (source, label).

    Preserves the natural HaluBench distribution in the test set so AUROC
    is comparable to the original HaluBench evaluation. The train pool gets
    the rest, also proportionally distributed.
    """
    strata = [f"{e['source']}__{e['label']}" for e in examples]
    indices = np.arange(len(examples))
    train_pool_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )
    return np.array(sorted(train_pool_idx)), np.array(sorted(test_idx))


def balanced_stratified_indices(strata, n_take, seed):
    """Roughly-balanced sampling across strata. Picks ~n_take/n_strata from
    each stratum, then fills the rest randomly. Used for TRAIN sampling so
    all sources are represented at every N. (NOT used for test.)"""
    rng = np.random.RandomState(seed)
    strata = np.array(strata)
    unique = np.unique(strata)
    per_strata = max(1, n_take // len(unique))

    chosen = []
    for s in unique:
        idxs = np.where(strata == s)[0]
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_strata].tolist())

    chosen_set = set(chosen)
    remaining = [i for i in range(len(strata)) if i not in chosen_set]
    rng.shuffle(remaining)
    needed = n_take - len(chosen)
    if needed > 0:
        chosen.extend(remaining[:needed])
    chosen = chosen[:n_take]
    return np.array(sorted(chosen))


def sample_train_val(train_pool_examples, n_train, val_ratio, seed):
    """Sample (train, val) from the pool, source x label stratified.

    val_ratio is relative to train size; 0.25 -> 80/20 train/val overall.
    Train sampling uses balanced source coverage so even N=112 contains all
    five HaluBench sources.
    """
    n_val = max(1, int(round(n_train * val_ratio)))
    n_total = n_train + n_val
    if n_total > len(train_pool_examples):
        raise ValueError(
            f"Requested {n_total} > train pool size {len(train_pool_examples)}"
        )

    strata = [f"{e['source']}__{e['label']}" for e in train_pool_examples]
    chosen_local = balanced_stratified_indices(strata, n_total, seed)
    chosen_examples = [train_pool_examples[i] for i in chosen_local]
    chosen_strata = [strata[i] for i in chosen_local]

    try:
        train_set, val_set = train_test_split(
            chosen_examples, test_size=n_val,
            random_state=seed, stratify=chosen_strata,
        )
    except ValueError:
        train_set, val_set = train_test_split(
            chosen_examples, test_size=n_val,
            random_state=seed,
            stratify=[e["label"] for e in chosen_examples],
        )
    return train_set, val_set


# =============================================================================
# Dataset + train/eval
# =============================================================================
class HaluDataset(Dataset):
    """Returns all tokenizer fields (input_ids, attention_mask, and
    token_type_ids if produced by the tokenizer) plus label. Matches the
    original S4 training pipeline conventions."""

    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["answer"], ex["context"],
            max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(ex["label"], dtype=torch.long)
        return item


def _move_to_device(batch):
    """Move all non-label tensors to DEVICE; return (inputs_dict, labels)."""
    labels = batch.pop("label") if "label" in batch else None
    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
    if labels is not None:
        labels = labels.to(DEVICE)
    return inputs, labels


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = dict(batch)  # don't mutate the DataLoader's batch
            inputs, labels = _move_to_device(batch)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(labels, scores):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def compute_metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)
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


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Train one (train_size, seed) configuration
# =============================================================================
def train_one_run(train_set, val_set, test_set, tokenizer, seed, smoke=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_loader = DataLoader(
        HaluDataset(train_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        HaluDataset(val_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        HaluDataset(test_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )

    n_pos = sum(e["label"] for e in train_set)
    n_neg = len(train_set) - n_pos
    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)
    class_weights = torch.tensor(
        [len(train_set) / (2 * n_neg), len(train_set) / (2 * n_pos)],
        dtype=torch.float,
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    n_epochs = 1 if smoke else MAX_EPOCHS
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    epoch_log = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = dict(batch)
            inputs, labels = _move_to_device(batch)
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            if smoke and n_batches >= 5:
                break

        val_probs, val_labels = evaluate(model, val_loader)
        val_auroc = (
            float(roc_auc_score(val_labels, val_probs))
            if len(set(val_labels)) > 1 else 0.0
        )
        print(
            f"      epoch {epoch}/{n_epochs}  "
            f"loss={total_loss / max(n_batches, 1):.4f}  "
            f"val_auroc={val_auroc:.4f}  (val_n={len(val_labels)})",
            flush=True,
        )
        epoch_log.append({
            "epoch": epoch, "loss": total_loss / max(n_batches, 1),
            "val_auroc": val_auroc,
        })

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"      early stop", flush=True)
                break

        if smoke:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_labels = evaluate(model, val_loader)
    best_t, _ = find_best_threshold(val_labels, val_probs)
    test_probs, test_labels = evaluate(model, test_loader)
    metrics = compute_metrics(test_labels, test_probs, best_t)
    metrics["best_val_auroc"] = best_val_auroc
    metrics["best_threshold"] = best_t
    metrics["epoch_log"] = epoch_log

    per_example = [
        {
            "idx": test_set[i]["idx"],
            "label": int(test_labels[i]),
            "score": float(test_probs[i]),
            "source": test_set[i]["source"],
        }
        for i in range(len(test_labels))
    ]

    del model
    free_gpu()
    return metrics, per_example


# =============================================================================
# Zero-shot reference
# =============================================================================
def evaluate_zero_shot(test_set, tokenizer):
    print("\nZero-shot reference: original S4 on fixed HaluBench test...",
          flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to(DEVICE)
    loader = DataLoader(
        HaluDataset(test_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    probs, labels = evaluate(model, loader)
    metrics = compute_metrics(labels, probs, threshold=0.5)
    metrics["threshold_note"] = "Zero-shot F1/Precision/Recall use threshold=0.5 (no held-out val)"
    print(f"  zero-shot S4: F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
          f"AUPRC={metrics['auprc']:.4f}", flush=True)
    print(f"  (F1 uses 0.5 threshold; AUROC/AUPRC are threshold-free)",
          flush=True)
    del model
    free_gpu()
    return metrics, [
        {
            "idx": test_set[i]["idx"],
            "label": int(labels[i]),
            "score": float(probs[i]),
            "source": test_set[i]["source"],
        }
        for i in range(len(labels))
    ]


# =============================================================================
# Aggregation + plot
# =============================================================================
def aggregate_by_size(results):
    by_size = {}
    for r in results:
        n = r["train_size"]
        by_size.setdefault(n, []).append(r)

    agg = {}
    for n, runs in by_size.items():
        metric_keys = ["f1", "precision", "recall", "auroc", "auprc", "brier"]
        agg[n] = {"n_seeds": len(runs)}
        for k in metric_keys:
            vals = [run[k] for run in runs if run.get(k) is not None]
            if not vals:
                agg[n][k] = None
                continue
            agg[n][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": [float(v) for v in vals],
            }
    return agg


def make_plot(aggregated, zero_shot_metrics, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot", flush=True)
        return

    sizes = sorted(aggregated.keys())
    means = [aggregated[n]["auroc"]["mean"] for n in sizes]
    stds = [aggregated[n]["auroc"]["std"] for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                linewidth=2, label="Adapted S4")
    ax.axhline(zero_shot_metrics["auroc"], color="tab:gray", linestyle="--",
               label=f"Zero-shot S4 ({zero_shot_metrics['auroc']:.3f})")
    ax.axhline(MINICHECK_AUROC_REF, color="tab:orange", linestyle="--",
               label=f"MiniCheck-7B ref ({MINICHECK_AUROC_REF:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("HaluBench training examples")
    ax.set_ylabel("Test AUROC (fixed 8000-example test)")
    ax.set_title("HaluBench adaptation curve — sample efficiency of S4")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}", flush=True)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Single small run (N=112, seed=42, 1 epoch)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    examples = load_halubench()
    cache_path = OUT_DIR / "test_train_pool_indices.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        train_pool_idx = np.array(cached["train_pool_idx"])
        test_idx = np.array(cached["test_idx"])
        print(f"\nLoaded cached test/pool split from {cache_path}", flush=True)
    else:
        train_pool_idx, test_idx = hold_out_fixed_test(examples, TEST_SIZE, seed=42)
        with open(cache_path, "w") as f:
            json.dump({
                "train_pool_idx": train_pool_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }, f)
        print(f"\nSaved test/pool split to {cache_path}", flush=True)

    train_pool = [examples[i] for i in train_pool_idx]
    test_set = [examples[i] for i in test_idx]
    print(f"  train pool: {len(train_pool)}  test: {len(test_set)}", flush=True)

    test_strata = Counter((e["source"], e["label"]) for e in test_set)
    print("  test set strata (source, label) — PROPORTIONAL to natural dist:",
          flush=True)
    for k, v in sorted(test_strata.items()):
        print(f"    {k}: {v}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(S4_MODEL_DIR)

    config = {
        "test_size": TEST_SIZE,
        "train_pool_size": len(train_pool),
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "model_dir": S4_MODEL_DIR,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "minicheck_auroc_ref": MINICHECK_AUROC_REF,
        "smoke": args.smoke,
        "notes": (
            "Test set is proportionally stratified (preserves HaluBench's "
            "natural distribution). Train set uses balanced source-label "
            "sampling so all five sources are represented at every N. "
            "Small-N val sets (28 at N=112) are noisy and may cause unstable "
            "checkpoint selection — reported as a limitation."
        ),
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    zs_metrics, zs_preds = evaluate_zero_shot(test_set, tokenizer)
    with open(PRED_DIR / "zero_shot_predictions.json", "w") as f:
        json.dump(zs_preds, f)

    if args.smoke:
        runs_to_do = [(112, 42)]
    else:
        runs_to_do = [(n, s) for n in TRAIN_SIZES for s in SEEDS]

    results = []
    for run_i, (n_train, seed) in enumerate(runs_to_do):
        print(f"\n{'=' * 70}", flush=True)
        print(f"Run {run_i + 1}/{len(runs_to_do)}: N={n_train}  seed={seed}",
              flush=True)
        print("=" * 70, flush=True)
        t0 = time.time()

        train_set, val_set = sample_train_val(
            train_pool, n_train, val_ratio=0.25, seed=seed
        )
        train_source_counts = dict(Counter(e["source"] for e in train_set))
        train_label_counts = dict(Counter(str(e["label"]) for e in train_set))
        val_label_counts = dict(Counter(str(e["label"]) for e in val_set))
        print(f"  train={len(train_set)}  val={len(val_set)}", flush=True)
        print(f"  train sources: {train_source_counts}", flush=True)
        print(f"  train labels:  {train_label_counts}", flush=True)
        print(f"  val labels:    {val_label_counts}", flush=True)

        metrics, per_example = train_one_run(
            train_set, val_set, test_set, tokenizer, seed, smoke=args.smoke
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.0f}s. test_F1={metrics['f1']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}",
              flush=True)

        run_record = {
            "train_size": n_train,
            "val_size": len(val_set),
            "seed": seed,
            "elapsed_s": elapsed,
            "train_source_counts": train_source_counts,
            "train_label_counts": train_label_counts,
            "val_label_counts": val_label_counts,
            **metrics,
        }
        results.append(run_record)

        pred_path = PRED_DIR / f"predictions_n{n_train}_seed{seed}.json"
        with open(pred_path, "w") as f:
            json.dump(per_example, f)

        with open(OUT_DIR / "results_incremental.json", "w") as f:
            json.dump({"per_run": results}, f, indent=2)

    aggregated = aggregate_by_size(results)
    final_results = {
        "config": config,
        "zero_shot": zs_metrics,
        "per_run": results,
        "aggregated": aggregated,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    lines = []
    lines.append("=" * 78)
    lines.append("HaluBench adaptation curve — sample efficiency of S4")
    lines.append("=" * 78)
    lines.append(f"Fixed test set: {len(test_set)} examples "
                 f"(proportionally stratified by source x label)")
    lines.append(f"Train pool:     {len(train_pool)} examples")
    lines.append(f"Seeds per N:    {len(SEEDS)}  Total runs: {len(results)}")
    lines.append("")
    lines.append("Reference points:")
    lines.append(f"  Zero-shot S4:    F1={zs_metrics['f1']:.4f}  "
                 f"AUROC={zs_metrics['auroc']:.4f}  "
                 f"AUPRC={zs_metrics['auprc']:.4f}")
    lines.append(f"                   (F1 uses threshold=0.5; AUROC/AUPRC are "
                 f"threshold-free)")
    lines.append(f"  MiniCheck-7B:    AUROC={MINICHECK_AUROC_REF:.4f}  "
                 f"(prior result from halubench_minicheck_results.json)")
    lines.append("")
    lines.append(f"{'N_train':<10}{'AUROC mean':>14}{'AUROC std':>12}"
                 f"{'F1 mean':>10}{'F1 std':>10}{'seeds':>8}")
    lines.append("-" * 64)
    for n in sorted(aggregated.keys()):
        a = aggregated[n]
        au = a.get("auroc")
        f1 = a.get("f1")
        if au and f1:
            lines.append(
                f"{n:<10}{au['mean']:>14.4f}{au['std']:>12.4f}"
                f"{f1['mean']:>10.4f}{f1['std']:>10.4f}{a['n_seeds']:>8}"
            )

    crossing_n = None
    for n in sorted(aggregated.keys()):
        au = aggregated[n].get("auroc")
        if au and au["mean"] >= MINICHECK_AUROC_REF:
            crossing_n = n
            break
    lines.append("")
    if crossing_n:
        lines.append(f"Adapted S4 matches/beats MiniCheck-7B at N >= {crossing_n}")
    else:
        lines.append("Adapted S4 did not match MiniCheck-7B reference at any "
                     "tested N.")
    lines.append("")
    lines.append("Note: small-N val sets (28 examples at N=112) are noisy; "
                 "variance across seeds at small N is likely large.")

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary)

    make_plot(aggregated, zs_metrics, OUT_DIR / "curve_plot.png")
    print(f"\nAll outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
