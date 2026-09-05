#!/usr/bin/env python
"""
Bidirectional cross-domain curves: RAGTruth <-> HaluBench from DeBERTa base.

Research question
=================
Does cross-domain generalization in hallucination detection exhibit a
diversity-vs-narrowness asymmetry? Specifically:
  - Narrow specialist (RAGTruth-trained) generalizes to a diverse benchmark
    (HaluBench)?
  - Diverse specialist (HaluBench-trained) generalizes to a single domain
    (RAGTruth)?

If HaluBench -> RAGTruth performs substantially better than RAGTruth ->
HaluBench at matched N, then training-time diversity is the dominant factor
in cross-domain robustness — not just data quantity.

Design
======
- Two curves on one axis, fully symmetric:
    (1) RAGTruth -> HaluBench: train on N RAGTruth examples (DeBERTa base),
        test on fixed 8000-example HaluBench (proportional stratified)
    (2) HaluBench -> RAGTruth: train on N HaluBench examples (DeBERTa base),
        test on fixed 2700-example RAGTruth test split

- Training sizes:  112 / 280 / 560 / 1120 / 2240
- Seeds per N:     3   (42, 123, 2024)
- Total runs:      30
- All runs start from cross-encoder/nli-deberta-v3-base, the same backbone
  S4 was originally trained from. A fresh binary classification head is
  added on top (with ignore_mismatched_sizes=True since the NLI head is
  3-way). This isolates the effect of TRAINING DATA — same architecture,
  same backbone init, different training source.
- Train sampling stratified by an appropriate domain-balance key:
    HaluBench:  source_ds x label  (5 sources x 2 labels = 10 strata)
    RAGTruth:   task_type x label  (3 task types x 2 labels = 6 strata)

Reuses
======
- Existing HaluBench test/train pool split from halubench_curve/
- RAGTruth test split as-is (all 2700 examples)
- Will load fresh DeBERTa base via HF Hub (one-time download)

Outputs
=======
    /workspace/cross_direction/
        config.json
        results.json
        results_incremental.json
        rt_to_hb/predictions_n*_seed*.json
        hb_to_rt/predictions_n*_seed*.json
        rt_to_hb/zero_shot.json   <- base DeBERTa on HaluBench, threshold=0.5
        hb_to_rt/zero_shot.json   <- base DeBERTa on RAGTruth, threshold=0.5
        summary.txt
        bidirectional_plot.png

Usage
=====
    # Smoke (one tiny run each direction, ~3 min)
    python /workspace/cross_direction.py --smoke

    # Full (~8 hours)
    nohup python -u /workspace/cross_direction.py \\
        > /workspace/cross_direction.log 2>&1 &
    tail -f /workspace/cross_direction.log
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
BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
OUT_DIR = Path("/workspace/cross_direction")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "rt_to_hb").mkdir(exist_ok=True)
(OUT_DIR / "hb_to_rt").mkdir(exist_ok=True)

# Existing HaluBench split cache (so test set matches the adaptation curve)
HALUBENCH_SPLIT_CACHE = Path(
    "/workspace/halubench_curve/test_train_pool_indices.json"
)

TRAIN_SIZES = [112, 280, 560, 1120, 2240]
SEEDS = [42, 123, 2024]

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2
VAL_RATIO = 0.25  # of n_train; gives 80/20 train/val overall
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Loading utilities
# =============================================================================
def load_halubench():
    print("Loading HaluBench (excluding RAGTruth source)...", flush=True)
    ds = load_dataset("PatronusAI/HaluBench", split="test")
    ds = ds.filter(lambda x: x["source_ds"] != "RAGTruth")
    examples = []
    for i, ex in enumerate(ds):
        examples.append({
            "source_idx": i,
            "context": ex["passage"],
            "answer": ex["answer"],
            "label": 1 if ex["label"] == "FAIL" else 0,
            "domain": ex["source_ds"],
        })
    print(f"  HaluBench total: {len(examples)}", flush=True)
    return examples


def is_hallucinated(example):
    labels = example["hallucination_labels_processed"]
    return labels["evident_conflict"] > 0 or labels["baseless_info"] > 0


def load_ragtruth():
    print("Loading RAGTruth...", flush=True)
    ds_train = load_dataset("wandb/RAGTruth-processed", split="train")
    ds_test = load_dataset("wandb/RAGTruth-processed", split="test")
    train = []
    for i in range(len(ds_train)):
        ex = ds_train[i]
        train.append({
            "source_idx": i,
            "context": ex["context"],
            "answer": ex["output"],
            "label": int(is_hallucinated(ex)),
            "domain": ex.get("task_type", "unknown"),
        })
    test = []
    for i in range(len(ds_test)):
        ex = ds_test[i]
        test.append({
            "source_idx": i,
            "context": ex["context"],
            "answer": ex["output"],
            "label": int(is_hallucinated(ex)),
            "domain": ex.get("task_type", "unknown"),
        })
    print(f"  RAGTruth train: {len(train)}  test: {len(test)}", flush=True)
    return train, test


def load_halubench_split():
    """Use the same HaluBench train_pool/test split as the adaptation curve."""
    if not HALUBENCH_SPLIT_CACHE.exists():
        raise FileNotFoundError(
            f"Need {HALUBENCH_SPLIT_CACHE} — generate it by running "
            f"halubench_curve.py first (or its smoke)."
        )
    with open(HALUBENCH_SPLIT_CACHE) as f:
        d = json.load(f)
    return np.array(d["train_pool_idx"]), np.array(d["test_idx"])


# =============================================================================
# Stratified sampling
# =============================================================================
def balanced_stratified_indices(strata, n_take, seed):
    """Roughly-balanced sampling across strata. Used for TRAIN sampling so
    all domain-label cells are represented at every N."""
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


def sample_train_val(pool_examples, n_train, val_ratio, seed):
    """Stratified train/val from pool examples using 'domain x label'."""
    n_val = max(1, int(round(n_train * val_ratio)))
    n_total = n_train + n_val
    if n_total > len(pool_examples):
        raise ValueError(
            f"Requested {n_total} > pool size {len(pool_examples)}"
        )
    strata = [f"{e['domain']}__{e['label']}" for e in pool_examples]
    chosen_local = balanced_stratified_indices(strata, n_total, seed)
    chosen_examples = [pool_examples[i] for i in chosen_local]
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
# Dataset
# =============================================================================
class CrossDataset(Dataset):
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


# =============================================================================
# Train/eval helpers
# =============================================================================
def _move(batch):
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
            batch = dict(batch)
            inputs, labels = _move(batch)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(labels, scores):
    if len(set(labels)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (np.array(scores) >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def compute_metrics(labels, scores, threshold):
    preds = (np.array(scores) >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "pos_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) > 1:
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
# Training core
# =============================================================================
def train_one_run(train_set, val_set, test_set, tokenizer, seed, smoke=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
    ).to(DEVICE)

    train_loader = DataLoader(
        CrossDataset(train_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        CrossDataset(val_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        CrossDataset(test_set, tokenizer, MAX_LENGTH),
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
            inputs, labels = _move(batch)
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
            if len(set(val_labels.tolist())) > 1 else 0.0
        )
        print(
            f"      epoch {epoch}/{n_epochs}  "
            f"loss={total_loss / max(n_batches, 1):.4f}  "
            f"val_auroc={val_auroc:.4f}  (val_n={len(val_labels)})",
            flush=True,
        )
        epoch_log.append({
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
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
    best_t = find_best_threshold(val_labels, val_probs)
    test_probs, test_labels = evaluate(model, test_loader)
    metrics = compute_metrics(test_labels, test_probs, best_t)
    metrics["best_val_auroc"] = best_val_auroc
    metrics["best_threshold"] = best_t
    metrics["epoch_log"] = epoch_log

    per_example = [
        {
            "source_idx": test_set[i]["source_idx"],
            "label": int(test_labels[i]),
            "score": float(test_probs[i]),
            "domain": test_set[i]["domain"],
        }
        for i in range(len(test_labels))
    ]

    del model
    free_gpu()
    return metrics, per_example


# =============================================================================
# Zero-shot reference (base DeBERTa with random init on label head)
# =============================================================================
def zero_shot_reference(test_set, tokenizer, label):
    """NLI cross-encoder backbone with a freshly initialized binary head
    (random weights on classifier layer). Score distribution is essentially
    chance-level for this binary task; useful only as N=0 sanity baseline.
    NOT comparable to S4-zero-shot from earlier experiments — that meant
    'RAGTruth-finetuned S4 on HaluBench', whereas this means 'untrained
    binary head on target test'. Different concept."""
    print(f"\nN=0 reference: untrained 2-way head on {label} test "
          f"(NLI-DeBERTa backbone)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
    ).to(DEVICE).eval()
    loader = DataLoader(
        CrossDataset(test_set, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    probs, labels = evaluate(model, loader)
    metrics = compute_metrics(labels, probs, threshold=0.5)
    print(f"  N=0 ({label}): F1={metrics['f1']:.4f}  "
          f"AUROC={metrics['auroc']:.4f}", flush=True)
    del model
    free_gpu()
    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Two tiny runs (one per direction), 1 epoch")
    args = parser.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load data
    hb_examples = load_halubench()
    rt_train, rt_test = load_ragtruth()

    # HaluBench split (reuse adaptation curve's cached split)
    hb_pool_idx, hb_test_idx = load_halubench_split()
    hb_pool = [hb_examples[i] for i in hb_pool_idx]
    hb_test = [hb_examples[i] for i in hb_test_idx]
    print(f"\nHaluBench split: pool={len(hb_pool)}  test={len(hb_test)}",
          flush=True)
    print(f"RAGTruth pool: {len(rt_train)}  test: {len(rt_test)}", flush=True)

    # Print domain distributions for sanity
    print("\nHaluBench pool (source x label):", flush=True)
    for k, v in sorted(Counter((e["domain"], e["label"]) for e in hb_pool).items()):
        print(f"  {k}: {v}", flush=True)
    print("\nRAGTruth pool (task_type x label):", flush=True)
    for k, v in sorted(Counter((e["domain"], e["label"]) for e in rt_train).items()):
        print(f"  {k}: {v}", flush=True)

    # Tokenizer (shared)
    print(f"\nLoading tokenizer: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Save config
    config = {
        "base_model": BASE_MODEL,
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "val_ratio": VAL_RATIO,
        "hb_pool_size": len(hb_pool),
        "hb_test_size": len(hb_test),
        "rt_pool_size": len(rt_train),
        "rt_test_size": len(rt_test),
        "smoke": args.smoke,
        "notes": (
            "Both directions trained from cross-encoder/nli-deberta-v3-base "
            "(same backbone as S4) with a freshly initialized 2-way head. "
            "Train sampling balanced across domain x label strata so all "
            "domains are represented at every N. Test sets are proportional "
            "(HaluBench fixed 8000 from earlier curve; RAGTruth official "
            "2700 test split). Zero-shot row is the untrained binary head "
            "and is shown only as a chance-level reference; it is NOT "
            "comparable to S4-zero-shot from earlier experiments."
        ),
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Zero-shot baselines
    zs_hb = zero_shot_reference(hb_test, tokenizer, "HaluBench")
    zs_rt = zero_shot_reference(rt_test, tokenizer, "RAGTruth")
    with open(OUT_DIR / "rt_to_hb" / "zero_shot.json", "w") as f:
        json.dump(zs_hb, f, indent=2)
    with open(OUT_DIR / "hb_to_rt" / "zero_shot.json", "w") as f:
        json.dump(zs_rt, f, indent=2)

    # Run loop
    if args.smoke:
        runs_to_do = [("rt_to_hb", 112, 42), ("hb_to_rt", 112, 42)]
    else:
        runs_to_do = []
        for direction in ["rt_to_hb", "hb_to_rt"]:
            for n in TRAIN_SIZES:
                for s in SEEDS:
                    runs_to_do.append((direction, n, s))

    results = []
    for run_i, (direction, n_train, seed) in enumerate(runs_to_do):
        print(f"\n{'=' * 72}", flush=True)
        print(f"Run {run_i + 1}/{len(runs_to_do)}: "
              f"{direction}  N={n_train}  seed={seed}", flush=True)
        print("=" * 72, flush=True)
        t0 = time.time()

        if direction == "rt_to_hb":
            train_set, val_set = sample_train_val(rt_train, n_train, VAL_RATIO, seed)
            test_set = hb_test
        else:
            train_set, val_set = sample_train_val(hb_pool, n_train, VAL_RATIO, seed)
            test_set = rt_test

        train_dom_counts = dict(Counter(e["domain"] for e in train_set))
        train_lbl_counts = dict(Counter(str(e["label"]) for e in train_set))
        val_lbl_counts = dict(Counter(str(e["label"]) for e in val_set))
        print(f"  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}",
              flush=True)
        print(f"  train domains: {train_dom_counts}", flush=True)
        print(f"  train labels:  {train_lbl_counts}", flush=True)
        print(f"  val labels:    {val_lbl_counts}", flush=True)

        metrics, per_example = train_one_run(
            train_set, val_set, test_set, tokenizer, seed, smoke=args.smoke
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.0f}s. test_F1={metrics['f1']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}",
              flush=True)

        run_record = {
            "direction": direction,
            "train_size": n_train,
            "val_size": len(val_set),
            "seed": seed,
            "elapsed_s": elapsed,
            "train_domain_counts": train_dom_counts,
            "train_label_counts": train_lbl_counts,
            "val_label_counts": val_lbl_counts,
            **metrics,
        }
        results.append(run_record)

        pred_path = OUT_DIR / direction / f"predictions_n{n_train}_seed{seed}.json"
        with open(pred_path, "w") as f:
            json.dump(per_example, f)

        with open(OUT_DIR / "results_incremental.json", "w") as f:
            json.dump({"per_run": results}, f, indent=2)

    # Aggregate
    def agg(direction):
        by_n = {}
        for r in results:
            if r["direction"] != direction:
                continue
            by_n.setdefault(r["train_size"], []).append(r)
        out = {}
        for n, runs in by_n.items():
            keys = ["f1", "precision", "recall", "auroc", "auprc", "brier"]
            row = {"n_seeds": len(runs)}
            for k in keys:
                vals = [run[k] for run in runs if run.get(k) is not None]
                if vals:
                    row[k] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": [float(v) for v in vals],
                    }
                else:
                    row[k] = None
            out[n] = row
        return out

    aggregated = {
        "rt_to_hb": agg("rt_to_hb"),
        "hb_to_rt": agg("hb_to_rt"),
    }

    final_results = {
        "config": config,
        "zero_shot": {"hb": zs_hb, "rt": zs_rt},
        "per_run": results,
        "aggregated": aggregated,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Summary
    lines = []
    lines.append("=" * 84)
    lines.append("Bidirectional cross-domain curves "
                 "(cross-encoder/nli-deberta-v3-base + fresh 2-way head)")
    lines.append("=" * 84)
    lines.append(f"Zero-shot reference (untrained 2-way head, N=0):  "
                 f"HaluBench AUROC={zs_hb['auroc']:.3f}  "
                 f"RAGTruth AUROC={zs_rt['auroc']:.3f}")
    lines.append("  (chance-level sanity baseline; not comparable to "
                 "S4-zero-shot from other experiments)")
    lines.append("")
    lines.append("RAGTruth -> HaluBench  (trained on RT, tested on HB):")
    lines.append(f"{'N':<8}{'AUROC':>16}{'F1':>16}{'seeds':>8}")
    for n in sorted(aggregated["rt_to_hb"].keys()):
        a = aggregated["rt_to_hb"][n]
        au = a.get("auroc"); f1 = a.get("f1")
        if au and f1:
            lines.append(f"{n:<8}{au['mean']:>7.4f}±{au['std']:.4f}"
                         f"{f1['mean']:>9.4f}±{f1['std']:.4f}"
                         f"{a['n_seeds']:>8}")
    lines.append("")
    lines.append("HaluBench -> RAGTruth  (trained on HB, tested on RT):")
    lines.append(f"{'N':<8}{'AUROC':>16}{'F1':>16}{'seeds':>8}")
    for n in sorted(aggregated["hb_to_rt"].keys()):
        a = aggregated["hb_to_rt"][n]
        au = a.get("auroc"); f1 = a.get("f1")
        if au and f1:
            lines.append(f"{n:<8}{au['mean']:>7.4f}±{au['std']:.4f}"
                         f"{f1['mean']:>9.4f}±{f1['std']:.4f}"
                         f"{a['n_seeds']:>8}")

    # Asymmetry: at each N, which direction transfers better?
    lines.append("")
    lines.append("Direction asymmetry at matched N (AUROC):")
    lines.append(f"{'N':<8}{'RT->HB':>16}{'HB->RT':>16}{'delta (HB->RT minus RT->HB)':>32}")
    for n in TRAIN_SIZES:
        a1 = aggregated["rt_to_hb"].get(n, {}).get("auroc")
        a2 = aggregated["hb_to_rt"].get(n, {}).get("auroc")
        if a1 and a2:
            d = a2["mean"] - a1["mean"]
            lines.append(f"{n:<8}{a1['mean']:>16.4f}{a2['mean']:>16.4f}"
                         f"{d:>+32.4f}")

    lines.append("")
    lines.append("Reading guide:")
    lines.append("  - delta > 0 systematically: training-time DIVERSITY (HB) "
                 "transfers better than specialization (RT)")
    lines.append("  - delta < 0 systematically: training-time SPECIALIZATION "
                 "(RT) transfers better (unlikely)")
    lines.append("  - delta ~ 0: cross-domain failure is symmetric — neither "
                 "direction is privileged")

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for direction, label, color in [
            ("rt_to_hb", "RAGTruth -> HaluBench", "tab:blue"),
            ("hb_to_rt", "HaluBench -> RAGTruth", "tab:green"),
        ]:
            sizes = sorted(aggregated[direction].keys())
            means = [aggregated[direction][n]["auroc"]["mean"] for n in sizes]
            stds = [aggregated[direction][n]["auroc"]["std"] for n in sizes]
            ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                        linewidth=2, label=label, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Test AUROC (held-out target domain)")
        ax.set_title("Bidirectional cross-domain curves\n(NLI-DeBERTa backbone, fresh 2-way head)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "bidirectional_plot.png", dpi=150)
        print(f"\nSaved plot to {OUT_DIR / 'bidirectional_plot.png'}", flush=True)
    except ImportError:
        print("matplotlib not available — skipping plot", flush=True)

    print(f"\nAll outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
