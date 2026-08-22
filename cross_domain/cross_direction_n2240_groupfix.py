#!/usr/bin/env python

import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------
# Reuse the exact existing sampling/data-loading implementations.
# ---------------------------------------------------------------------
sys.path.insert(0, "/workspace/repo/cross_domain")

import halubench_curve_groupfix as hbfix
import cross_direction as old_cross


# =====================================================================
# CONFIG
# =====================================================================

BASE_MODEL_DIR = "/workspace/nli_deberta_v3_base_original"
HB_SPLIT_PATH = "/workspace/halubench_group_split.json"

OUT_DIR = Path("/workspace/cross_direction_n2240_groupfix")
PRED_DIR = OUT_DIR / "predictions"
SPLIT_DIR = OUT_DIR / "splits"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_N = 2240
VAL_RATIO = 0.25
SEEDS = [42, 123, 2024]

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_INCREMENTAL = OUT_DIR / "results_incremental.json"
RESULTS_FINAL = OUT_DIR / "results.json"


# =====================================================================
# BASIC HELPERS
# =====================================================================

class PairDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        enc = self.tokenizer(
            ex["answer"],
            ex["context"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            k: v.squeeze(0)
            for k, v in enc.items()
        }

        item["label"] = torch.tensor(
            ex["label"],
            dtype=torch.long,
        )

        return item


def move_batch(batch):
    batch = dict(batch)
    labels = batch.pop("label").to(DEVICE)

    inputs = {
        k: v.to(DEVICE)
        for k, v in batch.items()
    }

    return inputs, labels


def evaluate(model, examples, tokenizer):
    loader = DataLoader(
        PairDataset(examples, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, y = move_batch(batch)

            logits = model(**inputs).logits

            probs = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            scores.extend(
                probs.detach().cpu().numpy().tolist()
            )

            labels.extend(
                y.detach().cpu().numpy().tolist()
            )

    return (
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
    )


def find_best_threshold(labels, scores):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.05, 0.95, 19):
        preds = (scores >= t).astype(int)

        f1 = f1_score(
            labels,
            preds,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t


def compute_metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)

    return {
        "n": int(len(labels)),
        "pos_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "f1": float(
            f1_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "brier": float(
            brier_score_loss(
                labels,
                scores,
            )
        ),
    }


def free_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def example_id(ex):
    if "idx" in ex:
        return int(ex["idx"])

    if "source_idx" in ex:
        return int(ex["source_idx"])

    return None


def example_domain(ex):
    if "source" in ex:
        return ex["source"]

    if "domain" in ex:
        return ex["domain"]

    return "unknown"


def save_predictions(
    path,
    examples,
    labels,
    scores,
    threshold,
):
    rows = []

    for i, ex in enumerate(examples):
        rows.append({
            "idx": example_id(ex),
            "domain": example_domain(ex),
            "label": int(labels[i]),
            "score": float(scores[i]),
            "prediction": int(
                scores[i] >= threshold
            ),
        })

    with open(path, "w") as f:
        json.dump(rows, f)


# =====================================================================
# TRAIN ONE BASE-INITIALIZED MODEL
# =====================================================================

def train_one(
    train_set,
    val_set,
    tokenizer,
    seed,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # IMPORTANT:
    # original 3-way NLI checkpoint -> fresh 2-way classification head.
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL_DIR,
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        .to(DEVICE)
    )

    train_loader = DataLoader(
        PairDataset(train_set, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    n_pos = sum(
        int(ex["label"])
        for ex in train_set
    )

    n_neg = len(train_set) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            "Training split has only one class."
        )

    class_weights = torch.tensor(
        [
            len(train_set) / (2 * n_neg),
            len(train_set) / (2 * n_pos),
        ],
        dtype=torch.float,
        device=DEVICE,
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_steps = (
        len(train_loader)
        * MAX_EPOCHS
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                0.1 * total_steps
            ),
            num_training_steps=total_steps,
        )
    )

    best_val_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    epoch_log = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs, labels = move_batch(batch)

            optimizer.zero_grad()

            logits = model(
                **inputs
            ).logits

            loss = loss_fn(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += float(
                loss.item()
            )

            n_batches += 1

        val_scores, val_labels = evaluate(
            model,
            val_set,
            tokenizer,
        )

        val_auroc = float(
            roc_auc_score(
                val_labels,
                val_scores,
            )
        )

        epoch_loss = (
            total_loss
            / max(n_batches, 1)
        )

        epoch_log.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "val_auroc": val_auroc,
        })

        print(
            f"      epoch={epoch} "
            f"loss={epoch_loss:.4f} "
            f"val_AUROC={val_auroc:.4f}",
            flush=True,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):
                print(
                    "      early stopping",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No best model state obtained."
        )

    model.load_state_dict(
        best_state
    )

    del best_state

    val_scores, val_labels = evaluate(
        model,
        val_set,
        tokenizer,
    )

    threshold = find_best_threshold(
        val_labels,
        val_scores,
    )

    return (
        model,
        threshold,
        best_val_auroc,
        epoch_log,
    )


# =====================================================================
# AGGREGATION
# =====================================================================

def aggregate(records):
    keys = [
        "f1",
        "precision",
        "recall",
        "auroc",
        "auprc",
        "brier",
    ]

    out = {
        "n_seeds": len(records)
    }

    for key in keys:
        vals = [
            float(r[key])
            for r in records
        ]

        out[key] = {
            "mean": float(
                np.mean(vals)
            ),
            "std": float(
                np.std(vals)
            ),
            "values": vals,
        }

    return out


def load_incremental():
    if not RESULTS_INCREMENTAL.exists():
        return {
            "base_hb_to_hb": [],
            "base_hb_to_rt": [],
            "base_rt_to_hb": [],
        }

    with open(
        RESULTS_INCREMENTAL
    ) as f:
        return json.load(f)["experiments"]


def save_incremental(experiments):
    payload = {
        "experiments": experiments
    }

    with open(
        RESULTS_INCREMENTAL,
        "w",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def completed_seeds(records):
    return {
        int(r["seed"])
        for r in records
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 88)
    print(
        "CORRECTED N=2240 BIDIRECTIONAL "
        "TRANSFER + INITIALIZATION CONTROL"
    )
    print("=" * 88)

    print(
        f"Device: {DEVICE}",
        flush=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    # ------------------------------------------------------------
    # Verify base checkpoint.
    # ------------------------------------------------------------

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]

    for fname in required:
        p = (
            Path(BASE_MODEL_DIR)
            / fname
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing base-model file: {p}"
            )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL_DIR,
            local_files_only=True,
        )
    )

    # ------------------------------------------------------------
    # Load corrected HaluBench.
    # ------------------------------------------------------------

    hb_examples = (
        hbfix.load_halubench()
    )

    with open(
        HB_SPLIT_PATH
    ) as f:
        split = json.load(f)

    hb_pool_idx = np.asarray(
        split[
            "train_filtered_indices"
        ],
        dtype=int,
    )

    hb_test_idx = np.asarray(
        split[
            "test_filtered_indices"
        ],
        dtype=int,
    )

    if (
        len(hb_pool_idx) != 6000
        or len(hb_test_idx) != 8000
    ):
        raise RuntimeError(
            "Unexpected canonical "
            "HaluBench split sizes."
        )

    hb_pool = [
        hb_examples[i]
        for i in hb_pool_idx
    ]

    hb_test = [
        hb_examples[i]
        for i in hb_test_idx
    ]

    pool_groups = {
        e["group"]
        for e in hb_pool
    }

    test_groups = {
        e["group"]
        for e in hb_test
    }

    if pool_groups & test_groups:
        raise RuntimeError(
            "HaluBench outer group leakage."
        )

    print(
        f"HaluBench pool={len(hb_pool)} "
        f"test={len(hb_test)} "
        f"group_overlap=0",
        flush=True,
    )

    # ------------------------------------------------------------
    # Load RAGTruth.
    # ------------------------------------------------------------

    rt_train_pool, rt_test = (
        old_cross.load_ragtruth()
    )

    print(
        f"RAGTruth train={len(rt_train_pool)} "
        f"test={len(rt_test)}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Save config.
    # ------------------------------------------------------------

    config = {
        "base_model_dir":
            BASE_MODEL_DIR,

        "base_model_origin":
            "cross-encoder/nli-deberta-v3-base",

        "train_n":
            TRAIN_N,

        "val_ratio":
            VAL_RATIO,

        "seeds":
            SEEDS,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "halubench_split":
            HB_SPLIT_PATH,

        "design": {
            "base_hb_to_hb":
                "Base NLI -> HaluBench N=2240 -> corrected HaluBench 8k",

            "base_hb_to_rt":
                "Same HaluBench-trained base model -> RAGTruth test",

            "base_rt_to_hb":
                "Base NLI -> RAGTruth N=2240 -> corrected HaluBench 8k",
        },

        "notes": (
            "HaluBench train/validation sampling "
            "reuses halubench_curve_groupfix.py "
            "exactly. RAGTruth matched-N sampling "
            "reuses cross_direction.py. "
            "Thresholds are selected only on the "
            "source-domain validation split."
        ),
    }

    with open(
        OUT_DIR / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    experiments = (
        load_incremental()
    )

    # ============================================================
    # PART A
    # BASE -> HALUBENCH
    #
    # One model per seed, evaluated on BOTH HB and RT.
    # ============================================================

    hb_done = (
        completed_seeds(
            experiments[
                "base_hb_to_hb"
            ]
        )
        &
        completed_seeds(
            experiments[
                "base_hb_to_rt"
            ]
        )
    )

    for seed in SEEDS:
        if seed in hb_done:
            print(
                f"\nHB seed {seed} already "
                f"complete -- skipping.",
                flush=True,
            )
            continue

        print("\n" + "=" * 88)
        print(
            f"BASE -> HALUBENCH "
            f"N={TRAIN_N}, seed={seed}"
        )
        print("=" * 88)

        hb_train, hb_val = (
            hbfix.sample_train_val(
                hb_pool,
                TRAIN_N,
                VAL_RATIO,
                seed,
            )
        )

        if (
            len(hb_train) != TRAIN_N
            or len(hb_val) != 560
        ):
            raise RuntimeError(
                "Unexpected HB train/val sizes."
            )

        overlap = (
            {
                e["group"]
                for e in hb_train
            }
            &
            {
                e["group"]
                for e in hb_val
            }
        )

        if overlap:
            raise RuntimeError(
                "HB train/val group leakage."
            )

        split_record = {
            "seed": seed,
            "hb_train_idx": [
                int(e["idx"])
                for e in hb_train
            ],
            "hb_val_idx": [
                int(e["idx"])
                for e in hb_val
            ],
            "hb_train_groups": [
                e["group"]
                for e in hb_train
            ],
            "hb_val_groups": [
                e["group"]
                for e in hb_val
            ],
        }

        with open(
            SPLIT_DIR
            / f"hb_seed{seed}.json",
            "w",
        ) as f:
            json.dump(
                split_record,
                f,
            )

        print(
            f"  train={len(hb_train)} "
            f"val={len(hb_val)}"
        )

        print(
            "  train sources:",
            dict(
                Counter(
                    e["source"]
                    for e in hb_train
                )
            ),
        )

        t0 = time.time()

        (
            model,
            threshold,
            best_val_auroc,
            epoch_log,
        ) = train_one(
            hb_train,
            hb_val,
            tokenizer,
            seed,
        )

        # -----------------------------
        # Evaluate same model on HB.
        # -----------------------------

        hb_scores, hb_labels = evaluate(
            model,
            hb_test,
            tokenizer,
        )

        hb_metrics = compute_metrics(
            hb_labels,
            hb_scores,
            threshold,
        )

        hb_record = {
            "seed": seed,
            "train_size": TRAIN_N,
            "val_size": len(hb_val),
            "best_val_auroc":
                best_val_auroc,
            "best_threshold":
                threshold,
            "epoch_log":
                epoch_log,
            **hb_metrics,
        }

        # -----------------------------
        # Evaluate SAME model on RT.
        # Source-domain HB validation
        # threshold is carried over.
        # -----------------------------

        rt_scores, rt_labels = evaluate(
            model,
            rt_test,
            tokenizer,
        )

        rt_metrics = compute_metrics(
            rt_labels,
            rt_scores,
            threshold,
        )

        rt_record = {
            "seed": seed,
            "train_size": TRAIN_N,
            "val_size": len(hb_val),
            "source_validation_threshold":
                threshold,
            **rt_metrics,
        }

        save_predictions(
            PRED_DIR
            / f"base_hb_to_hb_seed{seed}.json",
            hb_test,
            hb_labels,
            hb_scores,
            threshold,
        )

        save_predictions(
            PRED_DIR
            / f"base_hb_to_rt_seed{seed}.json",
            rt_test,
            rt_labels,
            rt_scores,
            threshold,
        )

        # Replace existing same-seed record
        # if resuming after a partial rerun.
        experiments[
            "base_hb_to_hb"
        ] = [
            r
            for r in experiments[
                "base_hb_to_hb"
            ]
            if int(r["seed"]) != seed
        ]

        experiments[
            "base_hb_to_rt"
        ] = [
            r
            for r in experiments[
                "base_hb_to_rt"
            ]
            if int(r["seed"]) != seed
        ]

        experiments[
            "base_hb_to_hb"
        ].append(
            hb_record
        )

        experiments[
            "base_hb_to_rt"
        ].append(
            rt_record
        )

        save_incremental(
            experiments
        )

        elapsed = time.time() - t0

        print(
            f"  HB -> HB: "
            f"F1={hb_metrics['f1']:.4f} "
            f"AUROC={hb_metrics['auroc']:.4f}",
            flush=True,
        )

        print(
            f"  HB -> RT: "
            f"F1={rt_metrics['f1']:.4f} "
            f"AUROC={rt_metrics['auroc']:.4f}",
            flush=True,
        )

        print(
            f"  elapsed={elapsed:.0f}s",
            flush=True,
        )

        del model
        free_gpu()

    # ============================================================
    # PART B
    # BASE -> RAGTRUTH -> HALUBENCH
    # ============================================================

    rt_done = completed_seeds(
        experiments[
            "base_rt_to_hb"
        ]
    )

    for seed in SEEDS:
        if seed in rt_done:
            print(
                f"\nRT seed {seed} already "
                f"complete -- skipping.",
                flush=True,
            )
            continue

        print("\n" + "=" * 88)
        print(
            f"BASE -> RAGTRUTH "
            f"N={TRAIN_N}, seed={seed}"
        )
        print("=" * 88)

        rt_train, rt_val = (
            old_cross.sample_train_val(
                rt_train_pool,
                TRAIN_N,
                VAL_RATIO,
                seed,
            )
        )

        if (
            len(rt_train) != TRAIN_N
            or len(rt_val) != 560
        ):
            raise RuntimeError(
                "Unexpected RT train/val sizes."
            )

        split_record = {
            "seed": seed,
            "rt_train_idx": [
                int(e["source_idx"])
                for e in rt_train
            ],
            "rt_val_idx": [
                int(e["source_idx"])
                for e in rt_val
            ],
        }

        with open(
            SPLIT_DIR
            / f"rt_seed{seed}.json",
            "w",
        ) as f:
            json.dump(
                split_record,
                f,
            )

        print(
            f"  train={len(rt_train)} "
            f"val={len(rt_val)}"
        )

        print(
            "  train tasks:",
            dict(
                Counter(
                    e["domain"]
                    for e in rt_train
                )
            ),
        )

        t0 = time.time()

        (
            model,
            threshold,
            best_val_auroc,
            epoch_log,
        ) = train_one(
            rt_train,
            rt_val,
            tokenizer,
            seed,
        )

        hb_scores, hb_labels = evaluate(
            model,
            hb_test,
            tokenizer,
        )

        metrics = compute_metrics(
            hb_labels,
            hb_scores,
            threshold,
        )

        record = {
            "seed": seed,
            "train_size": TRAIN_N,
            "val_size": len(rt_val),
            "best_val_auroc":
                best_val_auroc,
            "best_threshold":
                threshold,
            "epoch_log":
                epoch_log,
            **metrics,
        }

        save_predictions(
            PRED_DIR
            / f"base_rt_to_hb_seed{seed}.json",
            hb_test,
            hb_labels,
            hb_scores,
            threshold,
        )

        experiments[
            "base_rt_to_hb"
        ] = [
            r
            for r in experiments[
                "base_rt_to_hb"
            ]
            if int(r["seed"]) != seed
        ]

        experiments[
            "base_rt_to_hb"
        ].append(
            record
        )

        save_incremental(
            experiments
        )

        elapsed = time.time() - t0

        print(
            f"  RT -> HB: "
            f"F1={metrics['f1']:.4f} "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )

        print(
            f"  elapsed={elapsed:.0f}s",
            flush=True,
        )

        del model
        free_gpu()

    # ============================================================
    # FINAL AGGREGATION
    # ============================================================

    for name in experiments:
        experiments[name] = sorted(
            experiments[name],
            key=lambda x: int(
                x["seed"]
            ),
        )

    aggregated = {
        name: aggregate(records)
        for name, records
        in experiments.items()
    }

    # Existing corrected S4-init comparator,
    # if available.
    comparator = None

    existing = Path(
        "/workspace/halubench_curve_groupfix/results.json"
    )

    if existing.exists():
        try:
            with open(existing) as f:
                old = json.load(f)

            agg2240 = old[
                "aggregated"
            ].get(
                "2240",
                old["aggregated"].get(
                    2240
                )
            )

            comparator = agg2240
        except Exception as exc:
            comparator = {
                "load_error":
                    str(exc)
            }

    final = {
        "config": config,
        "experiments":
            experiments,
        "aggregated":
            aggregated,
        "s4_init_hb_n2240_comparator":
            comparator,
    }

    with open(
        RESULTS_FINAL,
        "w",
    ) as f:
        json.dump(
            final,
            f,
            indent=2,
        )

    lines = []

    lines.append("=" * 88)
    lines.append(
        "CORRECTED N=2240 "
        "BIDIRECTIONAL / INIT CONTROL"
    )
    lines.append("=" * 88)

    for name in [
        "base_hb_to_hb",
        "base_hb_to_rt",
        "base_rt_to_hb",
    ]:
        a = aggregated[name]

        lines.append("")
        lines.append(name)

        lines.append(
            "  F1:    "
            f"{a['f1']['mean']:.4f} "
            f"+/- {a['f1']['std']:.4f}"
        )

        lines.append(
            "  AUROC: "
            f"{a['auroc']['mean']:.4f} "
            f"+/- {a['auroc']['std']:.4f}"
        )

        lines.append(
            "  AUPRC: "
            f"{a['auprc']['mean']:.4f} "
            f"+/- {a['auprc']['std']:.4f}"
        )

    if comparator:
        try:
            lines.append("")
            lines.append(
                "Existing corrected "
                "S4-init HB -> HB N=2240"
            )

            lines.append(
                "  F1:    "
                f"{comparator['f1']['mean']:.4f} "
                f"+/- "
                f"{comparator['f1']['std']:.4f}"
            )

            lines.append(
                "  AUROC: "
                f"{comparator['auroc']['mean']:.4f} "
                f"+/- "
                f"{comparator['auroc']['std']:.4f}"
            )
        except Exception:
            pass

    summary = "\n".join(lines)

    with open(
        OUT_DIR / "summary.txt",
        "w",
    ) as f:
        f.write(summary + "\n")

    print("\n" + summary)
    print(
        f"\nSaved final results: "
        f"{RESULTS_FINAL}",
        flush=True,
    )


if __name__ == "__main__":
    main()
