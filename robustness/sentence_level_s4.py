#!/usr/bin/env python
"""
Sentence-level S4 scoring on RAGTruth++ matched examples.

Hypothesis
==========
The response-level S4 signal degrades on RAGTruth++ because RAGTruth++
annotates hallucinations at SPAN level — one unsupported phrase inside an
otherwise-supported answer flips the label. Response-level S4 may average
out such localized failures. Sentence-level scoring with max-style
aggregation could better track the stricter annotation.

Comparison framing (IMPORTANT for thesis writeup)
=================================================
The `response_s4` numbers here come from the existing single-pass S4 scoring
on the RAGTruth test/train splits (signal4_results_*.json), filtered to the
408 matched examples. This is DIRECT inference comparison — NOT the 5-fold
CV baseline (Condition A) from the RAGTruth++ retraining experiment. Both
are valid measurements of "response-level S4 on these examples," but the
retraining-experiment numbers used CV folds and slightly different thresholds.

Design
======
- Input: matched RAGTruth++ examples (~408), already produced by the
  retraining experiment at /workspace/ragtruth_pp_retrain/matched_data.json
- For each example:
    * split answer into sentences
    * for each answer sentence, query S4 with [a_sent | context]  (same
      input format as response-level training — Option A)
    * collect per-sentence hallucination probabilities
- Compute four aggregators per example:
    * max        — highest sentence-level hallucination prob
    * mean       — average across answer sentences
    * top2_mean  — mean of top 2 sentence-level probs
    * noisy_or   — 1 - prod(1 - p_i)  (probabilistic OR; can saturate for
                   long answers — interpret with care)
- Compare each aggregator against:
    * RAGTruth++ labels   (the stricter, hypothesis-relevant target)
    * Original RAGTruth labels  (sanity / continuity)
- Also load existing response-level S4 scores (signal4_results_*.json)

What success looks like
=======================
If the hypothesis holds, max / top2_mean should give higher AUROC on
RAGTruth++ labels than response-level S4 does on the same 408 examples,
WITHOUT a corresponding gain on original RAGTruth labels (which would
suggest sentence-level is just better in general, not specifically aligned
with the stricter annotation).

What failure looks like
=======================
All aggregators give similar AUROC to response-level S4 → granularity is
not the bottleneck; the RAGTruth++ gap is something else (calibration,
evidence selection, context truncation, etc.).

Outputs
=======
    /workspace/sentence_level_s4/
        per_example_scores.json   — per-example dict with all aggregator
                                    scores AND the sentence texts (for
                                    qualitative inspection)
        results.json              — metrics per aggregator vs both label sets
        summary.txt               — readable comparison table with deltas
                                    against response_s4 for both label sets

Usage
=====
    # Smoke test (10 examples, ~1 min)
    python /workspace/sentence_level_s4.py --smoke

    # Full RAGTruth++ matched run (~30-60 min on V100)
    nohup python -u /workspace/sentence_level_s4.py \\
        > /workspace/sentence_level_s4.log 2>&1 &
    tail -f /workspace/sentence_level_s4.log

    # Clean-test-only (no leakage in response_s4 baseline since those
    # examples were never in the original S4's training data)
    nohup python -u /workspace/sentence_level_s4.py --clean-test-only \\
        > /workspace/sentence_level_s4_clean.log 2>&1 &
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =============================================================================
# Config
# =============================================================================
S4_MODEL_DIR = "/workspace/signal4_model"
MATCHED_PATH = Path("/workspace/ragtruth_pp_retrain/matched_data.json")
S4_TEST_PATH = "/workspace/signal4_results_test.json"
S4_TRAIN_PATH = "/workspace/signal4_results_train.json"

OUT_DIR = Path("/workspace/sentence_level_s4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
BATCH_SIZE = 16  # answer-sentences processed in parallel per example
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Sentence splitting
# =============================================================================
def split_sentences(text):
    try:
        import nltk
        try:
            sents = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except ImportError:
        import re
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s]


# =============================================================================
# Scoring
# =============================================================================
def score_sentences_against_context(model, tokenizer, sentences, context, batch_size):
    if not sentences:
        return []
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i + batch_size]
            inputs = tokenizer(
                batch_sents,
                [context] * len(batch_sents),
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            ).to(DEVICE)
            logits = model(**inputs).logits
            p_hall = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p_hall.tolist())
    return probs


# =============================================================================
# Aggregators
# =============================================================================
def aggregate(sentence_probs):
    if not sentence_probs:
        return {"max": 0.0, "mean": 0.0, "top2_mean": 0.0, "noisy_or": 0.0,
                "n_sentences": 0}
    arr = np.array(sentence_probs)
    sorted_desc = np.sort(arr)[::-1]
    top2 = sorted_desc[: min(2, len(arr))]
    eps = 1e-7
    one_minus = np.clip(1.0 - arr, eps, 1.0)
    noisy_or = 1.0 - float(np.prod(one_minus))
    return {
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "top2_mean": float(top2.mean()),
        "noisy_or": noisy_or,
        "n_sentences": int(len(arr)),
    }


# =============================================================================
# Metrics
# =============================================================================
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


def metrics_for(labels, scores, threshold):
    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "pos_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if len(set(labels.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        out["brier"] = float(brier_score_loss(labels, scores))
    else:
        out["auroc"] = None
        out["auprc"] = None
        out["brier"] = None
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="10-example sanity check (~1 min)")
    parser.add_argument("--clean-test-only", action="store_true",
                        help=("Use only examples matched to original RAGTruth "
                              "test split (no leakage in response_s4 baseline)"))
    args = parser.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ----- Load matched RAGTruth++ examples (with both label sets) -----
    print(f"\nLoading matched RAGTruth++ data from {MATCHED_PATH}...", flush=True)
    if not MATCHED_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHED_PATH} not found. Run ragtruth_pp_retrain.py first to "
            f"generate matched_data.json."
        )
    with open(MATCHED_PATH) as f:
        matched = json.load(f)
    print(f"  matched: {len(matched)} examples", flush=True)

    n_from_train = sum(1 for m in matched if m["rt_split"] == "train")
    n_from_test = sum(1 for m in matched if m["rt_split"] == "test")
    print(f"  source: train={n_from_train}  test={n_from_test}", flush=True)
    n_pp_pos = sum(m["label_pp"] for m in matched)
    n_orig_pos = sum(m["label_orig"] for m in matched)
    print(f"  RAGTruth++ pos rate: {n_pp_pos / len(matched):.3f}", flush=True)
    print(f"  Original   pos rate: {n_orig_pos / len(matched):.3f}", flush=True)

    if args.clean_test_only:
        matched = [m for m in matched if m["rt_split"] == "test"]
        print(f"\nCLEAN-TEST FILTER: using {len(matched)} test-only examples",
              flush=True)

    if args.smoke:
        matched = matched[:10]
        print(f"SMOKE MODE: using first {len(matched)} examples", flush=True)

    # ----- Load existing response-level S4 scores for the same examples -----
    print("\nLoading existing response-level S4 scores...", flush=True)
    s4_test = json.load(open(S4_TEST_PATH))
    s4_train = json.load(open(S4_TRAIN_PATH))
    s4_lookup = {}
    for r in s4_test:
        s4_lookup[("test", r["idx"])] = r["signal4_score"]
    for r in s4_train:
        s4_lookup[("train", r["idx"])] = r["signal4_score"]
    print(f"  s4 test={len(s4_test)} train={len(s4_train)}", flush=True)

    # ----- Load S4 model -----
    print(f"\nLoading S4 from {S4_MODEL_DIR}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(S4_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to(DEVICE).eval()

    # ----- Sentence-level scoring loop -----
    print(f"\nScoring {len(matched)} examples at sentence level...", flush=True)
    per_example = []
    t0 = time.time()
    total_sentences = 0

    for i, ex in enumerate(matched):
        sentences = split_sentences(ex["answer"])
        sentence_probs = score_sentences_against_context(
            model, tokenizer, sentences, ex["context"], BATCH_SIZE
        )
        aggs = aggregate(sentence_probs)
        total_sentences += aggs["n_sentences"]

        resp_s4 = s4_lookup.get((ex["rt_split"], ex["rt_idx"]))

        per_example.append({
            "stable_id": ex["stable_id"],
            "rt_split": ex["rt_split"],
            "rt_idx": ex["rt_idx"],
            "task_type": ex["task_type"],
            "model": ex["model"],
            "label_pp": ex["label_pp"],
            "label_orig": ex["label_orig"],
            "n_sentences": aggs["n_sentences"],
            "sent_max": aggs["max"],
            "sent_mean": aggs["mean"],
            "sent_top2_mean": aggs["top2_mean"],
            "sent_noisy_or": aggs["noisy_or"],
            "sentences": sentences,                       # saved for inspection
            "sentence_probs": [round(p, 4) for p in sentence_probs],
            "response_s4": resp_s4,
        })

        if (i + 1) % 25 == 0 or i + 1 == len(matched):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(matched) - (i + 1)) / rate
            print(f"  {i + 1}/{len(matched)} examples  "
                  f"({total_sentences} sentences, {elapsed:.0f}s, "
                  f"eta {eta:.0f}s)", flush=True)

    print(f"\nTotal: {total_sentences} sentences scored "
          f"(avg {total_sentences / len(matched):.1f} per example)", flush=True)

    # ----- Save per-example -----
    suffix = "_clean" if args.clean_test_only else ""
    pe_path = OUT_DIR / f"per_example_scores{suffix}.json"
    with open(pe_path, "w") as f:
        json.dump(per_example, f, indent=2)
    print(f"\nSaved per-example scores to {pe_path}", flush=True)

    # ----- Evaluation -----
    label_sets = {
        "label_pp": np.array([r["label_pp"] for r in per_example]),
        "label_orig": np.array([r["label_orig"] for r in per_example]),
    }
    scorers = {
        "response_s4": [r["response_s4"] for r in per_example],
        "sent_max": [r["sent_max"] for r in per_example],
        "sent_mean": [r["sent_mean"] for r in per_example],
        "sent_top2_mean": [r["sent_top2_mean"] for r in per_example],
        "sent_noisy_or": [r["sent_noisy_or"] for r in per_example],
    }

    results = {
        "config": {
            "n_examples": len(per_example),
            "clean_test_only": args.clean_test_only,
            "smoke": args.smoke,
            "avg_sentences_per_example": total_sentences / max(len(per_example), 1),
        },
        "per_label_set": {},
    }
    for lbl_name, labels in label_sets.items():
        results["per_label_set"][lbl_name] = {}
        for scorer_name, scores in scorers.items():
            kept = [(l, s) for l, s in zip(labels, scores) if s is not None]
            if not kept:
                results["per_label_set"][lbl_name][scorer_name] = None
                continue
            kl, ks = zip(*kept)
            kl, ks = list(kl), list(ks)
            t = find_best_threshold(kl, ks)
            m = metrics_for(kl, ks, t)
            m["dropped_for_missing"] = len(labels) - len(kl)
            results["per_label_set"][lbl_name][scorer_name] = m

    res_path = OUT_DIR / f"results{suffix}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # ----- Readable summary -----
    lines = []
    lines.append("=" * 96)
    lines.append("Sentence-level S4 on RAGTruth++ matched examples"
                 + ("  [clean-test-only]" if args.clean_test_only else ""))
    lines.append("=" * 96)
    lines.append(f"n_examples: {len(per_example)}  "
                 f"avg n_sentences/example: {total_sentences / len(matched):.1f}")
    lines.append("")
    lines.append("NOTE: response_s4 here is the original S4 score on these examples "
                 "(NOT the CV-baseline")
    lines.append("       from the retraining experiment). F1 thresholds are tuned on "
                 "the same data (small sample);")
    lines.append("       AUROC/AUPRC are threshold-free and are the primary metrics.")
    lines.append("")

    for lbl_name in ["label_pp", "label_orig"]:
        lines.append("-" * 96)
        target = "RAGTruth++ labels" if lbl_name == "label_pp" else "Original RAGTruth labels"
        lines.append(f"Target: {target}")
        lines.append("-" * 96)
        lines.append(f"{'Scorer':<18}{'F1':>10}{'Precision':>12}{'Recall':>10}"
                     f"{'AUROC':>10}{'AUPRC':>10}{'thr':>8}")
        for scorer_name in ["response_s4", "sent_max", "sent_mean",
                             "sent_top2_mean", "sent_noisy_or"]:
            m = results["per_label_set"][lbl_name].get(scorer_name)
            if m is None:
                lines.append(f"{scorer_name:<18}  (no data)")
                continue
            au = m.get('auroc')
            ap = m.get('auprc')
            au_s = f"{au:>10.4f}" if au is not None else f"{'—':>10}"
            ap_s = f"{ap:>10.4f}" if ap is not None else f"{'—':>10}"
            lines.append(
                f"{scorer_name:<18}"
                f"{m['f1']:>10.4f}"
                f"{m['precision']:>12.4f}"
                f"{m['recall']:>10.4f}"
                f"{au_s}{ap_s}"
                f"{m['threshold']:>8.2f}"
            )
        lines.append("")

    # Headline deltas for BOTH label sets — supports the granularity claim
    lines.append("=" * 96)
    lines.append("Hypothesis test: aggregator vs response_s4, per label set")
    lines.append("(Granularity hypothesis holds best when ΔAUROC > 0 for ++ "
                 "AND ≈ 0 (or negative) for original)")
    lines.append("=" * 96)

    for lbl_name in ["label_pp", "label_orig"]:
        target = "RAGTruth++" if lbl_name == "label_pp" else "Original RAGTruth"
        base = results["per_label_set"][lbl_name].get("response_s4")
        if not base:
            continue
        lines.append("")
        lines.append(f"  vs response_s4 on {target}:")
        for scorer_name in ["sent_max", "sent_mean", "sent_top2_mean", "sent_noisy_or"]:
            m = results["per_label_set"][lbl_name].get(scorer_name)
            if not m:
                continue
            d_auroc = (m["auroc"] - base["auroc"]
                       if m["auroc"] and base["auroc"] else None)
            d_auprc = (m["auprc"] - base["auprc"]
                       if m["auprc"] and base["auprc"] else None)
            d_f1 = m["f1"] - base["f1"]
            d_au_str = f"{d_auroc:+.4f}" if d_auroc is not None else "—"
            d_ap_str = f"{d_auprc:+.4f}" if d_auprc is not None else "—"
            lines.append(f"    {scorer_name:<18}  "
                          f"ΔAUROC={d_au_str}  ΔAUPRC={d_ap_str}  "
                          f"ΔF1={d_f1:+.4f}")

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    sm_path = OUT_DIR / f"summary{suffix}.txt"
    with open(sm_path, "w") as f:
        f.write(summary)
    print(f"\nAll outputs in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
