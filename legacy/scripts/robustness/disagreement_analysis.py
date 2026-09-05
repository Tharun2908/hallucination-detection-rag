#!/usr/bin/env python
"""
Disagreement / error-overlap analysis between the lightweight fusion (S2+S4)
and MiniCheck-7B on RAGTruth test set.

For the thesis chapter on complementary errors and cost-aware verification.

What it does
============
1. Loads per-example test scores for S2 (raw_min_relevance), S4, and MiniCheck-7B.
2. Refits the fusion logreg (S2+S4 + metadata) on train, applies to test.
3. Classifies each test example into one of four buckets using model-specific
   tuned thresholds:
       - both_correct
       - both_wrong
       - lightweight_wins  (fusion correct, MiniCheck wrong)
       - minicheck_wins    (MiniCheck correct, fusion wrong)
4. Breaks down each bucket by:
       - task_type
       - generator model
       - hallucination subtype (evident_conflict vs baseless_info)
       - answer length quartile
       - context length quartile
       - raw S2 min relevance bin
5. Continuous-disagreement view: top-50 most divergent (|fusion - minicheck|)
   examples for qualitative inspection.
6. Outputs JSON + readable text summary.

Outputs
=======
    /workspace/disagreement/results.json
    /workspace/disagreement/summary.txt
    /workspace/disagreement/top_divergent_examples.json

Usage
=====
    python /workspace/disagreement_analysis.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Config — paths and signal-direction conventions
# =============================================================================
S4_TEST = "/workspace/signal4_results_test.json"
S4_TRAIN = "/workspace/signal4_results_train_oof.json"
S2_TEST = "/workspace/relevance_results_test_v2.json"
S2_TRAIN = "/workspace/relevance_results_train_v2.json"
MINICHECK_TEST = "/workspace/minicheck_results_test_7b.json"

OUT_DIR = Path("/workspace/disagreement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# S2 normalization from thesis notes
S2_MIN, S2_MAX = -11.430, 10.641

# Fusion threshold (from existing fusion results: best_threshold=0.35)
# We refit logreg in this script but use the same threshold convention.
FUSION_THRESHOLD = 0.35

# MiniCheck-7B threshold tuned for F1 on train (we tune it below — placeholder)
# minicheck_score is support probability (high = faithful), so hallucination
# probability = 1 - minicheck_score
MINICHECK_THRESHOLD = None  # tuned in run-time


# =============================================================================
# Helpers
# =============================================================================
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_feature_matrix(s2_records, s4_records, task_types_order, models_order):
    """Build the same feature matrix as fusion_logreg_s2s4.py:
       [norm(1 - s2), s4_score, one-hot task_type, one-hot model]
    Score-direction convention: both columns higher => more hallucination."""
    by_idx_s2 = {r["idx"]: r for r in s2_records}
    by_idx_s4 = {r["idx"]: r for r in s4_records}

    indices = sorted(set(by_idx_s2.keys()) & set(by_idx_s4.keys()))

    rows = []
    labels = []
    meta = []
    for idx in indices:
        r2 = by_idx_s2[idx]
        r4 = by_idx_s4[idx]
        # S2: higher raw_min_relevance => more supported => less hallucination.
        # Invert so that higher feature => more hallucination
        s2_hall = 1.0 - norm_s2(r2["raw_min_relevance"])
        # S4: already higher = more hallucination
        s4_hall = float(r4["signal4_score"])

        # One-hot metadata
        tt = r4["task_type"]
        mdl = r4["model"]
        tt_onehot = [1.0 if t == tt else 0.0 for t in task_types_order]
        mdl_onehot = [1.0 if m == mdl else 0.0 for m in models_order]

        rows.append([s2_hall, s4_hall] + tt_onehot + mdl_onehot)
        labels.append(int(r4["ground_truth_hallucination"]))
        meta.append({"idx": idx, "task_type": tt, "model": mdl,
                     "s2_hall": s2_hall, "s4_hall": s4_hall})

    return np.array(rows), np.array(labels), meta


def tune_threshold_f1(y_true, scores, grid=None):
    """Pick threshold maximizing F1 on the given (y, scores)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (np.array(scores) >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def quartile_bins(values, n_bins=4):
    """Return list of bin labels Q1..Q4 (lowest..highest) for each value."""
    arr = np.array(values)
    qs = np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    labels = []
    for v in arr:
        b = 1
        for q in qs:
            if v > q:
                b += 1
        labels.append(f"Q{b}")
    return labels


# =============================================================================
# Main analysis
# =============================================================================
def main():
    print("Loading per-example scores...")
    s4_test = load_json(S4_TEST)
    s4_train = load_json(S4_TRAIN)
    s2_test = load_json(S2_TEST)
    s2_train = load_json(S2_TRAIN)
    mc_test = load_json(MINICHECK_TEST)
    print(f"  S4 test={len(s4_test)} train={len(s4_train)}")
    print(f"  S2 test={len(s2_test)} train={len(s2_train)}")
    print(f"  MiniCheck test={len(mc_test)}")

    # Establish task_type / model ordering from train (consistent one-hot)
    task_types_order = sorted({r["task_type"] for r in s4_train})
    models_order = sorted({r["model"] for r in s4_train})
    print(f"  task_types: {task_types_order}")
    print(f"  models: {models_order}")

    # Build feature matrices
    print("\nBuilding feature matrices...")
    X_train, y_train, _ = build_feature_matrix(s2_train, s4_train, task_types_order, models_order)
    X_test, y_test, meta_test = build_feature_matrix(s2_test, s4_test, task_types_order, models_order)
    print(f"  train: {X_train.shape}  pos={y_train.sum()}/{len(y_train)}")
    print(f"  test:  {X_test.shape}  pos={y_test.sum()}/{len(y_test)}")

    # Refit fusion logreg on train (scale features, balanced class weight)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    fusion_scores_test = clf.predict_proba(X_test_s)[:, 1]
    train_scores = clf.predict_proba(X_train_s)[:, 1]

    # Tune fusion threshold on train
    fusion_t, fusion_train_f1 = tune_threshold_f1(y_train, train_scores)
    fusion_preds = (fusion_scores_test >= fusion_t).astype(int)
    fusion_test_f1 = f1_score(y_test, fusion_preds, zero_division=0)
    fusion_test_auroc = roc_auc_score(y_test, fusion_scores_test)
    print(f"\nFusion refit: thr={fusion_t:.2f}  "
          f"train_F1={fusion_train_f1:.4f}  test_F1={fusion_test_f1:.4f}  "
          f"test_AUROC={fusion_test_auroc:.4f}")
    print("  (Uses out-of-fold S4 train scores for the fusion refit.)")

    # MiniCheck hallucination probability = 1 - support
    mc_by_idx = {r["idx"]: r for r in mc_test}
    # Tune MiniCheck threshold on its own train predictions if available; here
    # only test is available, so use the existing `minicheck_label` field
    # (already binarized using their own threshold) plus a separate F1-tuned
    # threshold from test for comparison.
    mc_scores_test = np.array([1.0 - mc_by_idx[m["idx"]]["minicheck_score"] for m in meta_test])
    mc_t, _ = tune_threshold_f1(y_test, mc_scores_test)  # tune on test — noted as limitation
    mc_preds = (mc_scores_test >= mc_t).astype(int)
    mc_test_f1 = f1_score(y_test, mc_preds, zero_division=0)
    mc_test_auroc = roc_auc_score(y_test, mc_scores_test)
    print(f"MiniCheck: thr={mc_t:.2f}  test_F1={mc_test_f1:.4f}  test_AUROC={mc_test_auroc:.4f}")
    print("  (NOTE: MC threshold tuned on test for fair F1; AUROC unaffected)")

    # ----------------------------------------------------------------------- #
    # Load RAGTruth test for extra metadata (subtype, lengths)
    # ----------------------------------------------------------------------- #
    print("\nLoading RAGTruth test for metadata enrichment...")
    ds_test = load_dataset("wandb/RAGTruth-processed", split="test")
    rt_by_idx = {}
    for i in range(len(ds_test)):
        ex = ds_test[i]
        labels = ex.get("hallucination_labels_processed", {}) or {}
        rt_by_idx[i] = {
            "answer_len": len(ex["output"]),
            "context_len": len(ex["context"]),
            "evident_conflict": labels.get("evident_conflict", 0),
            "baseless_info": labels.get("baseless_info", 0),
        }
    print(f"  loaded {len(rt_by_idx)} RAGTruth rows")

    # ----------------------------------------------------------------------- #
    # Bucket classification
    # ----------------------------------------------------------------------- #
    print("\nClassifying examples into 4 error buckets...")
    buckets = {
        "both_correct": [],
        "both_wrong": [],
        "lightweight_wins": [],   # fusion correct, MC wrong
        "minicheck_wins": [],     # MC correct, fusion wrong
    }
    per_example = []
    for i, m in enumerate(meta_test):
        idx = m["idx"]
        y = int(y_test[i])
        fp = int(fusion_preds[i])
        mp = int(mc_preds[i])
        f_corr = (fp == y)
        m_corr = (mp == y)
        if f_corr and m_corr:
            bucket = "both_correct"
        elif (not f_corr) and (not m_corr):
            bucket = "both_wrong"
        elif f_corr and not m_corr:
            bucket = "lightweight_wins"
        else:
            bucket = "minicheck_wins"

        rt = rt_by_idx.get(idx, {})
        record = {
            "idx": idx,
            "task_type": m["task_type"],
            "model": m["model"],
            "ground_truth": y,
            "fusion_score": float(fusion_scores_test[i]),
            "fusion_pred": fp,
            "minicheck_score_hall": float(mc_scores_test[i]),
            "minicheck_pred": mp,
            "s2_hall": float(m["s2_hall"]),
            "s4_hall": float(m["s4_hall"]),
            "bucket": bucket,
            "answer_len": rt.get("answer_len"),
            "context_len": rt.get("context_len"),
            "evident_conflict": rt.get("evident_conflict", 0),
            "baseless_info": rt.get("baseless_info", 0),
            "subtype": (
                "evident_conflict" if rt.get("evident_conflict", 0) > 0 and rt.get("baseless_info", 0) == 0
                else "baseless_info" if rt.get("baseless_info", 0) > 0 and rt.get("evident_conflict", 0) == 0
                else "mixed" if rt.get("evident_conflict", 0) > 0 and rt.get("baseless_info", 0) > 0
                else "none"
            ),
        }
        per_example.append(record)
        buckets[bucket].append(record)

    n = len(per_example)
    print(f"  total: {n}")
    for b, recs in buckets.items():
        print(f"    {b}: {len(recs)} ({len(recs) / n:.1%})")

    # Add length-quartile labels (computed on the full test set, not per-bucket)
    answer_q = quartile_bins([r["answer_len"] for r in per_example])
    context_q = quartile_bins([r["context_len"] for r in per_example])
    for r, aq, cq in zip(per_example, answer_q, context_q):
        r["answer_len_q"] = aq
        r["context_len_q"] = cq

    # ----------------------------------------------------------------------- #
    # Bucket breakdowns by attributes
    # ----------------------------------------------------------------------- #
    def crosstab(records, attr):
        c = Counter(r[attr] for r in records if r[attr] is not None)
        total = sum(c.values()) or 1
        return {k: {"count": v, "pct": v / total} for k, v in sorted(c.items())}

    def positive_rate(records):
        if not records:
            return None
        return float(np.mean([r["ground_truth"] for r in records]))

    breakdown = {}
    for b, recs in buckets.items():
        breakdown[b] = {
            "n": len(recs),
            "pct_of_all": len(recs) / n,
            "pos_rate": positive_rate(recs),
            "by_task_type": crosstab(recs, "task_type"),
            "by_model": crosstab(recs, "model"),
            "by_subtype": crosstab(recs, "subtype"),
            "by_answer_len_q": crosstab(recs, "answer_len_q"),
            "by_context_len_q": crosstab(recs, "context_len_q"),
        }

    # ----------------------------------------------------------------------- #
    # Top-50 most divergent examples (continuous score view)
    # ----------------------------------------------------------------------- #
    print("\nTop divergent examples (|fusion - minicheck_hall_prob|):")
    for r in per_example:
        r["abs_divergence"] = abs(r["fusion_score"] - r["minicheck_score_hall"])
    top_divergent = sorted(per_example, key=lambda x: x["abs_divergence"], reverse=True)[:50]
    print(f"  saving top-50 to top_divergent_examples.json")
    with open(OUT_DIR / "top_divergent_examples.json", "w") as f:
        json.dump(top_divergent, f, indent=2)

    # ----------------------------------------------------------------------- #
    # Save everything
    # ----------------------------------------------------------------------- #
    results = {
        "config": {
            "fusion_threshold": float(fusion_t),
            "minicheck_threshold": float(mc_t),
            "n_test": n,
            "fusion_test_f1": float(fusion_test_f1),
            "fusion_test_auroc": float(fusion_test_auroc),
            "minicheck_test_f1": float(mc_test_f1),
            "minicheck_test_auroc": float(mc_test_auroc),
        },
        "buckets": breakdown,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "per_example.json", "w") as f:
        json.dump(per_example, f, indent=2)

    # ----------------------------------------------------------------------- #
    # Readable summary
    # ----------------------------------------------------------------------- #
    lines = []
    lines.append("=" * 78)
    lines.append("Disagreement / error-overlap analysis: Fusion (S2+S4) vs MiniCheck-7B")
    lines.append("RAGTruth test (n={})".format(n))
    lines.append("=" * 78)
    lines.append(f"Fusion:    F1={fusion_test_f1:.4f}  AUROC={fusion_test_auroc:.4f}  thr={fusion_t:.2f}")
    lines.append(f"MiniCheck: F1={mc_test_f1:.4f}  AUROC={mc_test_auroc:.4f}  thr={mc_t:.2f}")
    lines.append("")
    lines.append(f"{'Bucket':<22}{'n':>8}{'%':>10}{'pos_rate':>12}")
    lines.append("-" * 52)
    for b in ["both_correct", "both_wrong", "lightweight_wins", "minicheck_wins"]:
        bd = breakdown[b]
        pr = f"{bd['pos_rate']:.3f}" if bd["pos_rate"] is not None else "—"
        lines.append(f"{b:<22}{bd['n']:>8}{bd['pct_of_all']:>9.1%}{pr:>12}")

    # Key narrative: cascade gain bucket
    lines.append("")
    lines.append("CASCADE GAIN BUCKET — examples where MiniCheck wins but lightweight loses:")
    lines.append("    (these are the examples that justify escalation)")
    mw = breakdown["minicheck_wins"]
    lines.append(f"    n={mw['n']}  pos_rate={mw['pos_rate']:.3f}")
    lines.append(f"    by_task_type:  " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in mw["by_task_type"].items()))
    lines.append(f"    by_model:      " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in mw["by_model"].items()))
    lines.append(f"    by_subtype:    " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in mw["by_subtype"].items()))
    lines.append(f"    by_answer_len: " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in mw["by_answer_len_q"].items()))
    lines.append(f"    by_context_len:" + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in mw["by_context_len_q"].items()))

    lines.append("")
    lines.append("LIGHTWEIGHT WINS — examples where lightweight is correct but MiniCheck wrong:")
    lines.append("    (these justify NOT escalating; cost saved)")
    lw = breakdown["lightweight_wins"]
    lines.append(f"    n={lw['n']}  pos_rate={lw['pos_rate']:.3f}")
    lines.append(f"    by_task_type:  " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in lw["by_task_type"].items()))
    lines.append(f"    by_model:      " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in lw["by_model"].items()))
    lines.append(f"    by_subtype:    " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in lw["by_subtype"].items()))

    lines.append("")
    lines.append("BOTH WRONG — genuinely hard examples for both verifiers:")
    bw = breakdown["both_wrong"]
    lines.append(f"    n={bw['n']}  pos_rate={bw['pos_rate']:.3f}")
    lines.append(f"    by_task_type:  " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in bw["by_task_type"].items()))
    lines.append(f"    by_subtype:    " + ", ".join(f"{k}={v['count']}({v['pct']:.0%})"
                                                     for k, v in bw["by_subtype"].items()))

    # Concentration ratio: are the cascade gains evenly distributed or concentrated?
    lines.append("")
    lines.append("Concentration check: largest task_type share in 'minicheck_wins' bucket:")
    largest = max(mw["by_task_type"].items(), key=lambda kv: kv[1]["count"])
    lines.append(f"    {largest[0]}: {largest[1]['count']}/{mw['n']} = {largest[1]['pct']:.1%}")
    largest_mw_model = max(mw["by_model"].items(), key=lambda kv: kv[1]["count"])
    lines.append(f"    {largest_mw_model[0]}: {largest_mw_model[1]['count']}/{mw['n']} = {largest_mw_model[1]['pct']:.1%}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
