#!/usr/bin/env python
"""
Per-source breakdown of the HaluBench adaptation curve.

Question: which sources drive the strong aggregate AUROC? Does adapted S4
generalize uniformly across DROP / FinanceBench / covidQA / halueval /
pubmedQA, or does it lean on one source?

Reads per-run predictions from /workspace/halubench_curve/per_run_predictions/
and computes AUROC, AUPRC, F1 per source per (N, seed). Aggregates across
seeds. Compares to MiniCheck-7B per-source numbers from the prior run.

Outputs
=======
    /workspace/halubench_curve/per_source_results.json
    /workspace/halubench_curve/per_source_plot.png
    /workspace/halubench_curve/per_source_summary.txt
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


# =============================================================================
# Config
# =============================================================================
CURVE_DIR = Path("/workspace/halubench_curve")
PRED_DIR = CURVE_DIR / "per_run_predictions"
MINICHECK_PATH = Path("/workspace/halubench_minicheck_results.json")
TRAIN_SIZES = [112, 280, 560, 1120, 2240]
SEEDS = [42, 123, 2024]
SOURCES = ["DROP", "FinanceBench", "covidQA", "halueval", "pubmedQA"]


# =============================================================================
# Helpers
# =============================================================================
def find_best_threshold(labels, scores):
    """Sweep thresholds to maximize F1 (matches main curve script)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (np.array(scores) >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def metrics_for(labels, scores, threshold):
    if len(labels) == 0:
        return None
    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "pos_rate": float(np.mean(labels)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if len(set(labels.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def load_predictions(path):
    with open(path) as f:
        return json.load(f)


def per_source_metrics(predictions, threshold):
    """Group predictions by source, compute metrics per source."""
    by_src = defaultdict(lambda: {"labels": [], "scores": []})
    for p in predictions:
        by_src[p["source"]]["labels"].append(p["label"])
        by_src[p["source"]]["scores"].append(p["score"])
    return {
        src: metrics_for(v["labels"], v["scores"], threshold)
        for src, v in by_src.items()
    }


def aggregate_seeds(per_seed_metrics):
    """Given list of per-source-metric dicts (one per seed), return mean+std."""
    if not per_seed_metrics:
        return None
    sources = set()
    for m in per_seed_metrics:
        sources.update(m.keys())
    agg = {}
    for src in sources:
        vals = {"auroc": [], "auprc": [], "f1": [], "precision": [], "recall": [], "n": []}
        for m in per_seed_metrics:
            sm = m.get(src)
            if sm is None:
                continue
            for k in vals:
                if sm.get(k) is not None:
                    vals[k].append(sm[k])
        agg[src] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "values": [float(x) for x in v]}
            if v else None
            for k, v in vals.items()
        }
    return agg


# =============================================================================
# Main
# =============================================================================
def main():
    # --- Zero-shot per-source ---
    print("Computing zero-shot per-source...")
    zs_preds = load_predictions(PRED_DIR / "zero_shot_predictions.json")
    # Zero-shot used threshold=0.5 in the main script; keep that for F1 comparability
    zs_per_source = per_source_metrics(zs_preds, threshold=0.5)
    print(f"  zero-shot sources: {list(zs_per_source.keys())}")

    # --- Adapted S4 per-source per N per seed ---
    print("\nComputing adapted S4 per-source for each (N, seed)...")
    by_n = {}
    for n in TRAIN_SIZES:
        per_seed = []
        for seed in SEEDS:
            pred_path = PRED_DIR / f"predictions_n{n}_seed{seed}.json"
            if not pred_path.exists():
                print(f"  missing: {pred_path}")
                continue
            preds = load_predictions(pred_path)
            # Tune threshold globally on this run's predictions
            # (matches main script approach: threshold tuned on val, but val
            # isn't saved per-source. Best we can do here without retraining
            # is use the run's overall threshold from results.json.)
            labels = [p["label"] for p in preds]
            scores = [p["score"] for p in preds]
            # Load the threshold from results.json instead of re-tuning on test
            t = None  # filled below
            per_seed.append((preds, labels, scores))
        by_n[n] = per_seed

    # Load the per-run thresholds from results.json so we use the SAME threshold
    # the main script used (tuned on val, applied to test)
    with open(CURVE_DIR / "results.json") as f:
        main_results = json.load(f)
    threshold_lookup = {}
    for r in main_results["per_run"]:
        threshold_lookup[(r["train_size"], r["seed"])] = r["best_threshold"]

    # Now compute per-source metrics using the correct threshold per run
    print("\nComputing per-source metrics per run...")
    by_n_per_seed_metrics = {}
    for n in TRAIN_SIZES:
        per_seed_metrics = []
        for seed in SEEDS:
            pred_path = PRED_DIR / f"predictions_n{n}_seed{seed}.json"
            if not pred_path.exists():
                continue
            preds = load_predictions(pred_path)
            t = threshold_lookup.get((n, seed), 0.5)
            psm = per_source_metrics(preds, threshold=t)
            per_seed_metrics.append(psm)
        by_n_per_seed_metrics[n] = per_seed_metrics

    # Aggregate across seeds
    print("\nAggregating across seeds...")
    aggregated = {n: aggregate_seeds(seeds) for n, seeds in by_n_per_seed_metrics.items()}

    # --- MiniCheck per-source (reference) ---
    print("\nLoading MiniCheck-7B per-source reference...")
    mc_per_source = {}
    if MINICHECK_PATH.exists():
        with open(MINICHECK_PATH) as f:
            mc = json.load(f)
        if "per_domain" in mc:
            for src, m in mc["per_domain"].items():
                mc_per_source[src] = {
                    "auroc": m.get("auroc"),
                    "auprc": m.get("auprc"),
                    "f1": m.get("f1"),
                    "n": m.get("n"),
                    "pos_rate": m.get("pos_rate"),
                }
            print(f"  loaded MiniCheck per-source for: {list(mc_per_source.keys())}")
    else:
        print(f"  WARN: {MINICHECK_PATH} not found")

    # --- Save results ---
    output = {
        "zero_shot_per_source": zs_per_source,
        "adapted_s4_per_source_per_N": aggregated,
        "minicheck_per_source": mc_per_source,
        "sources": SOURCES,
        "train_sizes": TRAIN_SIZES,
    }
    out_path = CURVE_DIR / "per_source_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # --- Readable summary ---
    lines = []
    lines.append("=" * 100)
    lines.append("HaluBench adaptation curve — PER-SOURCE breakdown")
    lines.append("=" * 100)
    lines.append("Each cell shows AUROC mean ± std across 3 seeds (except zero-shot and MiniCheck, single runs)")
    lines.append("")
    header = f"{'Source':<16}{'ZS S4':>10}" + "".join(f"{f'N={n}':>14}" for n in TRAIN_SIZES) + f"{'MiniCheck':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for src in SOURCES:
        zs = zs_per_source.get(src, {})
        zs_str = f"{zs.get('auroc', float('nan')):.3f}" if zs and zs.get('auroc') is not None else "—"
        row = f"{src:<16}{zs_str:>10}"
        for n in TRAIN_SIZES:
            cell = aggregated.get(n, {}).get(src, {})
            au = cell.get("auroc") if cell else None
            if au is not None:
                row += f"{au['mean']:>7.3f}±{au['std']:.3f}"[:14].rjust(14)
            else:
                row += f"{'—':>14}"
        mc = mc_per_source.get(src, {})
        mc_str = f"{mc.get('auroc', float('nan')):.3f}" if mc and mc.get('auroc') is not None else "—"
        row += f"{mc_str:>12}"
        lines.append(row)

    lines.append("")
    lines.append("Same view for F1 (each cell = mean ± std across seeds; threshold tuned on val per run):")
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for src in SOURCES:
        zs = zs_per_source.get(src, {})
        zs_str = f"{zs.get('f1', float('nan')):.3f}" if zs and zs.get('f1') is not None else "—"
        row = f"{src:<16}{zs_str:>10}"
        for n in TRAIN_SIZES:
            cell = aggregated.get(n, {}).get(src, {})
            f1m = cell.get("f1") if cell else None
            if f1m is not None:
                row += f"{f1m['mean']:>7.3f}±{f1m['std']:.3f}"[:14].rjust(14)
            else:
                row += f"{'—':>14}"
        mc = mc_per_source.get(src, {})
        mc_str = f"{mc.get('f1', float('nan')):.3f}" if mc and mc.get('f1') is not None else "—"
        row += f"{mc_str:>12}"
        lines.append(row)

    # Crossover analysis per source
    lines.append("")
    lines.append("Per-source crossover with MiniCheck-7B (AUROC):")
    for src in SOURCES:
        mc_au = mc_per_source.get(src, {}).get("auroc")
        if mc_au is None:
            continue
        cross_n = None
        for n in TRAIN_SIZES:
            cell = aggregated.get(n, {}).get(src, {})
            au = cell.get("auroc") if cell else None
            if au and au["mean"] >= mc_au:
                cross_n = n
                break
        if cross_n is not None:
            lines.append(f"  {src:<16} crosses MiniCheck (AUROC {mc_au:.3f}) at N >= {cross_n}")
        else:
            lines.append(f"  {src:<16} does NOT cross MiniCheck (AUROC {mc_au:.3f}) at any tested N")

    # Hardest source identification
    lines.append("")
    lines.append("Hardest source at N=2240 (lowest AUROC):")
    n_top = 2240
    src_aurocs_at_top = []
    for src in SOURCES:
        cell = aggregated.get(n_top, {}).get(src, {})
        au = cell.get("auroc") if cell else None
        if au is not None:
            src_aurocs_at_top.append((src, au["mean"]))
    src_aurocs_at_top.sort(key=lambda x: x[1])
    for src, v in src_aurocs_at_top:
        lines.append(f"  {src:<16} AUROC={v:.4f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(CURVE_DIR / "per_source_summary.txt", "w") as f:
        f.write(summary)

    # --- Plot: per-source curves ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for src, color in zip(SOURCES, colors):
            means, stds = [], []
            for n in TRAIN_SIZES:
                cell = aggregated.get(n, {}).get(src, {})
                au = cell.get("auroc") if cell else None
                if au is None:
                    means.append(np.nan)
                    stds.append(0)
                else:
                    means.append(au["mean"])
                    stds.append(au["std"])
            ax.errorbar(TRAIN_SIZES, means, yerr=stds, marker="o", capsize=3,
                        linewidth=1.5, label=src, color=color)
            # MiniCheck reference line per source (dotted, same color)
            mc_au = mc_per_source.get(src, {}).get("auroc")
            if mc_au is not None:
                ax.axhline(mc_au, color=color, linestyle=":", alpha=0.5, linewidth=1)

        ax.set_xscale("log")
        ax.set_xlabel("HaluBench training examples")
        ax.set_ylabel("Test AUROC")
        ax.set_title("Per-source adaptation curves\n(dotted lines: MiniCheck-7B reference per source)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        out = CURVE_DIR / "per_source_plot.png"
        fig.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
