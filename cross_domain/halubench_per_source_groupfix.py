import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


CURVE_DIR = Path(
    "/workspace/halubench_curve_groupfix"
)

PRED_DIR = (
    CURVE_DIR
    / "per_run_predictions"
)

RESULTS_PATH = (
    CURVE_DIR
    / "results.json"
)

# Full-14k cached MiniCheck scores.
MC_PATH = Path(
    "/workspace/halubench_per_example_scores.json"
)

# Canonical corrected 8k split.
SPLIT_PATH = Path(
    "/workspace/halubench_group_split.json"
)

OUT_JSON = (
    CURVE_DIR
    / "per_source_results_groupfix.json"
)

OUT_TXT = (
    CURVE_DIR
    / "per_source_summary_groupfix.txt"
)

TRAIN_SIZES = [
    112,
    280,
    560,
    1120,
    2240,
]

SEEDS = [
    42,
    123,
    2024,
]

SOURCES = [
    "DROP",
    "FinanceBench",
    "covidQA",
    "halueval",
    "pubmedQA",
]

MC_THRESHOLD = 0.80


# ============================================================
# Metrics
# ============================================================

def metrics_for(
    labels,
    scores,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    preds = (
        scores >= threshold
    ).astype(int)

    result = {
        "n":
            int(len(labels)),

        "pos_rate":
            float(labels.mean()),

        "threshold":
            float(threshold),

        "f1":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division=0,
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    preds,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    preds,
                    zero_division=0,
                )
            ),
    }

    if len(set(labels.tolist())) > 1:

        result["auroc"] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        result["auprc"] = float(
            average_precision_score(
                labels,
                scores,
            )
        )

    else:

        result["auroc"] = None
        result["auprc"] = None

    return result


def per_source(
    predictions,
    threshold,
):
    grouped = defaultdict(
        lambda: {
            "labels": [],
            "scores": [],
        }
    )

    for row in predictions:

        grouped[
            row["source"]
        ]["labels"].append(
            int(row["label"])
        )

        grouped[
            row["source"]
        ]["scores"].append(
            float(row["score"])
        )

    result = {}

    for source in SOURCES:

        if source not in grouped:
            continue

        result[source] = metrics_for(
            grouped[source]["labels"],
            grouped[source]["scores"],
            threshold,
        )

    return result


def aggregate_runs(
    per_seed_results,
):
    output = {}

    for source in SOURCES:

        src_runs = [
            run[source]
            for run in per_seed_results
            if source in run
        ]

        if not src_runs:
            continue

        output[source] = {}

        for metric in [
            "auroc",
            "auprc",
            "f1",
            "precision",
            "recall",
        ]:

            vals = [
                r[metric]
                for r in src_runs
                if r.get(metric) is not None
            ]

            output[source][metric] = {
                "mean":
                    float(np.mean(vals)),

                "std":
                    float(np.std(vals)),

                "values":
                    [
                        float(v)
                        for v in vals
                    ],
            }

        output[source]["n"] = (
            src_runs[0]["n"]
        )

        output[source]["pos_rate"] = (
            src_runs[0]["pos_rate"]
        )

    return output


# ============================================================
# Load corrected curve results
# ============================================================

with open(RESULTS_PATH) as f:
    main_results = json.load(f)


threshold_lookup = {}

for row in main_results["per_run"]:

    threshold_lookup[
        (
            int(row["train_size"]),
            int(row["seed"]),
        )
    ] = float(
        row["best_threshold"]
    )


assert len(threshold_lookup) == 15, (
    f"Expected 15 corrected runs, "
    f"found {len(threshold_lookup)}"
)


# ============================================================
# Zero-shot corrected fixed-8k
# ============================================================

zero_path = (
    PRED_DIR
    / "zero_shot_predictions.json"
)

with open(zero_path) as f:
    zero_preds = json.load(f)

assert len(zero_preds) == 8000

zero_per_source = per_source(
    zero_preds,
    threshold=0.5,
)


# ============================================================
# Adapted S4 corrected fixed-8k
# ============================================================

adapted = {}


for n in TRAIN_SIZES:

    seed_results = []

    for seed in SEEDS:

        path = (
            PRED_DIR
            / f"predictions_n{n}_seed{seed}.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        with open(path) as f:
            preds = json.load(f)

        assert len(preds) == 8000

        threshold = threshold_lookup[
            (n, seed)
        ]

        seed_results.append(
            per_source(
                preds,
                threshold,
            )
        )

    adapted[str(n)] = aggregate_runs(
        seed_results
    )


# ============================================================
# MiniCheck on same corrected fixed-8k
# ============================================================

with open(MC_PATH) as f:
    mc_all = json.load(f)

assert len(mc_all) == 14000


mc_by_idx = {
    int(row["idx"]): row
    for row in mc_all
}


with open(SPLIT_PATH) as f:
    split = json.load(f)


test_ids = [
    int(x)
    for x in split[
        "test_filtered_indices"
    ]
]

assert len(test_ids) == 8000


mc_preds = []

for idx in test_ids:

    row = mc_by_idx[idx]

    mc_preds.append({
        "idx":
            idx,

        "label":
            int(row["label"]),

        "score":
            float(row["mc_hall"]),

        "source":
            row["source"],
    })


mc_per_source = per_source(
    mc_preds,
    threshold=MC_THRESHOLD,
)


# ============================================================
# Print AUROC table
# ============================================================

print(
    "=" * 113
)

print(
    "HALUBENCH GROUP-DISJOINT PER-SOURCE AUROC"
)

print(
    "=" * 113
)


header = (
    f"{'Source':<15}"
    f"{'Zero-shot':>12}"
    f"{'N=112':>16}"
    f"{'N=280':>16}"
    f"{'N=560':>16}"
    f"{'N=1120':>16}"
    f"{'N=2240':>16}"
    f"{'MiniCheck':>14}"
)

print(header)

print("-" * len(header))


lines = []

lines.append(
    "HaluBench group-disjoint per-source results\n"
)

lines.append(
    "Group key: source_ds + normalized question + normalized passage\n"
)

lines.append(
    "Fixed test N=8000; adaptation pool N=6000; "
    "outer and inner group overlap = 0.\n\n"
)


for source in SOURCES:

    zero = (
        zero_per_source[source]["auroc"]
    )

    mc = (
        mc_per_source[source]["auroc"]
    )

    cells = []

    for n in TRAIN_SIZES:

        r = adapted[str(n)][source][
            "auroc"
        ]

        cells.append(
            f"{r['mean']:.4f}±{r['std']:.4f}"
        )


    print(
        f"{source:<15}"
        f"{zero:>12.4f}"
        f"{cells[0]:>16}"
        f"{cells[1]:>16}"
        f"{cells[2]:>16}"
        f"{cells[3]:>16}"
        f"{cells[4]:>16}"
        f"{mc:>14.4f}"
    )


    lines.append(
        f"{source}\n"
    )

    lines.append(
        f"  zero-shot S4 AUROC: "
        f"{zero:.4f}\n"
    )

    for n in TRAIN_SIZES:

        r = adapted[str(n)][source][
            "auroc"
        ]

        lines.append(
            f"  N={n:<4} AUROC: "
            f"{r['mean']:.4f} "
            f"± {r['std']:.4f}\n"
        )

    lines.append(
        f"  MiniCheck-7B AUROC: "
        f"{mc:.4f}\n\n"
    )


# ============================================================
# Also print N=2240 detailed metrics
# ============================================================

print(
    "\n" + "=" * 105
)

print(
    "N=2240 ADAPTED S4 — DETAILED PER-SOURCE"
)

print(
    "=" * 105
)

print(
    f"{'Source':<15}"
    f"{'AUROC':>18}"
    f"{'AUPRC':>18}"
    f"{'F1':>18}"
    f"{'N':>8}"
    f"{'Pos rate':>12}"
)

print("-" * 105)


for source in SOURCES:

    r = adapted["2240"][source]

    auroc = f"{r['auroc']['mean']:.4f}±{r['auroc']['std']:.4f}"
    auprc = f"{r['auprc']['mean']:.4f}±{r['auprc']['std']:.4f}"
    f1 = f"{r['f1']['mean']:.4f}±{r['f1']['std']:.4f}"

    print(
        f"{source:<15}"
        f"{auroc:>18}"
        f"{auprc:>18}"
        f"{f1:>18}"
        f"{r['n']:>8}"
        f"{r['pos_rate']:>12.4f}"
    )


# ============================================================
# Save
# ============================================================

output = {
    "protocol": {
        "group_definition":
            "source_ds + normalized question + normalized passage",

        "fixed_test_n":
            8000,

        "adaptation_pool_n":
            6000,

        "outer_group_overlap":
            0,

        "inner_group_overlap":
            0,

        "train_sizes":
            TRAIN_SIZES,

        "seeds":
            SEEDS,
    },

    "zero_shot_s4":
        zero_per_source,

    "adapted_s4":
        adapted,

    "minicheck_7b":
        mc_per_source,
}


with open(OUT_JSON, "w") as f:
    json.dump(
        output,
        f,
        indent=2,
    )


with open(OUT_TXT, "w") as f:
    f.writelines(
        lines
    )


print(
    "\nSaved:",
    OUT_JSON
)

print(
    "Saved:",
    OUT_TXT
)
