import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)


SEED = 42
N_BOOT = 5000

S4_THRESHOLD = 0.50
FUSION_THRESHOLD = 0.45
MC_THRESHOLD = 0.80

# Canonical metadata-free RAGTruth fusion coefficients.
COEF_S2 = -1.359009650504923
COEF_S4 = 3.1555282346796876
INTERCEPT = -1.4168645142264538

S2S4_PATH = Path(
    "/workspace/halubench_final_s2s4_scores.json"
)

MC_PATH = Path(
    "/workspace/halubench_per_example_scores.json"
)

SPLIT_PATH = Path(
    "/workspace/halubench_group_split.json"
)

OUT = Path(
    "/workspace/bootstrap_halubench_groupfix_results.json"
)


def sigmoid(x):
    x = np.asarray(x, dtype=float)

    out = np.empty_like(x)

    pos = x >= 0
    neg = ~pos

    out[pos] = (
        1.0
        / (
            1.0
            + np.exp(-x[pos])
        )
    )

    ex = np.exp(x[neg])

    out[neg] = (
        ex
        / (1.0 + ex)
    )

    return out


def compute_ece(
    scores,
    labels,
    n_bins=10,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (scores >= bins[i])
                & (scores <= bins[i + 1])
            )
        else:
            mask = (
                (scores >= bins[i])
                & (scores < bins[i + 1])
            )

        if mask.sum() == 0:
            continue

        total += (
            mask.sum()
            * abs(
                labels[mask].mean()
                - scores[mask].mean()
            )
        )

    return float(
        total / len(labels)
    )


def metrics(
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

    return {
        "f1":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division=0,
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    labels,
                    scores,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    labels,
                    scores,
                )
            ),

        "ece":
            compute_ece(
                scores,
                labels,
            ),
    }


# ============================================================
# Load full 14k cached scores
# ============================================================

with open(S2S4_PATH) as f:
    s2s4_all = json.load(f)

with open(MC_PATH) as f:
    mc_all = json.load(f)

with open(SPLIT_PATH) as f:
    split = json.load(f)


assert len(s2s4_all) == 14000
assert len(mc_all) == 14000


s2s4_by_idx = {
    int(r["idx"]): r
    for r in s2s4_all
}

mc_by_idx = {
    int(r["idx"]): r
    for r in mc_all
}


test_idx = [
    int(x)
    for x in split[
        "test_filtered_indices"
    ]
]

assert len(test_idx) == 8000


# ============================================================
# Build exact corrected fixed-8k arrays
# ============================================================

labels = []
sources = []
s4_scores = []
s2_support = []
mc_scores = []


for idx in test_idx:

    a = s2s4_by_idx[idx]
    b = mc_by_idx[idx]

    assert int(a["idx"]) == idx
    assert int(b["idx"]) == idx

    assert int(a["label"]) == int(
        b["label"]
    )

    assert str(a["source"]) == str(
        b["source"]
    )

    labels.append(
        int(a["label"])
    )

    sources.append(
        str(a["source"])
    )

    s4_scores.append(
        float(a["s4_score"])
    )

    s2_support.append(
        float(a["s2_support"])
    )

    mc_scores.append(
        float(b["mc_hall"])
    )


labels = np.asarray(
    labels,
    dtype=int,
)

sources = np.asarray(
    sources,
    dtype=object,
)

s4_scores = np.asarray(
    s4_scores,
    dtype=float,
)

s2_support = np.asarray(
    s2_support,
    dtype=float,
)

mc_scores = np.asarray(
    mc_scores,
    dtype=float,
)


fusion_logit = (
    INTERCEPT
    + COEF_S2 * s2_support
    + COEF_S4 * s4_scores
)

fusion_scores = sigmoid(
    fusion_logit
)


assert len(labels) == 8000
assert abs(labels.mean() - 0.5) < 1e-12


# ============================================================
# Point estimates
# ============================================================

point = {
    "s4":
        metrics(
            labels,
            s4_scores,
            S4_THRESHOLD,
        ),

    "fusion":
        metrics(
            labels,
            fusion_scores,
            FUSION_THRESHOLD,
        ),

    "minicheck":
        metrics(
            labels,
            mc_scores,
            MC_THRESHOLD,
        ),
}


print("=" * 90)
print("HALUBENCH GROUP-DISJOINT FIXED-8K POINT ESTIMATES")
print("=" * 90)

for name in [
    "s4",
    "fusion",
    "minicheck",
]:

    m = point[name]

    print(
        f"{name:<12} "
        f"F1={m['f1']:.4f} "
        f"AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} "
        f"ECE={m['ece']:.4f}"
    )


# Fail immediately if we accidentally reconstructed
# a different endpoint from the already-verified one.
expected = {
    "s4": {
        "f1": 0.4432,
        "auroc": 0.5272,
        "auprc": 0.5025,
        "ece": 0.2894,
    },

    "fusion": {
        "f1": 0.3622,
        "auroc": 0.5319,
        "auprc": 0.5142,
        "ece": 0.2500,
    },

    "minicheck": {
        "f1": 0.7283,
        "auroc": 0.7974,
        "auprc": 0.8359,
        "ece": 0.1784,
    },
}


for model in expected:

    for metric in expected[model]:

        got = point[model][metric]
        want = expected[model][metric]

        if abs(got - want) > 0.00015:
            raise RuntimeError(
                f"Endpoint mismatch: "
                f"{model}/{metric}: "
                f"got {got:.6f}, "
                f"expected ~{want:.4f}"
            )


print(
    "\nEndpoint verification: PASSED"
)


# ============================================================
# Build source x label strata
# ============================================================

strata = defaultdict(list)

for i, (src, y) in enumerate(
    zip(
        sources,
        labels,
    )
):
    strata[
        (str(src), int(y))
    ].append(i)


strata = {
    k: np.asarray(v, dtype=int)
    for k, v in strata.items()
}


print(
    "\nBootstrap strata:"
)

for key in sorted(strata):

    print(
        f"  {key[0]:<15} "
        f"label={key[1]} "
        f"n={len(strata[key])}"
    )


# ============================================================
# Paired source x label stratified bootstrap
# ============================================================

METRICS = [
    "f1",
    "auroc",
    "auprc",
    "ece",
]


store = {
    "s4": {
        m: []
        for m in METRICS
    },

    "fusion": {
        m: []
        for m in METRICS
    },

    "minicheck": {
        m: []
        for m in METRICS
    },

    "delta_fusion_minus_s4": {
        m: []
        for m in METRICS
    },

    "delta_minicheck_minus_s4": {
        m: []
        for m in METRICS
    },

    "delta_minicheck_minus_fusion": {
        m: []
        for m in METRICS
    },
}


rng = np.random.default_rng(
    SEED
)


for b in range(N_BOOT):

    sampled_parts = []

    for key in sorted(strata):

        idxs = strata[key]

        sampled = rng.choice(
            idxs,
            size=len(idxs),
            replace=True,
        )

        sampled_parts.append(
            sampled
        )

    idx = np.concatenate(
        sampled_parts
    )

    rng.shuffle(idx)

    y_b = labels[idx]

    m_s4 = metrics(
        y_b,
        s4_scores[idx],
        S4_THRESHOLD,
    )

    m_fusion = metrics(
        y_b,
        fusion_scores[idx],
        FUSION_THRESHOLD,
    )

    m_mc = metrics(
        y_b,
        mc_scores[idx],
        MC_THRESHOLD,
    )


    for metric in METRICS:

        store[
            "s4"
        ][metric].append(
            m_s4[metric]
        )

        store[
            "fusion"
        ][metric].append(
            m_fusion[metric]
        )

        store[
            "minicheck"
        ][metric].append(
            m_mc[metric]
        )


        store[
            "delta_fusion_minus_s4"
        ][metric].append(
            m_fusion[metric]
            - m_s4[metric]
        )

        store[
            "delta_minicheck_minus_s4"
        ][metric].append(
            m_mc[metric]
            - m_s4[metric]
        )

        store[
            "delta_minicheck_minus_fusion"
        ][metric].append(
            m_mc[metric]
            - m_fusion[metric]
        )


# ============================================================
# Summaries
# ============================================================

def summarise(values):

    a = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(a.mean()),

        "lower_95":
            float(
                np.percentile(
                    a,
                    2.5,
                )
            ),

        "upper_95":
            float(
                np.percentile(
                    a,
                    97.5,
                )
            ),
    }


bootstrap = {
    name: {
        metric:
            summarise(values)
        for metric, values
        in metric_dict.items()
    }
    for name, metric_dict
    in store.items()
}


print(
    "\n" + "=" * 90
)

print(
    f"95% SOURCE x LABEL STRATIFIED "
    f"BOOTSTRAP CIs ({N_BOOT} resamples)"
)

print("=" * 90)


for model in [
    "s4",
    "fusion",
    "minicheck",
]:

    print(
        f"\n{model.upper()}"
    )

    for metric in METRICS:

        r = bootstrap[
            model
        ][metric]

        print(
            f"  {metric.upper():<6} "
            f"{point[model][metric]:.4f} "
            f"[{r['lower_95']:.4f}, "
            f"{r['upper_95']:.4f}]"
        )


delta_specs = [
    (
        "delta_fusion_minus_s4",
        "FUSION - S4",
        "fusion",
        "s4",
    ),

    (
        "delta_minicheck_minus_s4",
        "MINICHECK - S4",
        "minicheck",
        "s4",
    ),

    (
        "delta_minicheck_minus_fusion",
        "MINICHECK - FUSION",
        "minicheck",
        "fusion",
    ),
]


for key, label, a, b in delta_specs:

    print(
        f"\nPAIRED DELTA: {label}"
    )

    for metric in METRICS:

        r = bootstrap[
            key
        ][metric]

        point_delta = (
            point[a][metric]
            - point[b][metric]
        )

        print(
            f"  {metric.upper():<6} "
            f"{point_delta:+.4f} "
            f"[{r['lower_95']:+.4f}, "
            f"{r['upper_95']:+.4f}]"
        )


# ============================================================
# Save
# ============================================================

output = {
    "protocol": {
        "dataset":
            "HaluBench corrected group-disjoint fixed test",

        "n":
            8000,

        "positive_rate":
            float(labels.mean()),

        "group_definition":
            split["group_definition"],

        "n_bootstrap":
            N_BOOT,

        "bootstrap":
            "paired nonparametric bootstrap stratified by source x label",

        "ci":
            "95% percentile",

        "thresholds_fixed": {
            "s4":
                S4_THRESHOLD,

            "fusion":
                FUSION_THRESHOLD,

            "minicheck":
                MC_THRESHOLD,
        },

        "fusion_coefficients": {
            "intercept":
                INTERCEPT,

            "coef_s2_support":
                COEF_S2,

            "coef_s4":
                COEF_S4,
        },

        "important_note":
            "Bootstrap captures fixed-test sampling uncertainty while preserving source-label composition; it does not capture model-training uncertainty.",
    },

    "point_estimates":
        point,

    "bootstrap":
        bootstrap,
}


with open(OUT, "w") as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print(
    "\nSaved:",
    OUT
)
