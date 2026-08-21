import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)


SEED = 42
N_BOOT = 5000

S2_MIN = -11.430
S2_MAX = 10.641

S4_THRESHOLD = 0.55
FUSION_THRESHOLD = 0.40

OUT = Path(
    "/workspace/bootstrap_ragtruth_main_results.json"
)


def norm_s2(x):
    return float(
        np.clip(
            (float(x) - S2_MIN)
            / (S2_MAX - S2_MIN),
            0.0,
            1.0,
        )
    )


def load_map(path):
    with open(path) as f:
        return {
            int(r["idx"]): r
            for r in json.load(f)
        }


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

        if not mask.any():
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
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    pred = (
        scores >= threshold
    ).astype(int)

    return {
        "f1":
            float(
                f1_score(
                    labels,
                    pred,
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
# Load canonical train/test signals
# ============================================================

s1_train = load_map(
    "/workspace/nli_results_train_v2.json"
)

s2_train = load_map(
    "/workspace/relevance_results_train_v2.json"
)

s4_train = load_map(
    "/workspace/signal4_results_train_oof.json"
)

s1_test = load_map(
    "/workspace/nli_results_test_v2.json"
)

s2_test = load_map(
    "/workspace/relevance_results_test_v2.json"
)

s4_test = load_map(
    "/workspace/signal4_results_test.json"
)


def build(
    s1,
    s2,
    s4,
):
    common = sorted(
        set(s1)
        & set(s2)
        & set(s4)
    )

    rows = []

    for idx in common:

        a = s1[idx]
        b = s2[idx]
        c = s4[idx]

        if (
            b["raw_min_relevance"] is None
            or c["signal4_score"] is None
        ):
            continue

        rows.append({
            "idx":
                idx,

            "label":
                int(
                    a[
                        "ground_truth_hallucination"
                    ]
                ),

            "task":
                a["task_type"],

            "model":
                a["model"],

            "s2":
                norm_s2(
                    b[
                        "raw_min_relevance"
                    ]
                ),

            "s4":
                float(
                    c[
                        "signal4_score"
                    ]
                ),
        })

    return rows


train = build(
    s1_train,
    s2_train,
    s4_train,
)

test = build(
    s1_test,
    s2_test,
    s4_test,
)


assert len(train) == 15090
assert len(test) == 2700


# ============================================================
# Fit canonical final S2 + S4 + metadata fusion once.
#
# Bootstrap evaluates sampling uncertainty of the FIXED model.
# ============================================================

X_train_base = np.array([
    [r["s2"], r["s4"]]
    for r in train
])

X_test_base = np.array([
    [r["s2"], r["s4"]]
    for r in test
])

y_train = np.array([
    r["label"]
    for r in train
])

y_test = np.array([
    r["label"]
    for r in test
])


cat_train = np.array([
    [r["task"], r["model"]]
    for r in train
])

cat_test = np.array([
    [r["task"], r["model"]]
    for r in test
])


try:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )


X_train = np.hstack([
    X_train_base,
    ohe.fit_transform(
        cat_train
    ),
])

X_test = np.hstack([
    X_test_base,
    ohe.transform(
        cat_test
    ),
])


clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

clf.fit(
    X_train,
    y_train,
)


fusion_scores = (
    clf.predict_proba(
        X_test
    )[:, 1]
)

s4_scores = np.array([
    r["s4"]
    for r in test
])


# ============================================================
# Point estimates
# ============================================================

point = {
    "s4":
        metrics(
            y_test,
            s4_scores,
            S4_THRESHOLD,
        ),

    "fusion":
        metrics(
            y_test,
            fusion_scores,
            FUSION_THRESHOLD,
        ),
}


print("=" * 80)
print("POINT ESTIMATES")
print("=" * 80)

for model in [
    "s4",
    "fusion",
]:
    m = point[model]

    print(
        f"{model:<10} "
        f"F1={m['f1']:.4f} "
        f"AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} "
        f"ECE={m['ece']:.4f}"
    )


# ============================================================
# Stratified paired bootstrap
#
# Positive and negative examples are resampled separately,
# preserving the observed class counts.
#
# The SAME sampled indices are used for S4 and fusion.
# ============================================================

rng = np.random.default_rng(
    SEED
)

pos_idx = np.where(
    y_test == 1
)[0]

neg_idx = np.where(
    y_test == 0
)[0]


store = {
    "s4": {
        k: []
        for k in [
            "f1",
            "auroc",
            "auprc",
            "ece",
        ]
    },

    "fusion": {
        k: []
        for k in [
            "f1",
            "auroc",
            "auprc",
            "ece",
        ]
    },

    "delta_fusion_minus_s4": {
        k: []
        for k in [
            "f1",
            "auroc",
            "auprc",
            "ece",
        ]
    },
}


for b in range(N_BOOT):

    sampled_pos = rng.choice(
        pos_idx,
        size=len(pos_idx),
        replace=True,
    )

    sampled_neg = rng.choice(
        neg_idx,
        size=len(neg_idx),
        replace=True,
    )

    idx = np.concatenate([
        sampled_pos,
        sampled_neg,
    ])

    # Shuffle only for completeness.
    rng.shuffle(idx)

    labels_b = y_test[idx]

    s4_b = metrics(
        labels_b,
        s4_scores[idx],
        S4_THRESHOLD,
    )

    fusion_b = metrics(
        labels_b,
        fusion_scores[idx],
        FUSION_THRESHOLD,
    )

    for metric in [
        "f1",
        "auroc",
        "auprc",
        "ece",
    ]:

        store["s4"][metric].append(
            s4_b[metric]
        )

        store["fusion"][metric].append(
            fusion_b[metric]
        )

        store[
            "delta_fusion_minus_s4"
        ][metric].append(
            fusion_b[metric]
            - s4_b[metric]
        )


# ============================================================
# Percentile CIs
# ============================================================

def summarise(values):
    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(
                arr.mean()
            ),

        "lower_95":
            float(
                np.percentile(
                    arr,
                    2.5,
                )
            ),

        "upper_95":
            float(
                np.percentile(
                    arr,
                    97.5,
                )
            ),
    }


bootstrap = {}

for condition, d in store.items():

    bootstrap[condition] = {
        metric:
            summarise(values)
        for metric, values
        in d.items()
    }


print(
    "\n" + "=" * 80
)

print(
    f"95% STRATIFIED BOOTSTRAP CIs "
    f"({N_BOOT} resamples)"
)

print("=" * 80)


for model in [
    "s4",
    "fusion",
]:

    print(
        f"\n{model.upper()}"
    )

    for metric in [
        "f1",
        "auroc",
        "auprc",
        "ece",
    ]:

        r = bootstrap[
            model
        ][metric]

        print(
            f"  {metric.upper():<6} "
            f"{point[model][metric]:.4f} "
            f"[{r['lower_95']:.4f}, "
            f"{r['upper_95']:.4f}]"
        )


print(
    "\nPAIRED DELTA: "
    "FUSION - S4"
)

for metric in [
    "f1",
    "auroc",
    "auprc",
    "ece",
]:

    r = bootstrap[
        "delta_fusion_minus_s4"
    ][metric]

    point_delta = (
        point["fusion"][metric]
        - point["s4"][metric]
    )

    print(
        f"  {metric.upper():<6} "
        f"{point_delta:+.4f} "
        f"[{r['lower_95']:+.4f}, "
        f"{r['upper_95']:+.4f}]"
    )


output = {
    "protocol": {
        "dataset":
            "RAGTruth test",

        "n":
            len(test),

        "n_bootstrap":
            N_BOOT,

        "bootstrap":
            "stratified nonparametric paired bootstrap",

        "ci":
            "95% percentile",

        "thresholds_fixed":
            {
                "s4": S4_THRESHOLD,
                "fusion": FUSION_THRESHOLD,
            },

        "important_note":
            "Bootstrap captures test-sample uncertainty for fixed trained models; it does not capture model-training uncertainty.",
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
