import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# Paths
# ============================================================

REL_TRAIN = "/workspace/relevance_results_train_v2.json"
REL_TEST  = "/workspace/relevance_results_test_v2.json"
S4_TRAIN  = "/workspace/signal4_results_train_oof.json"
S4_TEST   = "/workspace/signal4_results_test.json"

OUT_PATH = "/workspace/fusion_decomposition_review_results.json"

S2_MIN = -11.430
S2_MAX = 10.641

THRESHOLDS = np.arange(0.05, 0.96, 0.05)

# ============================================================
# Utilities
# ============================================================

def norm_s2(v):
    return float(
        max(
            0.0,
            min(
                1.0,
                (v - S2_MIN) / (S2_MAX - S2_MIN)
            )
        )
    )


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probs >= bins[i])
                & (probs <= bins[i + 1])
            )
        else:
            mask = (
                (probs >= bins[i])
                & (probs < bins[i + 1])
            )

        if mask.sum() == 0:
            continue

        acc = labels[mask].mean()
        conf = probs[mask].mean()

        total += (
            mask.sum()
            * abs(acc - conf)
        )

    return float(total / len(labels))


def best_threshold(probs, labels):
    best_t = 0.5
    best_f1 = -1.0

    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)

        score = f1_score(
            labels,
            preds,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, float(best_f1)


def evaluate(y, probs, threshold):
    preds = (
        probs >= threshold
    ).astype(int)

    return {
        "threshold": round(float(threshold), 4),
        "f1": round(
            float(f1_score(
                y,
                preds,
                zero_division=0
            )),
            4
        ),
        "precision": round(
            float(precision_score(
                y,
                preds,
                zero_division=0
            )),
            4
        ),
        "recall": round(
            float(recall_score(
                y,
                preds,
                zero_division=0
            )),
            4
        ),
        "auroc": round(
            float(roc_auc_score(
                y,
                probs
            )),
            4
        ),
        "auprc": round(
            float(average_precision_score(
                y,
                probs
            )),
            4
        ),
        "ece": round(
            float(ece(
                probs,
                y
            )),
            4
        ),
    }


# ============================================================
# Load / align data
# ============================================================

with open(REL_TRAIN) as f:
    rel_train = {
        int(x["idx"]): x
        for x in json.load(f)
    }

with open(REL_TEST) as f:
    rel_test = {
        int(x["idx"]): x
        for x in json.load(f)
    }

with open(S4_TRAIN) as f:
    s4_train = {
        int(x["idx"]): x
        for x in json.load(f)
    }

with open(S4_TEST) as f:
    s4_test = {
        int(x["idx"]): x
        for x in json.load(f)
    }


def build_rows(rel_map, s4_map):
    rows = []

    common = sorted(
        rel_map.keys()
        & s4_map.keys()
    )

    for idx in common:
        r2 = rel_map[idx]
        r4 = s4_map[idx]

        raw_s2 = r2.get(
            "raw_min_relevance"
        )

        s4_score = r4.get(
            "signal4_score"
        )

        if raw_s2 is None or s4_score is None:
            continue

        y2 = int(
            r2["ground_truth_hallucination"]
        )

        y4 = int(
            r4["ground_truth_hallucination"]
        )

        assert y2 == y4

        rows.append({
            "idx": idx,
            "label": y2,
            "s2": norm_s2(
                float(raw_s2)
            ),
            "s4": float(s4_score),
            "task": r2["task_type"],
            "model": r2["model"],
        })

    return rows


train_rows = build_rows(
    rel_train,
    s4_train
)

test_rows = build_rows(
    rel_test,
    s4_test
)

print(
    f"Train examples: {len(train_rows)}"
)
print(
    f"Test examples : {len(test_rows)}"
)

y_train = np.array(
    [x["label"] for x in train_rows],
    dtype=int
)

y_test = np.array(
    [x["label"] for x in test_rows],
    dtype=int
)

print(
    f"Train positive rate: {y_train.mean():.4f}"
)
print(
    f"Test positive rate : {y_test.mean():.4f}"
)


# ============================================================
# Model construction
# ============================================================

VARIANTS = {
    "S4_logistic": {
        "numeric": ["s4"],
        "metadata": False,
    },

    "S4_metadata": {
        "numeric": ["s4"],
        "metadata": True,
    },

    "S2_S4": {
        "numeric": ["s2", "s4"],
        "metadata": False,
    },

    "S2_S4_metadata": {
        "numeric": ["s2", "s4"],
        "metadata": True,
    },
}


def make_features(
    rows,
    numeric_names,
    use_metadata,
    encoder=None,
    fit_encoder=False,
):
    numeric = np.array([
        [
            row[name]
            for name in numeric_names
        ]
        for row in rows
    ])

    if not use_metadata:
        return numeric, encoder

    cats = [
        [
            row["task"],
            row["model"]
        ]
        for row in rows
    ]

    if fit_encoder:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded = encoder.fit_transform(
            cats
        )

    else:
        encoded = encoder.transform(
            cats
        )

    X = np.hstack([
        numeric,
        encoded
    ])

    return X, encoder


# ============================================================
# For every variant:
#
# A) reproduce current protocol:
#    fit on full train
#    threshold on fitted train predictions
#
# B) corrected threshold protocol:
#    5-fold OOF meta-model predictions
#    choose threshold on OOF probabilities
#
# Final test probabilities always come from model fitted
# on the complete training set.
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

all_results = {}

print("\n" + "=" * 108)
print("FUSION DECOMPOSITION / META-THRESHOLD DIAGNOSTIC")
print("=" * 108)

for variant_name, cfg in VARIANTS.items():

    numeric_names = cfg["numeric"]
    use_metadata = cfg["metadata"]

    # --------------------------------------------------------
    # 1. Generate OOF predictions of the META model
    # --------------------------------------------------------

    oof_prob = np.zeros(
        len(train_rows),
        dtype=float
    )

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(
            np.zeros(len(y_train)),
            y_train
        ),
        start=1
    ):

        fold_train = [
            train_rows[i]
            for i in tr_idx
        ]

        fold_val = [
            train_rows[i]
            for i in va_idx
        ]

        X_tr, enc = make_features(
            fold_train,
            numeric_names,
            use_metadata,
            encoder=None,
            fit_encoder=use_metadata,
        )

        X_va, _ = make_features(
            fold_val,
            numeric_names,
            use_metadata,
            encoder=enc,
            fit_encoder=False,
        )

        clf_fold = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        clf_fold.fit(
            X_tr,
            y_train[tr_idx]
        )

        oof_prob[va_idx] = (
            clf_fold.predict_proba(
                X_va
            )[:, 1]
        )

    oof_threshold, oof_train_f1 = (
        best_threshold(
            oof_prob,
            y_train
        )
    )

    # --------------------------------------------------------
    # 2. Fit final meta model on all train data
    # --------------------------------------------------------

    X_train_full, enc_full = make_features(
        train_rows,
        numeric_names,
        use_metadata,
        encoder=None,
        fit_encoder=use_metadata,
    )

    X_test_full, _ = make_features(
        test_rows,
        numeric_names,
        use_metadata,
        encoder=enc_full,
        fit_encoder=False,
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    clf.fit(
        X_train_full,
        y_train
    )

    # --------------------------------------------------------
    # Existing / old train-fit threshold
    # --------------------------------------------------------

    fitted_train_prob = (
        clf.predict_proba(
            X_train_full
        )[:, 1]
    )

    old_threshold, old_train_f1 = (
        best_threshold(
            fitted_train_prob,
            y_train
        )
    )

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    test_prob = (
        clf.predict_proba(
            X_test_full
        )[:, 1]
    )

    old_metrics = evaluate(
        y_test,
        test_prob,
        old_threshold
    )

    oof_metrics = evaluate(
        y_test,
        test_prob,
        oof_threshold
    )

    all_results[
        variant_name
    ] = {
        "features": {
            "numeric": numeric_names,
            "metadata": use_metadata,
        },

        "old_train_fit_threshold": {
            "train_threshold": round(
                old_threshold,
                4
            ),
            "train_f1": round(
                old_train_f1,
                4
            ),
            "test_metrics": old_metrics,
        },

        "meta_oof_threshold": {
            "train_threshold": round(
                oof_threshold,
                4
            ),
            "oof_train_f1": round(
                oof_train_f1,
                4
            ),
            "test_metrics": oof_metrics,
        },
    }

    print(
        f"\n{variant_name}"
    )

    print(
        "  OLD threshold "
        f"{old_threshold:.2f} -> "
        f"Test F1={old_metrics['f1']:.4f}"
    )

    print(
        "  OOF threshold "
        f"{oof_threshold:.2f} -> "
        f"Test F1={oof_metrics['f1']:.4f}"
    )

    print(
        "  Ranking/calibration -> "
        f"AUROC={oof_metrics['auroc']:.4f} | "
        f"AUPRC={oof_metrics['auprc']:.4f} | "
        f"ECE={oof_metrics['ece']:.4f}"
    )


# ============================================================
# Compact final table
# ============================================================

print("\n" + "=" * 108)
print("FINAL TABLE USING META-OOF THRESHOLD")
print("=" * 108)

print(
    f"{'Variant':<22}"
    f"{'Thr':>7}"
    f"{'F1':>8}"
    f"{'Prec':>8}"
    f"{'Rec':>8}"
    f"{'AUROC':>9}"
    f"{'AUPRC':>9}"
    f"{'ECE':>8}"
)

for name in VARIANTS:
    m = all_results[
        name
    ][
        "meta_oof_threshold"
    ][
        "test_metrics"
    ]

    print(
        f"{name:<22}"
        f"{m['threshold']:>7.2f}"
        f"{m['f1']:>8.4f}"
        f"{m['precision']:>8.4f}"
        f"{m['recall']:>8.4f}"
        f"{m['auroc']:>9.4f}"
        f"{m['auprc']:>9.4f}"
        f"{m['ece']:>8.4f}"
    )


# ============================================================
# Save
# ============================================================

output = {
    "purpose": (
        "Supervisor-review fusion decomposition and "
        "meta-model threshold diagnostic"
    ),
    "protocol": {
        "s4_train_source":
            "signal4_results_train_oof.json",
        "s4_test_source":
            "signal4_results_test.json",
        "meta_threshold":
            "5-fold stratified out-of-fold predictions "
            "of logistic-regression meta-model",
        "threshold_grid":
            "0.05 to 0.95 inclusive, step 0.05",
        "random_state": 42,
    },
    "results": all_results,
}

with open(
    OUT_PATH,
    "w"
) as f:
    json.dump(
        output,
        f,
        indent=2
    )

print(
    f"\nSaved to {OUT_PATH}"
)
