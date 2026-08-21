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

S2_MIN = -11.430
S2_MAX = 10.641

SEED = 42
N_FOLDS = 5

OUT = "/workspace/leave_one_strict_review_results.json"


def norm_s2(x):
    return float(
        max(
            0.0,
            min(
                1.0,
                (x - S2_MIN) / (S2_MAX - S2_MIN),
            ),
        )
    )


def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bins = np.linspace(0, 1, n_bins + 1)
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

        total += (
            mask.sum()
            * abs(
                labels[mask].mean()
                - probs[mask].mean()
            )
        )

    return float(total / len(labels))


def metrics(labels, probs, threshold):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    preds = (
        probs >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()),
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
                probs,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                probs,
            )
        ),
        "ece": ece(
            probs,
            labels,
        ),
    }


def best_threshold(labels, probs):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(
        0.05,
        0.951,
        0.05,
    ):
        p = (
            np.asarray(probs) >= t
        ).astype(int)

        score = f1_score(
            labels,
            p,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def load_map(path):
    with open(path) as f:
        return {
            int(r["idx"]): r
            for r in json.load(f)
        }


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


def build_examples(
    s1,
    s2,
    s4,
):
    common = sorted(
        set(s1)
        & set(s2)
        & set(s4)
    )

    out = []

    for idx in common:
        a = s1[idx]
        b = s2[idx]
        c = s4[idx]

        if (
            b["raw_min_relevance"] is None
            or c["signal4_score"] is None
        ):
            continue

        out.append({
            "idx": idx,
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
                    c["signal4_score"]
                ),
        })

    return out


train_examples = build_examples(
    s1_train,
    s2_train,
    s4_train,
)

test_examples = build_examples(
    s1_test,
    s2_test,
    s4_test,
)


print(
    "RAGTruth train:",
    len(train_examples)
)

print(
    "RAGTruth test :",
    len(test_examples)
)


# ------------------------------------------------------------
# Fit one fusion configuration
# ------------------------------------------------------------

def matrices(
    rows,
    with_metadata,
    ohe=None,
    fit_ohe=False,
):
    base = np.array([
        [r["s2"], r["s4"]]
        for r in rows
    ])

    if not with_metadata:
        return base, None

    cats = np.array([
        [r["task"], r["model"]]
        for r in rows
    ])

    if fit_ohe:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded = ohe.fit_transform(
            cats
        )
    else:
        encoded = ohe.transform(
            cats
        )

    return (
        np.hstack([
            base,
            encoded,
        ]),
        ohe,
    )


def fit_evaluate(
    train_rows,
    test_rows,
    with_metadata,
):
    y_train = np.array([
        r["label"]
        for r in train_rows
    ])

    y_test = np.array([
        r["label"]
        for r in test_rows
    ])

    # --------------------------------------------------------
    # OOF fusion predictions solely for threshold selection.
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(
        len(train_rows),
        dtype=float,
    )

    train_rows_arr = np.array(
        train_rows,
        dtype=object,
    )

    for tr_idx, va_idx in skf.split(
        np.zeros(len(train_rows)),
        y_train,
    ):
        fold_train = (
            train_rows_arr[tr_idx].tolist()
        )

        fold_val = (
            train_rows_arr[va_idx].tolist()
        )

        X_tr, fold_ohe = matrices(
            fold_train,
            with_metadata,
            fit_ohe=with_metadata,
        )

        X_va, _ = matrices(
            fold_val,
            with_metadata,
            ohe=fold_ohe,
            fit_ohe=False,
        )

        clf = LogisticRegression(
            max_iter=1000,
            random_state=SEED,
        )

        clf.fit(
            X_tr,
            y_train[tr_idx],
        )

        oof[va_idx] = (
            clf.predict_proba(
                X_va
            )[:, 1]
        )

    threshold = best_threshold(
        y_train,
        oof,
    )

    # --------------------------------------------------------
    # Final fusion model on all allowed original train rows.
    # --------------------------------------------------------

    X_train, final_ohe = matrices(
        train_rows,
        with_metadata,
        fit_ohe=with_metadata,
    )

    X_test, _ = matrices(
        test_rows,
        with_metadata,
        ohe=final_ohe,
        fit_ohe=False,
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    clf.fit(
        X_train,
        y_train,
    )

    test_probs = (
        clf.predict_proba(
            X_test
        )[:, 1]
    )

    return metrics(
        y_test,
        test_probs,
        threshold,
    )


# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------

def run_leave_one(
    field,
):
    values = sorted(
        set(
            r[field]
            for r in train_examples
            + test_examples
        )
    )

    results = {}

    print(
        "\n" + "=" * 105
    )

    print(
        "STRICT LEAVE-ONE-"
        + field.upper()
        + "-OUT FUSION TRANSFER"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Held out':<28}"
        f"{'Setting':<12}"
        f"{'N train':>9}"
        f"{'N test':>8}"
        f"{'F1':>9}"
        f"{'AUROC':>10}"
        f"{'AUPRC':>10}"
        f"{'ECE':>10}"
        f"{'Thr':>8}"
    )

    print("-" * 105)

    for held_out in values:

        # STRICT:
        # original train only for fitting fusion
        train_rows = [
            r
            for r in train_examples
            if r[field] != held_out
        ]

        # original test only for evaluation
        test_rows = [
            r
            for r in test_examples
            if r[field] == held_out
        ]

        if not test_rows:
            continue

        results[held_out] = {}

        for name, metadata in [
            ("no_meta", False),
            ("metadata", True),
        ]:
            m = fit_evaluate(
                train_rows,
                test_rows,
                metadata,
            )

            results[
                held_out
            ][name] = m

            results[
                held_out
            ]["n_train"] = len(
                train_rows
            )

            print(
                f"{held_out:<28}"
                f"{name:<12}"
                f"{len(train_rows):>9}"
                f"{m['n']:>8}"
                f"{m['f1']:>9.4f}"
                f"{m['auroc']:>10.4f}"
                f"{m['auprc']:>10.4f}"
                f"{m['ece']:>10.4f}"
                f"{m['threshold']:>8.2f}"
            )

    return results


task_results = run_leave_one(
    "task"
)

generator_results = run_leave_one(
    "model"
)


output = {
    "protocol": {
        "fusion_train":
            "original RAGTruth train only, excluding held-out group",

        "evaluation":
            "original RAGTruth test only, held-out group",

        "s4_caveat":
            "S4 itself is not retrained with the held-out group removed; this measures fusion-layer transfer, not end-to-end unseen-group generalization",

        "threshold":
            "selected from 5-fold OOF fusion predictions on allowed training examples",
    },

    "leave_one_task":
        task_results,

    "leave_one_generator":
        generator_results,
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
