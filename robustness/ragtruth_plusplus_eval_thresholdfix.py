import json
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

OUT = "/workspace/ragtruth_plusplus_results_thresholdfix.json"

S2_MIN, S2_MAX = -11.430, 10.641

def norm_s2(val):
    return float(
        max(
            0.0,
            min(
                1.0,
                (val - S2_MIN) / (S2_MAX - S2_MIN)
            )
        )
    )


def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

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

        ece += (
            mask.sum()
            * abs(
                labels[mask].mean()
                - probs[mask].mean()
            )
        )

    return float(ece / len(probs))


def evaluate(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "f1": round(
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),
            4
        ),
        "precision": round(
            float(
                precision_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),
            4
        ),
        "recall": round(
            float(
                recall_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),
            4
        ),
        "auroc": round(
            float(
                roc_auc_score(
                    labels,
                    scores
                )
            ),
            4
        ),
        "auprc": round(
            float(
                average_precision_score(
                    labels,
                    scores
                )
            ),
            4
        ),
        "ece": round(
            compute_ece(
                scores,
                labels
            ),
            4
        ),
        "n": int(len(labels)),
        "pos_rate": round(
            float(labels.mean()),
            4
        ),
    }


# ============================================================
# RAGTruth++
# ============================================================

print("Loading RAGTruth++...")

messages = pd.read_csv(
    "hf://datasets/blue-guardrails/ragtruth-plus-plus/messages.csv"
)

spans = pd.read_csv(
    "hf://datasets/blue-guardrails/ragtruth-plus-plus/hallucination_spans.csv"
)

messages["meta_parsed"] = messages["meta"].apply(
    json.loads
)

assistant = messages[
    messages["role"] == "assistant"
].copy()

hall_ids = set(
    spans["message_stable_id"].unique()
)

assistant["rtp_label"] = (
    assistant["stable_id"]
    .isin(hall_ids)
    .astype(int)
)


# ============================================================
# Original RAGTruth test
# ============================================================

print("Loading RAGTruth test...")

ds_test = load_dataset(
    "wandb/RAGTruth-processed",
    split="test"
)

id_to_idx = {}

for idx, ex in enumerate(ds_test):
    rt_id = str(ex["id"])

    if rt_id in id_to_idx:
        raise RuntimeError(
            f"Duplicate RAGTruth test ID: {rt_id}"
        )

    id_to_idx[rt_id] = idx


# ============================================================
# Saved verifier scores — TEST ONLY
# ============================================================

def load_score_map(path):
    with open(path) as f:
        return {
            int(r["idx"]): r
            for r in json.load(f)
        }


s1 = load_score_map(
    "/workspace/nli_results_test_v2.json"
)

s2 = load_score_map(
    "/workspace/relevance_results_test_v2.json"
)

s4 = load_score_map(
    "/workspace/signal4_results_test.json"
)

mc = load_score_map(
    "/workspace/minicheck_results_test_7b.json"
)


# ============================================================
# Exact ID alignment
# ============================================================

aligned = []

for pp_row_idx, row in assistant.iterrows():

    meta = row["meta_parsed"]

    original_id = str(
        meta["original_id"]
    )

    if original_id not in id_to_idx:
        raise RuntimeError(
            f"original_id not found in RAGTruth test: {original_id}"
        )

    idx = id_to_idx[original_id]
    rt = ds_test[idx]

    # Strong integrity checks
    assert str(rt["output"]) == str(row["text"])
    assert str(rt["model"]) == str(meta["model"])

    if (
        idx not in s1
        or idx not in s2
        or idx not in s4
        or idx not in mc
    ):
        raise RuntimeError(
            f"Missing saved score for RAGTruth test idx={idx}"
        )

    r1 = s1[idx]
    r2 = s2[idx]
    r4 = s4[idx]
    rmc = mc[idx]

    vals = [
        r1["nli_score"],
        r2["raw_min_relevance"],
        r4["signal4_score"],
        rmc["minicheck_score"],
    ]

    if any(v is None for v in vals):
        raise RuntimeError(
            f"Null score for idx={idx}"
        )

    aligned.append({
        "pp_row": int(pp_row_idx),
        "stable_id": str(row["stable_id"]),
        "original_id": original_id,
        "rt_idx": int(idx),

        "rt_label": int(
            r1["ground_truth_hallucination"]
        ),

        "rtp_label": int(
            row["rtp_label"]
        ),

        "s1": 1.0 - float(
            r1["nli_score"]
        ),

        "s2": 1.0 - norm_s2(
            float(
                r2["raw_min_relevance"]
            )
        ),

        "s4": float(
            r4["signal4_score"]
        ),

        "mc": 1.0 - float(
            rmc["minicheck_score"]
        ),
    })


assert len(aligned) == 408

print(f"Aligned exactly: {len(aligned)}")


# ============================================================
# Arrays
# ============================================================

rt_labels = np.array([
    x["rt_label"]
    for x in aligned
])

pp_labels = np.array([
    x["rtp_label"]
    for x in aligned
])

s1_scores = np.array([
    x["s1"]
    for x in aligned
])

s2_scores = np.array([
    x["s2"]
    for x in aligned
])

s4_scores = np.array([
    x["s4"]
    for x in aligned
])

mc_scores = np.array([
    x["mc"]
    for x in aligned
])

simple_avg = (
    s2_scores + s4_scores
) / 2.0


print(
    "Original RAGTruth positive rate:",
    f"{rt_labels.mean():.4f}"
)

print(
    "RAGTruth++ positive rate:",
    f"{pp_labels.mean():.4f}"
)

print(
    "Label changes:",
    int(
        (rt_labels != pp_labels).sum()
    ),
    "/",
    len(pp_labels)
)


# ============================================================
# Same fixed RAGTruth-trained operating points
# as original detector-sensitivity analysis
# ============================================================

METHODS = {
    "S1": (
        s1_scores,
        0.35
    ),

    "S2": (
        s2_scores,
        0.45
    ),

    "S4": (
        s4_scores,
        0.55
    ),

    "MiniCheck-7B": (
        mc_scores,
        0.80
    ),

    "Simple average S2+S4": (
        simple_avg,
        0.55
    ),
}


results = {}

print("\n" + "=" * 90)
print("CORRECTED RAGTRUTH++ RESULTS — ID-BASED ALIGNMENT")
print("=" * 90)

print(
    f"{'Method':<24}"
    f"{'F1':>8}"
    f"{'P':>8}"
    f"{'R':>8}"
    f"{'AUROC':>9}"
    f"{'AUPRC':>9}"
    f"{'ECE':>8}"
)

for name, (scores, threshold) in METHODS.items():

    m = evaluate(
        scores,
        pp_labels,
        threshold
    )

    results[name] = m

    print(
        f"{name:<24}"
        f"{m['f1']:>8.4f}"
        f"{m['precision']:>8.4f}"
        f"{m['recall']:>8.4f}"
        f"{m['auroc']:>9.4f}"
        f"{m['auprc']:>9.4f}"
        f"{m['ece']:>8.4f}"
    )


output = {
    "matching": {
        "method":
            "exact RAGTruth++ meta.original_id -> RAGTruth test id",
        "n": len(aligned),
        "all_from_ragtruth_test": True,
        "text_model_integrity_checks": True,
    },

    "label_comparison": {
        "ragtruth_positive_rate":
            round(float(rt_labels.mean()), 4),

        "ragtruth_plus_plus_positive_rate":
            round(float(pp_labels.mean()), 4),

        "n_label_changes":
            int(
                (rt_labels != pp_labels).sum()
            ),
    },

    "results": results,

    "aligned_examples": aligned,
}


with open(
    OUT,
    "w"
) as f:
    json.dump(
        output,
        f,
        indent=2
    )


print("\nSaved:", OUT)
