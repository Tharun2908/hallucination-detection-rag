import json
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

# ------------------------------------------------------------
# Canonical RAGTruth-trained metadata-free S2+S4 LogReg
# ------------------------------------------------------------

COEF_S2_SUPPORT = -1.359009650504923
COEF_S4 = 3.1555282346796876
INTERCEPT = -1.4168645142264538

FUSION_THRESHOLD = 0.45

# Corrected fixed-8k MiniCheck reference used this threshold
MC_THRESHOLD = 0.80

ESCALATION_RATES = [0, 5, 10, 20, 30, 50, 75, 100]

FINAL_S2S4_PATH = "/workspace/halubench_final_s2s4_scores.json"
MC_PATH = "/workspace/halubench_per_example_scores.json"
SPLIT_PATH = "/workspace/halubench_curve/test_train_pool_indices.json"

OUT_PATH = "/workspace/halubench_cascade_fixed8k_results.json"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])

        if mask.sum() == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()

        ece += (
            mask.sum()
            * abs(bin_acc - bin_conf)
        )

    return float(ece / len(probs))


def metrics(labels, scores, preds):
    return {
        "f1": float(
            f1_score(labels, preds, zero_division=0)
        ),
        "precision": float(
            precision_score(labels, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, zero_division=0)
        ),
        "auroc": float(
            roc_auc_score(labels, scores)
        ),
        "auprc": float(
            average_precision_score(labels, scores)
        ),
        "ece": float(
            compute_ece(scores, labels)
        ),
    }


# ------------------------------------------------------------
# Load FINAL sentence-level S2/S4
# ------------------------------------------------------------

with open(FINAL_S2S4_PATH) as f:
    final_rows = json.load(f)

# Old file is used ONLY for cached MiniCheck scores
with open(MC_PATH) as f:
    mc_rows = json.load(f)

with open(SPLIT_PATH) as f:
    split = json.load(f)


print(f"Final S2/S4 examples : {len(final_rows)}")
print(f"Cached MC examples   : {len(mc_rows)}")
print(f"Fixed test indices   : {len(split['test_idx'])}")

assert len(final_rows) == 14000
assert len(mc_rows) == 14000
assert len(split["test_idx"]) == 8000


# ------------------------------------------------------------
# Align files by idx
# ------------------------------------------------------------

final_by_idx = {
    int(r["idx"]): r
    for r in final_rows
}

mc_by_idx = {
    int(r["idx"]): r
    for r in mc_rows
}

common = sorted(
    final_by_idx.keys()
    & mc_by_idx.keys()
)

assert len(common) == 14000

# Verify that old file and final file refer to same examples.
for idx in common:
    a = final_by_idx[idx]
    b = mc_by_idx[idx]

    if (
        int(a["label"]) != int(b["label"])
        or a["source"] != b["source"]
    ):
        raise RuntimeError(
            f"Alignment mismatch at idx={idx}"
        )

print("S2/S4 and MiniCheck alignment: PASSED")


# ------------------------------------------------------------
# Reconstruct FINAL metadata-free fusion on all 14k
#
# IMPORTANT:
# use s2_support DIRECTLY from final sentence-level file.
# ------------------------------------------------------------

labels_all = np.array([
    int(final_by_idx[i]["label"])
    for i in common
])

s2_support_all = np.array([
    float(final_by_idx[i]["s2_support"])
    for i in common
])

s4_all = np.array([
    float(final_by_idx[i]["s4_score"])
    for i in common
])

mc_all = np.array([
    float(mc_by_idx[i]["mc_hall"])
    for i in common
])

logits_all = (
    INTERCEPT
    + COEF_S2_SUPPORT * s2_support_all
    + COEF_S4 * s4_all
)

fusion_prob_all = sigmoid(logits_all)


# ------------------------------------------------------------
# Full-14k self-check
# Canonical Chapter 5 result:
#
# F1    ≈ 0.3647
# AUROC ≈ 0.5324
# AUPRC ≈ 0.5127
# ------------------------------------------------------------

fusion_preds_all = (
    fusion_prob_all >= FUSION_THRESHOLD
).astype(int)

full_metrics = metrics(
    labels_all,
    fusion_prob_all,
    fusion_preds_all,
)

print("\n" + "=" * 78)
print("FULL 14K FINAL-FUSION SELF-CHECK")
print("=" * 78)

print(
    f"F1    = {full_metrics['f1']:.4f} "
    f"expected ≈ 0.3647"
)

print(
    f"AUROC = {full_metrics['auroc']:.4f} "
    f"expected ≈ 0.5324"
)

print(
    f"AUPRC = {full_metrics['auprc']:.4f} "
    f"expected ≈ 0.5127"
)

if abs(full_metrics["f1"] - 0.3647) > 0.002:
    raise RuntimeError(
        "Final 14k fusion F1 self-check failed."
    )

if abs(full_metrics["auroc"] - 0.5324) > 0.002:
    raise RuntimeError(
        "Final 14k fusion AUROC self-check failed."
    )

print("FINAL FUSION SELF-CHECK PASSED")


# ------------------------------------------------------------
# Exact fixed 8k test subset
# ------------------------------------------------------------

test_ids = [
    int(i)
    for i in split["test_idx"]
]

# common ordering above is simply idx 0..13999,
# but use explicit map for safety.
position = {
    idx: pos
    for pos, idx in enumerate(common)
}

test_pos = np.array([
    position[i]
    for i in test_ids
])

labels = labels_all[test_pos]
fusion_prob = fusion_prob_all[test_pos]
mc_scores = mc_all[test_pos]

assert len(labels) == 8000

print(
    "\nFixed-8k positive rate:",
    f"{labels.mean():.6f}"
)


# ------------------------------------------------------------
# Endpoint metrics
# ------------------------------------------------------------

fusion_preds = (
    fusion_prob >= FUSION_THRESHOLD
).astype(int)

fusion_endpoint = metrics(
    labels,
    fusion_prob,
    fusion_preds,
)

mc_preds = (
    mc_scores >= MC_THRESHOLD
).astype(int)

mc_endpoint = metrics(
    labels,
    mc_scores,
    mc_preds,
)


print("\n" + "=" * 78)
print("FIXED-8K ENDPOINT CHECKS")
print("=" * 78)

print("\nr=0 — RAGTruth-trained metadata-free S2+S4")
print(
    f"F1={fusion_endpoint['f1']:.4f} | "
    f"AUROC={fusion_endpoint['auroc']:.4f} | "
    f"AUPRC={fusion_endpoint['auprc']:.4f} | "
    f"ECE={fusion_endpoint['ece']:.4f}"
)

print("\nr=100 — MiniCheck-7B")
print(
    f"F1={mc_endpoint['f1']:.4f} | "
    f"AUROC={mc_endpoint['auroc']:.4f} | "
    f"AUPRC={mc_endpoint['auprc']:.4f} | "
    f"ECE={mc_endpoint['ece']:.4f}"
)

print(
    "\nExpected MiniCheck fixed-8k: "
    "F1=0.7275 | AUROC=0.8021 | AUPRC=0.8365"
)

if abs(mc_endpoint["f1"] - 0.7275) > 0.001:
    raise RuntimeError(
        "MiniCheck fixed-8k F1 check failed."
    )

if abs(mc_endpoint["auroc"] - 0.8021) > 0.001:
    raise RuntimeError(
        "MiniCheck fixed-8k AUROC check failed."
    )

print("MINICHECK ENDPOINT CHECK PASSED")


# ------------------------------------------------------------
# Cascade
#
# Same routing rule as Chapter 3 / RAGTruth:
#
# uncertainty = |p_fusion - 0.5|
#
# smallest values escalated first.
# ------------------------------------------------------------

uncertainty = np.abs(
    fusion_prob - 0.5
)

results = []

print("\n" + "=" * 92)
print("HALUBENCH FIXED-8K CASCADE")
print("=" * 92)

for rate in ESCALATION_RATES:

    n_escalate = int(
        len(labels) * rate / 100
    )

    escalate_mask = np.zeros(
        len(labels),
        dtype=bool
    )

    if n_escalate > 0:
        ranked = np.argsort(
            uncertainty
        )

        escalate_mask[
            ranked[:n_escalate]
        ] = True

    # Continuous replacement score
    cascade_scores = fusion_prob.copy()

    cascade_scores[
        escalate_mask
    ] = mc_scores[
        escalate_mask
    ]

    # Binary decision:
    # each verifier keeps its own decision threshold.
    preds = (
        fusion_prob >= FUSION_THRESHOLD
    ).astype(int)

    preds[
        escalate_mask
    ] = (
        mc_scores[
            escalate_mask
        ] >= MC_THRESHOLD
    ).astype(int)

    m = metrics(
        labels,
        cascade_scores,
        preds,
    )

    cost = (
        1.0
        + 10.0 * rate / 100.0
    )

    result = {
        "escalation_rate": rate,
        "n_escalated": n_escalate,
        "cost": float(cost),
        **m,
    }

    results.append(result)

    print(
        f"{rate:3d}% | "
        f"F1={m['f1']:.4f} | "
        f"AUROC={m['auroc']:.4f} | "
        f"AUPRC={m['auprc']:.4f} | "
        f"ECE={m['ece']:.4f} | "
        f"Cost={cost:.1f}x"
    )


# ------------------------------------------------------------
# Analyze result
# ------------------------------------------------------------

best = max(
    results,
    key=lambda x: x["f1"]
)

print("\n" + "=" * 78)
print("BEST OBSERVED F1")
print("=" * 78)

print(
    f"Escalation = {best['escalation_rate']}%\n"
    f"F1         = {best['f1']:.4f}\n"
    f"AUROC      = {best['auroc']:.4f}\n"
    f"AUPRC      = {best['auprc']:.4f}\n"
    f"ECE        = {best['ece']:.4f}\n"
    f"Cost       = {best['cost']:.1f}x"
)


f1_values = [
    r["f1"]
    for r in results
]

non_decreasing = all(
    f1_values[i] <= f1_values[i + 1]
    for i in range(len(f1_values) - 1)
)

print(
    "\nF1 non-decreasing across tested rates:",
    non_decreasing
)

print("\nF1 step changes:")

for a, b in zip(
    results[:-1],
    results[1:]
):
    print(
        f"{a['escalation_rate']:3d}% -> "
        f"{b['escalation_rate']:3d}% : "
        f"{b['f1'] - a['f1']:+.4f}"
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output = {
    "experiment": (
        "Cross-domain cascade on fixed 8000-example "
        "HaluBench test split"
    ),
    "n_test": 8000,
    "positive_rate": float(labels.mean()),
    "fusion": {
        "training_domain": "RAGTruth",
        "metadata_free": True,
        "coef_s2_support": COEF_S2_SUPPORT,
        "coef_s4": COEF_S4,
        "intercept": INTERCEPT,
        "threshold": FUSION_THRESHOLD,
        "s2_source": (
            "final sentence-level minimum relevance"
        ),
    },
    "minicheck_threshold": MC_THRESHOLD,
    "routing": (
        "ascending abs(fusion_probability - 0.5)"
    ),
    "cost_model": "1 + 10*r",
    "full14k_final_fusion_self_check": full_metrics,
    "fixed8k_fusion_endpoint": fusion_endpoint,
    "fixed8k_minicheck_endpoint": mc_endpoint,
    "results": results,
    "best_f1_operating_point": best,
    "f1_non_decreasing": non_decreasing,
}

with open(OUT_PATH, "w") as f:
    json.dump(
        output,
        f,
        indent=2
    )

print(f"\nSaved to {OUT_PATH}")
