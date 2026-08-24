import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# ECE DEFINITIONS
# ============================================================

def canonical_ece(scores, labels, n_bins=10):
    """
    Canonical thesis ECE:
      - score = P(hallucination)
      - 10 equal-width bins
      - compare mean predicted hallucination probability
        against empirical hallucination rate
      - last bin includes score == 1.0
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
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

    return float(total / len(labels))


def legacy_ece(scores, labels, n_bins=10):
    """
    Older implementation:
    upper bound exclusive for EVERY bin,
    so exact score == 1.0 is omitted.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
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

    return float(total / len(labels))


audit_rows = []


def record(section, name, scores, labels, reported):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    can = canonical_ece(scores, labels)
    old = legacy_ece(scores, labels)

    exact_one = int(
        np.sum(scores == 1.0)
    )

    exact_zero = int(
        np.sum(scores == 0.0)
    )

    audit_rows.append({
        "section": section,
        "method": name,
        "n": int(len(labels)),
        "reported_ece": float(reported),
        "canonical_ece": can,
        "legacy_ece": old,
        "canonical_minus_reported":
            can - float(reported),
        "canonical_minus_legacy":
            can - old,
        "exact_score_1": exact_one,
        "exact_score_0": exact_zero,
    })


def load(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# TABLE 4.1
# ============================================================

print("=" * 100)
print("A. TABLE 4.1 STANDALONE METHODS")
print("=" * 100)


# S1
d = load("/workspace/nli_results_test_v2.json")

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    1.0 - float(r["nli_score"])
    for r in d
])

record(
    "Table 4.1",
    "S1 NLI",
    scores,
    y,
    0.2910,
)


# S2
train = load(
    "/workspace/relevance_results_train_v2.json"
)
test = load(
    "/workspace/relevance_results_test_v2.json"
)

train_raw = np.array([
    float(r["raw_min_relevance"])
    for r in train
])

mn = train_raw.min()
mx = train_raw.max()

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in test
])

support = np.clip(
    (
        np.array([
            float(r["raw_min_relevance"])
            for r in test
        ])
        - mn
    )
    / (mx - mn),
    0.0,
    1.0,
)

scores = 1.0 - support

record(
    "Table 4.1",
    "S2 Relevance",
    scores,
    y,
    0.2314,
)


# S3 — use preserved full run, not 10-row debug cache
s3_candidates = [
    "/workspace/consistency_old_results/consistency_results_test_old.json",
    "/workspace/consistency_results_test.json",
]

s3_path = None

for p in s3_candidates:
    if Path(p).exists():
        candidate = load(p)

        if len(candidate) == 2700:
            s3_path = p
            d = candidate
            break

if s3_path is None:
    raise RuntimeError(
        "Could not find full 2700-row S3 test file."
    )

print("S3 source:", s3_path)

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    1.0 - float(r["consistency_score"])
    for r in d
])

record(
    "Table 4.1",
    "S3 Consistency",
    scores,
    y,
    0.2206,
)


# S4
d = load(
    "/workspace/signal4_results_test.json"
)

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    float(r["signal4_score"])
    for r in d
])

record(
    "Table 4.1",
    "S4 Fine-tuned",
    scores,
    y,
    0.1289,
)


# S5
d = load(
    "/workspace/signal5_v2_precision_results_test_mean.json"
)

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    1.0 - float(r["signal5_score"])
    for r in d
])

record(
    "Table 4.1",
    "S5 BERTScore",
    scores,
    y,
    0.2568,
)


# S6 / Signal 8
d = load(
    "/workspace/signal8_results_test.json"
)

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    float(r["signal8_score"])
    for r in d
])

record(
    "Table 4.1",
    "S6 Distilled",
    scores,
    y,
    0.2650,
)


# MiniCheck RoBERTa
d = load(
    "/workspace/minicheck_results_test_roberta.json"
)

d = [
    r for r in d
    if r["minicheck_score"] is not None
]

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    1.0 - float(r["minicheck_score"])
    for r in d
])

record(
    "Table 4.1",
    "MiniCheck-RoBERTa",
    scores,
    y,
    0.1700,
)


# MiniCheck 7B
d = load(
    "/workspace/minicheck_results_test_7b.json"
)

d = [
    r for r in d
    if r["minicheck_score"] is not None
]

y = np.array([
    int(r["ground_truth_hallucination"])
    for r in d
])

scores = np.array([
    1.0 - float(r["minicheck_score"])
    for r in d
])

record(
    "Table 4.1",
    "MiniCheck-7B",
    scores,
    y,
    0.2696,
)


# ============================================================
# FINAL RAGTRUTH FUSION
# ============================================================

print("\n" + "=" * 100)
print("B. FINAL RAGTRUTH FUSION")
print("=" * 100)

S2_MIN = -11.430
S2_MAX = 10.641


def norm_s2(v):
    return float(
        np.clip(
            (float(v) - S2_MIN)
            / (S2_MAX - S2_MIN),
            0.0,
            1.0,
        )
    )


rel_train = {
    int(r["idx"]): r
    for r in load(
        "/workspace/relevance_results_train_v2.json"
    )
}

rel_test = {
    int(r["idx"]): r
    for r in load(
        "/workspace/relevance_results_test_v2.json"
    )
}

s4_train = {
    int(r["idx"]): r
    for r in load(
        "/workspace/signal4_results_train_oof.json"
    )
}

s4_test = {
    int(r["idx"]): r
    for r in load(
        "/workspace/signal4_results_test.json"
    )
}


def build_fusion(rel, s4):
    rows = []

    for idx in sorted(
        set(rel) & set(s4)
    ):
        a = rel[idx]
        b = s4[idx]

        rows.append({
            "y":
                int(
                    a[
                        "ground_truth_hallucination"
                    ]
                ),
            "s2":
                norm_s2(
                    a["raw_min_relevance"]
                ),
            "s4":
                float(
                    b["signal4_score"]
                ),
            "task":
                a["task_type"],
            "model":
                a["model"],
        })

    return rows


tr = build_fusion(
    rel_train,
    s4_train,
)

te = build_fusion(
    rel_test,
    s4_test,
)

Xtr_num = np.array([
    [r["s2"], r["s4"]]
    for r in tr
])

Xte_num = np.array([
    [r["s2"], r["s4"]]
    for r in te
])

cats_tr = [
    [r["task"], r["model"]]
    for r in tr
]

cats_te = [
    [r["task"], r["model"]]
    for r in te
]

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

Xtr = np.hstack([
    Xtr_num,
    ohe.fit_transform(cats_tr),
])

Xte = np.hstack([
    Xte_num,
    ohe.transform(cats_te),
])

ytr = np.array([
    r["y"]
    for r in tr
])

yte = np.array([
    r["y"]
    for r in te
])

clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

clf.fit(
    Xtr,
    ytr,
)

fusion_scores = (
    clf.predict_proba(Xte)[:, 1]
)

record(
    "RAGTruth fusion",
    "S2+S4+metadata",
    fusion_scores,
    yte,
    0.0583,
)


# ============================================================
# RAGTRUTH++
# ============================================================

print("\n" + "=" * 100)
print("C. RAGTRUTH++")
print("=" * 100)

rtpp_path = (
    "/workspace/repo/results/robustness/"
    "ragtruth_pp_idfix/"
    "ragtruth_plusplus_results_thresholdfix.json"
)

rtpp = load(rtpp_path)

rows = rtpp["aligned_examples"]

y = np.array([
    int(r["rtp_label"])
    for r in rows
])

rtpp_specs = [
    ("S1", "s1", 0.2594),
    ("S2", "s2", 0.3192),
    ("S4", "s4", 0.4630),
    ("MiniCheck-7B", "mc", 0.2593),
]

for name, key, reported in rtpp_specs:
    scores = np.array([
        float(r[key])
        for r in rows
    ])

    record(
        "RAGTruth++",
        name,
        scores,
        y,
        reported,
    )


avg_scores = np.array([
    (
        float(r["s2"])
        + float(r["s4"])
    ) / 2.0
    for r in rows
])

record(
    "RAGTruth++",
    "Simple average S2+S4",
    avg_scores,
    y,
    0.3746,
)


# ============================================================
# HALUBENCH CORRECTED FIXED 8K
# ============================================================

print("\n" + "=" * 100)
print("D. HALUBENCH CORRECTED FIXED 8K")
print("=" * 100)

hb_s2s4 = load(
    "/workspace/halubench_final_s2s4_scores.json"
)

hb_mc = load(
    "/workspace/halubench_per_example_scores.json"
)

hb_split = load(
    "/workspace/halubench_group_split.json"
)

s2s4_map = {
    int(r["idx"]): r
    for r in hb_s2s4
}

mc_map = {
    int(r["idx"]): r
    for r in hb_mc
}

idxs = [
    int(x)
    for x in hb_split[
        "test_filtered_indices"
    ]
]

y = np.array([
    int(s2s4_map[i]["label"])
    for i in idxs
])

s4_scores = np.array([
    float(s2s4_map[i]["s4_score"])
    for i in idxs
])

s2_support = np.array([
    float(s2s4_map[i]["s2_support"])
    for i in idxs
])

mc_scores = np.array([
    float(mc_map[i]["mc_hall"])
    for i in idxs
])

INTERCEPT = -1.4168645142264538
COEF_S2 = -1.359009650504923
COEF_S4 = 3.1555282346796876

logit = (
    INTERCEPT
    + COEF_S2 * s2_support
    + COEF_S4 * s4_scores
)

fusion_scores = (
    1.0 / (1.0 + np.exp(-logit))
)

record(
    "HaluBench",
    "S4",
    s4_scores,
    y,
    0.2894,
)

record(
    "HaluBench",
    "Fusion",
    fusion_scores,
    y,
    0.2500,
)

record(
    "HaluBench",
    "MiniCheck-7B",
    mc_scores,
    y,
    0.1784,
)


# ============================================================
# PRINT FINAL AUDIT
# ============================================================

print("\n" + "=" * 132)
print("ECE CONSISTENCY AUDIT")
print("=" * 132)

print(
    f"{'Section':<19}"
    f"{'Method':<24}"
    f"{'Reported':>10}"
    f"{'Canonical':>12}"
    f"{'Legacy':>10}"
    f"{'Can-Rep':>11}"
    f"{'Can-Leg':>11}"
    f"{'#1.0':>8}"
)

print("-" * 132)

for r in audit_rows:
    print(
        f"{r['section']:<19}"
        f"{r['method']:<24}"
        f"{r['reported_ece']:>10.4f}"
        f"{r['canonical_ece']:>12.4f}"
        f"{r['legacy_ece']:>10.4f}"
        f"{r['canonical_minus_reported']:>+11.4f}"
        f"{r['canonical_minus_legacy']:>+11.4f}"
        f"{r['exact_score_1']:>8}"
    )


print("\n" + "=" * 132)
print("MISMATCHES > 0.00015")
print("=" * 132)

mismatches = [
    r for r in audit_rows
    if abs(
        r["canonical_minus_reported"]
    ) > 0.00015
]

if not mismatches:
    print("NONE — all reported ECEs are canonical-consistent.")
else:
    for r in mismatches:
        print(
            f"{r['section']} / "
            f"{r['method']}: "
            f"reported={r['reported_ece']:.4f}, "
            f"canonical={r['canonical_ece']:.4f}, "
            f"delta={r['canonical_minus_reported']:+.4f}, "
            f"exact_score_1={r['exact_score_1']}"
        )


out = {
    "canonical_definition": (
        "10 equal-width bins over [0,1]; "
        "score is hallucination probability; "
        "bin statistic compares mean predicted "
        "hallucination probability against empirical "
        "hallucination rate; last bin includes score=1.0"
    ),
    "legacy_difference": (
        "older implementation excluded exact score=1.0 "
        "from the final bin"
    ),
    "rows": audit_rows,
    "mismatches_over_0_00015":
        mismatches,
}

with open(
    "/workspace/ece_consistency_audit_results.json",
    "w",
) as f:
    json.dump(
        out,
        f,
        indent=2,
    )

print(
    "\nSaved: "
    "/workspace/ece_consistency_audit_results.json"
)
