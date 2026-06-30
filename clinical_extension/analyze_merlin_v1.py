#!/usr/bin/env python3
"""
MERLIN-DDx V1 rationale-faithfulness analysis.

Run this AFTER Dennis's team has filled the three label columns
(evidence_label, rule_label, binary_faithful) in the labeling sheet.

It evaluates each thesis verifier (MiniCheck-7B, fusion S2+S4, S4, and S2 if
available) as a hallucination detector, where every score is oriented so that
HIGHER = MORE hallucination (this orientation is already applied at scoring
time in score_v1_pod.py, so no inversion happens here).

The central design point from the task handoff:
  - The verifier only "sees" the EVIDENCE axis (is the claim grounded in the
    note?), not the RULE-COMPLIANCE axis (did the reasoning apply the V1
    annotation rules correctly?).
  - So the apples-to-apples test is verifier-score vs `evidence_label`.
  - `binary_faithful` (= supported AND compliant) is reported SEPARATELY, and
    the GAP between the two metrics is itself the result: it quantifies the
    rule-compliance blind spot.
  - Absence claims (predicted_value == -1, ~44% of rationales) are the hard
    case for entailment/MiniCheck-style verifiers, so everything is also broken
    down by value class (present / absent / unmentioned).

Usage:
  python analyze_merlin_v1.py \
      --csv merlin_v1_labeling_sheet.csv \
      --raw_scores merlin_v1_raw_scores.json \
      --outdir merlin_v1_analysis

--raw_scores is optional; it is only used to (a) recover s2_hall, which is not
in the human-facing CSV, and (b) backfill any score column the CSV is missing.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Score columns, in reporting order. Higher = more hallucination for all of them.
SIGNAL_COLS = ["minicheck_hall", "fusion_hall", "s4_hall", "s2_hall"]

# How an evidence_label maps to a binary hallucination target.
# "unclear" is excluded from the binary metric (kept as NaN) but still counted.
EVIDENCE_TO_HALL = {
    "supported": 0.0,
    "unsupported": 1.0,
    "contradicted": 1.0,
    "unclear": np.nan,
}

VALUE_NAMES = {1: "present", -1: "absent", 0: "unmentioned"}


def norm_str(x):
    """Normalize a label string: lowercase, collapse separators.

    Turns 'Non-Compliant', 'non compliant', 'NON_COMPLIANT' all into
    'non_compliant' so small labeling inconsistencies don't break the mapping.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def to_binary_faithful(x):
    """Coerce the binary_faithful column to {0, 1, NaN} from int/str/bool."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "yes", "faithful"}:
        return 1.0
    if s in {"0", "0.0", "false", "no", "unfaithful"}:
        return 0.0
    return np.nan


def compute_metrics(y_true, scores):
    """AUROC, AUPRC and F1 for one (target, signal) pair.

    Rows where either the target or the score is NaN are dropped. F1 is reported
    two ways: f1_best is the maximum F1 over all candidate thresholds (an
    optimistic, in-sample 'oracle' number, useful for ranking signals on this
    small set) and f1_at_0p5 is F1 at a fixed 0.5 cut. AUROC/AUPRC are the
    primary, threshold-free metrics. Degenerate cases (no rows, single class)
    return NaN rather than crashing.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[mask], s[mask]

    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()) if len(y) else 0,
        "auroc": np.nan,
        "auprc": np.nan,
        "f1_best": np.nan,
        "f1_threshold": np.nan,
        "f1_at_0p5": np.nan,
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        return out

    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))

    best_f1, best_t = -1.0, np.nan
    for t in np.unique(s):
        pred = (s >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    out["f1_best"] = float(best_f1)
    out["f1_threshold"] = float(best_t)
    out["f1_at_0p5"] = float(f1_score(y, (s >= 0.5).astype(int), zero_division=0))
    return out


def load_and_merge(csv_path, raw_scores_path):
    """Load the labeling sheet and optionally merge raw scores for s2_hall."""
    df = pd.read_csv(csv_path)
    df["item_id"] = df["item_id"].astype(str)

    if raw_scores_path and os.path.exists(raw_scores_path):
        with open(raw_scores_path) as fh:
            raw = json.load(fh)
        # raw is keyed by item id (string) -> {s2_hall, s4_hall, fusion_hall, minicheck_hall}
        raw_df = pd.DataFrame(raw).T.reset_index().rename(columns={"index": "item_id"})
        raw_df["item_id"] = raw_df["item_id"].astype(str)
        keep = ["item_id"] + [c for c in SIGNAL_COLS if c in raw_df.columns]
        raw_df = raw_df[keep]
        # Merge with a suffix, then prefer existing CSV values, backfill from raw.
        df = df.merge(raw_df, on="item_id", how="left", suffixes=("", "_raw"))
        for c in SIGNAL_COLS:
            raw_c = f"{c}_raw"
            if raw_c in df.columns:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    df[c] = df[c].fillna(pd.to_numeric(df[raw_c], errors="coerce"))
                else:
                    df[c] = pd.to_numeric(df[raw_c], errors="coerce")
                df = df.drop(columns=[raw_c])

    for c in SIGNAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="labeling sheet CSV with labels filled in")
    ap.add_argument("--raw_scores", default=None, help="optional merlin_v1_raw_scores.json (adds s2_hall)")
    ap.add_argument("--outdir", default="merlin_v1_analysis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_and_merge(args.csv, args.raw_scores)
    n_items = len(df)

    # ---- normalize labels --------------------------------------------------
    df["evidence_norm"] = df["evidence_label"].map(norm_str)
    df["rule_norm"] = df["rule_label"].map(norm_str)
    df["binary_faithful_num"] = df["binary_faithful"].map(to_binary_faithful)
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce").astype("Int64")

    # report unmapped evidence values so the user can fix the mapping if needed
    found_ev = sorted(v for v in df["evidence_norm"].dropna().unique())
    unmapped_ev = [v for v in found_ev if v not in EVIDENCE_TO_HALL]

    # ---- targets -----------------------------------------------------------
    # y_evidence: 1 = hallucination on the evidence axis (the real test)
    df["y_evidence"] = df["evidence_norm"].map(EVIDENCE_TO_HALL)
    # y_binary: 1 = not faithful (combines evidence AND rule compliance)
    df["y_binary"] = 1.0 - df["binary_faithful_num"]

    n_evidence_labeled = int(df["evidence_norm"].notna().sum())
    n_binary_labeled = int(df["binary_faithful_num"].notna().sum())

    signals = [c for c in SIGNAL_COLS if c in df.columns and df[c].notna().any()]

    results = {
        "meta": {
            "n_items": n_items,
            "signals_found": signals,
            "n_evidence_labeled": n_evidence_labeled,
            "n_binary_labeled": n_binary_labeled,
            "evidence_values_found": found_ev,
            "unmapped_evidence_values": unmapped_ev,
            "evidence_to_hall_mapping": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                         for k, v in EVIDENCE_TO_HALL.items()},
        }
    }

    if n_evidence_labeled == 0 and n_binary_labeled == 0:
        msg = ("No labels found yet (evidence_label and binary_faithful are both "
               "empty). Run this again once the labeling sheet is filled in.")
        print(msg)
        results["status"] = "no_labels"
        with open(os.path.join(args.outdir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        return

    if unmapped_ev:
        print(f"WARNING: unmapped evidence_label values {unmapped_ev} — they are "
              f"excluded from the evidence metric. Edit EVIDENCE_TO_HALL if needed.\n")

    # ---- overall metrics: each signal vs each target -----------------------
    overall = {}
    for sig in signals:
        overall[sig] = {
            "vs_evidence": compute_metrics(df["y_evidence"], df[sig]),
            "vs_binary_faithful": compute_metrics(df["y_binary"], df[sig]),
        }
    results["overall"] = overall

    # ---- gap: rule-compliance blind spot -----------------------------------
    # Positive gap => signal does BETTER on the evidence axis than on the
    # combined axis => binary_faithful penalizes it for rule failures it cannot
    # structurally detect.
    gap = {}
    for sig in signals:
        e, b = overall[sig]["vs_evidence"], overall[sig]["vs_binary_faithful"]
        gap[sig] = {
            "auroc_gap": (e["auroc"] - b["auroc"]) if not (np.isnan(e["auroc"]) or np.isnan(b["auroc"])) else np.nan,
            "auprc_gap": (e["auprc"] - b["auprc"]) if not (np.isnan(e["auprc"]) or np.isnan(b["auprc"])) else np.nan,
            "f1_best_gap": (e["f1_best"] - b["f1_best"]) if not (np.isnan(e["f1_best"]) or np.isnan(b["f1_best"])) else np.nan,
        }
    results["rule_compliance_blind_spot_gap"] = gap

    # ---- per-value-class breakdown -----------------------------------------
    per_value = {}
    for val, name in VALUE_NAMES.items():
        sub = df[df["predicted_value"] == val]
        per_value[name] = {"value": val, "n": int(len(sub)), "signals": {}}
        for sig in signals:
            per_value[name]["signals"][sig] = {
                "vs_evidence": compute_metrics(sub["y_evidence"], sub[sig]),
                "vs_binary_faithful": compute_metrics(sub["y_binary"], sub[sig]),
            }
    results["per_value_class"] = per_value

    # ---- confusion structure -----------------------------------------------
    # The headline cell: evidence supported BUT rule non-compliant. For these,
    # the verifier is CORRECT on its own (evidence) axis to call them faithful,
    # so verifier scores should be LOW here even though binary_faithful = 0.
    ev = df["evidence_norm"]
    rule = df["rule_norm"]
    crosstab = pd.crosstab(ev, rule, dropna=False)
    supported_noncompliant = df[(ev == "supported") & (rule == "non_compliant")]

    confusion = {
        "evidence_x_rule_counts": crosstab.to_dict(),
        "supported_but_noncompliant": {
            "n": int(len(supported_noncompliant)),
            "mean_scores": {sig: (float(supported_noncompliant[sig].mean())
                                  if len(supported_noncompliant) and supported_noncompliant[sig].notna().any()
                                  else None) for sig in signals},
        },
    }

    # The absence-claim trap: among items the human judged evidence-supported,
    # do verifiers still assign higher hallucination to ABSENT claims than to
    # PRESENT ones? If so, that is the conflation of "no evidence" with
    # "unfaithful" appearing numerically.
    trap = {}
    for val, name in VALUE_NAMES.items():
        sub = df[(df["predicted_value"] == val) & (ev == "supported")]
        trap[name] = {
            "n": int(len(sub)),
            "mean_scores": {sig: (float(sub[sig].mean()) if len(sub) and sub[sig].notna().any() else None)
                            for sig in signals},
        }
    confusion["absence_trap_mean_scores_among_supported"] = trap
    results["confusion"] = confusion

    # ---- case-level descriptive faithfulness (Dennis's point 3) ------------
    if df["binary_faithful_num"].notna().any():
        case_mean = df.groupby("case_id")["binary_faithful_num"].mean()
        results["case_level_binary_faithful"] = {
            "mean_over_cases": float(case_mean.mean()),
            "n_cases": int(case_mean.notna().sum()),
        }

    # ---- write + print -----------------------------------------------------
    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)

    summary = render_summary(results, signals)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write(summary)
    print(summary)


def _fmt(x):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def render_summary(results, signals):
    L = []
    m = results["meta"]
    L.append("=" * 72)
    L.append("MERLIN-DDx V1 faithfulness analysis")
    L.append("=" * 72)
    L.append(f"Items: {m['n_items']}   Signals: {', '.join(signals)}")
    L.append(f"Evidence-labeled: {m['n_evidence_labeled']}   binary_faithful-labeled: {m['n_binary_labeled']}")
    if m["unmapped_evidence_values"]:
        L.append(f"!! Unmapped evidence values (excluded): {m['unmapped_evidence_values']}")
    L.append("")

    L.append("--- OVERALL: AUROC / AUPRC / best-F1 -------------------------------")
    L.append("(the real test is 'vs evidence_label'; 'vs binary_faithful' is the")
    L.append(" combined axis and will look worse by the rule-compliance gap)")
    L.append("")
    header = f"{'signal':<16}{'target':<20}{'AUROC':>8}{'AUPRC':>8}{'F1*':>8}{'n':>6}{'pos':>6}"
    L.append(header)
    L.append("-" * len(header))
    for sig in signals:
        for tgt_key, tgt_name in [("vs_evidence", "evidence_label"), ("vs_binary_faithful", "binary_faithful")]:
            r = results["overall"][sig][tgt_key]
            L.append(f"{sig:<16}{tgt_name:<20}{_fmt(r['auroc']):>8}{_fmt(r['auprc']):>8}"
                     f"{_fmt(r['f1_best']):>8}{r['n']:>6}{r['n_pos']:>6}")
        L.append("")

    L.append("--- RULE-COMPLIANCE BLIND SPOT (evidence minus binary_faithful) ----")
    L.append("(positive = signal is penalized by binary_faithful for rule failures")
    L.append(" it cannot see; this gap is itself a thesis result)")
    L.append("")
    L.append(f"{'signal':<16}{'dAUROC':>10}{'dAUPRC':>10}{'dF1*':>10}")
    for sig in signals:
        g = results["rule_compliance_blind_spot_gap"][sig]
        L.append(f"{sig:<16}{_fmt(g['auroc_gap']):>10}{_fmt(g['auprc_gap']):>10}{_fmt(g['f1_best_gap']):>10}")
    L.append("")

    L.append("--- PER-VALUE-CLASS AUROC vs evidence_label ------------------------")
    L.append("(discriminative power should live in absent/unmentioned; a healthy")
    L.append(" headline driven only by easy 'present' positives would show here)")
    L.append("")
    L.append(f"{'signal':<16}{'present':>10}{'absent':>10}{'unmentd':>10}")
    for sig in signals:
        cells = []
        for name in ["present", "absent", "unmentioned"]:
            r = results["per_value_class"][name]["signals"][sig]["vs_evidence"]
            cells.append(_fmt(r["auroc"]))
        L.append(f"{sig:<16}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}")
    counts = {n: results["per_value_class"][n]["n"] for n in ["present", "absent", "unmentioned"]}
    L.append(f"{'(n per class)':<16}{counts['present']:>10}{counts['absent']:>10}{counts['unmentioned']:>10}")
    L.append("")

    L.append("--- CONFUSION: supported BUT rule-non-compliant --------------------")
    snc = results["confusion"]["supported_but_noncompliant"]
    L.append(f"n = {snc['n']}  (verifier is correct-on-evidence to call these faithful,")
    L.append(" so low mean scores here = correct behavior that binary_faithful punishes)")
    L.append("  mean scores: " + "  ".join(f"{s}={_fmt(snc['mean_scores'][s])}" for s in signals))
    L.append("")

    L.append("--- ABSENCE TRAP: mean score among evidence-SUPPORTED items --------")
    L.append("(if absent >> present, verifiers conflate 'no evidence' with 'unfaithful')")
    trap = results["confusion"]["absence_trap_mean_scores_among_supported"]
    for sig in signals:
        row = "  ".join(f"{name}={_fmt(trap[name]['mean_scores'][sig])}"
                        for name in ["present", "absent", "unmentioned"])
        L.append(f"  {sig:<16} {row}")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    main()
