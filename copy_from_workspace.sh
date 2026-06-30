#!/usr/bin/env bash
# copy_from_workspace.sh
#
# One-shot script that copies the thesis code and aggregate results from
# /workspace (the GPU pod's persistent volume) into the repo layout.
#
# Run from the repo root: bash scripts/copy_from_workspace.sh
#
# Safe to run multiple times (cp -u). Does NOT copy:
#   - model checkpoints (too large; released separately on HuggingFace)
#   - per-example score JSONs (too large; not needed for headline reproduction)
#   - cached datasets or HuggingFace model caches
#   - MERLIN-DDx clinical data (not for public release)

set -euo pipefail

WS="${WORKSPACE_DIR:-/workspace}"
REPO="${REPO_DIR:-$(pwd)}"

if [[ ! -d "$WS" ]]; then
  echo "ERROR: workspace directory '$WS' does not exist."
  echo "       Set WORKSPACE_DIR or run this on the pod."
  exit 1
fi

echo "Copying from $WS  →  $REPO"
echo

# -------- helpers --------
copy_if_exists() {
  # copy_if_exists <src> <dst_dir>
  local src="$1" dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$dst"
    cp -u "$src" "$dst/"
    echo "  ok   $(basename "$src")  →  $dst/"
  else
    echo "  miss $src (skipped)"
  fi
}

# -------- signals --------
echo "[signals/]"
for f in \
  nli_verifier_full_v2.py \
  relevance_verifier_full_v2.py \
  Final_consistency_verifier_v4.py \
  signal4_finetune.py \
  signal4_score_train.py \
  signal5_bertscore.py \
  signal8_distillation.py \
  minicheck_baseline.py
do
  copy_if_exists "$WS/$f" "$REPO/signals"
done
echo

# -------- fusion --------
echo "[fusion/]"
for f in \
  fusion_logreg_s2s4.py \
  fusion_logreg_s2s4_no_meta.py \
  fusion_logreg_no_s3.py \
  fusion_logreg_s5.py \
  fusion_logreg_s4.py \
  fusion_logreg_no_meta.py \
  fusion_logreg.py \
  fusion_final.py \
  fusion_final_s4.py \
  fusion_s2s4s8.py
do
  copy_if_exists "$WS/$f" "$REPO/fusion"
done
echo

# -------- evaluation --------
echo "[evaluation/]"
for f in \
  complete_metrics.py \
  leave_one_task_out.py \
  leave_one_generator_out.py \
  task_type_analysis.py \
  model_breakdown_analysis.py \
  clustering_analysis.py \
  nli_conflict_analysis.py \
  nli_baseless_analysis.py
do
  copy_if_exists "$WS/$f" "$REPO/evaluation"
done
echo

# -------- robustness --------
echo "[robustness/]"
for f in \
  ragtruth_plusplus_eval.py \
  ragtruth_pp_retrain.py \
  sentence_level_s4.py \
  disagreement_analysis.py
do
  copy_if_exists "$WS/$f" "$REPO/robustness"
done
echo

# -------- cross_domain --------
echo "[cross_domain/]"
for f in \
  halubench_eval.py \
  halubench_minicheck.py \
  halubench_scores.py \
  halubench_minicheck_only.py \
  halubench_calibration.py \
  halubench_fewshot.py \
  halubench_curve.py \
  per_source_breakdown.py \
  cross_direction.py
do
  copy_if_exists "$WS/$f" "$REPO/cross_domain"
done
echo

# -------- cascade --------
echo "[cascade/]"
for f in cascaded_verifier.py cascaded_verifier_halubench.py
do
  copy_if_exists "$WS/$f" "$REPO/cascade"
done
echo

# -------- efficiency --------
echo "[efficiency/]"
copy_if_exists "$WS/efficiency_benchmark.py" "$REPO/efficiency"
echo

# -------- clinical extension --------
echo "[clinical_extension/]"
for f in build_v1_sheet.py score_v1_pod.py analyze_merlin_v1.py
do
  copy_if_exists "$WS/merlin/$f" "$REPO/clinical_extension"
done
echo

# -------- aggregate result JSONs (small, useful) --------
echo "[results/signals/]"
for f in \
  nli_metrics_v2.json \
  relevance_metrics_v2.json \
  consistency_metrics.json \
  signal4_metrics.json \
  signal5_metrics_mean.json \
  signal8_metrics.json \
  minicheck_metrics_7b.json \
  complete_metrics_results.json
do
  copy_if_exists "$WS/$f" "$REPO/results/signals"
done
echo

echo "[results/fusion/]"
for f in \
  fusion_logreg_s2s4_results.json \
  fusion_logreg_no_s3_results.json \
  fusion_logreg_s5_results.json \
  fusion_logreg_s4_results.json \
  fusion_results_s4.json
do
  copy_if_exists "$WS/$f" "$REPO/results/fusion"
done
echo

echo "[results/evaluation/]"
for f in \
  leave_one_task_out_results.json \
  leave_one_generator_out_results.json \
  task_type_results.json \
  model_breakdown_results.json \
  clustering_results.json
do
  copy_if_exists "$WS/$f" "$REPO/results/evaluation"
done
echo

copy_renamed() {
  # copy_renamed <src> <dst_dir> <new_basename>
  # Avoids basename collisions when multiple sources are named results.json.
  local src="$1" dst="$2" new="$3"
  if [[ -e "$src" ]]; then
    mkdir -p "$dst"
    cp -u "$src" "$dst/$new"
    echo "  ok   $(basename "$src")  →  $dst/$new"
  else
    echo "  miss $src (skipped)"
  fi
}

echo "[results/robustness/]"
copy_if_exists "$WS/ragtruth_plusplus_results.json" "$REPO/results/robustness"
copy_renamed   "$WS/ragtruth_pp_retrain/clean_test/results.json" "$REPO/results/robustness" "ragtruth_pp_retrain_clean_test_results.json"
copy_renamed   "$WS/sentence_level_s4/results.json"              "$REPO/results/robustness" "sentence_level_s4_results.json"
copy_renamed   "$WS/disagreement/results.json"                   "$REPO/results/robustness" "disagreement_results.json"
copy_renamed   "$WS/disagreement/summary.txt"                    "$REPO/results/robustness" "disagreement_summary.txt"
echo

echo "[results/cross_domain/]"
for f in \
  halubench_results.json \
  halubench_minicheck_results.json \
  halubench_calibration_results.json \
  halubench_fewshot_results.json
do
  copy_if_exists "$WS/$f" "$REPO/results/cross_domain"
done
copy_renamed   "$WS/halubench_curve/results.json"            "$REPO/results/cross_domain" "halubench_curve_results.json"
copy_if_exists "$WS/halubench_curve/per_source_results.json" "$REPO/results/cross_domain"
copy_renamed   "$WS/cross_direction/results.json"            "$REPO/results/cross_domain" "cross_direction_results.json"
echo

echo "[results/cascade/]"
copy_if_exists "$WS/cascaded_verifier_results.json" "$REPO/results/cascade"
copy_if_exists "$WS/cascaded_verifier_halubench_results.json" "$REPO/results/cascade"
echo

echo "[results/efficiency/]"
copy_if_exists "$WS/efficiency/combined.json" "$REPO/results/efficiency"
copy_if_exists "$WS/efficiency/summary.txt"   "$REPO/results/efficiency"
echo

# -------- figures --------
echo "[figures/]"
copy_if_exists "$WS/cascaded_verifier_plot.png"           "$REPO/figures"
copy_if_exists "$WS/cascaded_verifier_halubench_plot.png" "$REPO/figures"
copy_if_exists "$WS/halubench_curve/curve_plot.png"       "$REPO/figures"
copy_if_exists "$WS/halubench_curve/per_source_plot.png"  "$REPO/figures"
copy_if_exists "$WS/cross_direction/bidirectional_plot.png" "$REPO/figures"
echo

echo "Done. Review with:  git status"
echo
echo "Reminder — these are deliberately NOT copied:"
echo "  - signal4_model/         (checkpoint; release on HuggingFace instead)"
echo "  - models--*/             (HuggingFace caches)"
echo "  - per-example score JSONs (large; available on request)"
echo "  - merlin_v1_*.csv / .jsonl / raw_scores.json  (clinical data, not for public release)"
