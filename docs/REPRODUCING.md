# Reproducing results from a fresh checkout

Run commands from the repository root unless an absolute script path is used.
The active scripts in `signals`, `fusion`, `evaluation`, `robustness`,
`cross_domain`, and `efficiency` use `research_paths.py`. Both direct script
execution and `python -m package.module` work.

## 1. CPU quickstart: recompute a published result

This path uses the three committed HaluBench inputs. It requires no GPU, model
weights, dataset downloads, or full training environment. With Python 3.12:

```bash
python -m venv .venv-analysis
# Linux/macOS:
source .venv-analysis/bin/activate
# Windows Command Prompt instead: .venv-analysis\Scripts\activate
python -m pip install numpy==1.26.4 scikit-learn==1.5.2
python research_paths.py --check halubench
python cross_domain/halubench_groupfix_thresholdfix.py
python -m unittest discover -s tests -v
```

Expected AUROCs on the canonical 8,000-example group-disjoint test split:
S4 **0.5272**, metadata-free S2+S4 **0.5319**, MiniCheck-7B **0.7974**.
The output is `.artifacts/halubench_groupfix_thresholdfix_results.json`.
The test compares all endpoint and cascade results with the committed artifact
and checks execution from a different working directory.
This reproduces the analysis of saved predictions, not the model inference.

## 2. Choose the artifact directory

By default, generated scores, checkpoints, and analysis outputs go under
`.artifacts/` in this checkout, which Git ignores. To reuse an existing cluster
workspace or place large files on another disk, set one variable before running:

```bash
export RAG_WORKSPACE=/path/to/artifacts
# Original cluster layout: export RAG_WORKSPACE=/workspace
# Windows Command Prompt: set RAG_WORKSPACE=D:\rag-artifacts
python research_paths.py
```

Relative values are resolved against the repository root. `~` is expanded.
No root-level `/workspace` directory or source-code edits are required.

For these three **inputs only**, a regenerated file in `RAG_WORKSPACE` takes
precedence; otherwise the committed copy is used:

| Artifact name | Committed path |
| --- | --- |
| `halubench_group_split.json` | `results/cross_domain/halubench_groupfix/halubench_group_split.json` |
| `halubench_final_s2s4_scores.json` | `results/cross_domain/halubench_final_s2s4_scores.json` |
| `halubench_per_example_scores.json` | `results/cross_domain/halubench_per_example_scores.json` |

Other missing caches are not silently replaced by aggregate metrics. Outputs
always go to the artifact directory, so reproduction does not overwrite the
committed canonical results. A file at the same generated output path can be
replaced by rerunning a script; use a new artifact directory for separate runs.

## 3. Regenerate RAGTruth scores before fusion

Use a separate Linux GPU environment with the full `requirements.txt` and the
NLTK setup in the README. The lightweight CPU environment above is insufficient
for model inference or training. See `INFRASTRUCTURE.md` for the historical GPU
stack; a path portability check does not validate CUDA or retraining outcomes.

```bash
python research_paths.py --check fusion
```

A fresh clone reports four missing files. Generate them in this order:

| Command | Required/generated artifacts under `RAG_WORKSPACE` |
| --- | --- |
| `python signals/relevance_verifier_full_v2.py` | Produces `relevance_results_train_v2.json` and `relevance_results_test_v2.json` |
| `python signals/signal4_finetune.py` | Produces `signal4_model/` and `signal4_results_test.json` |
| `python signals/signal4_oof_train_scores.py` | Trains five fold models and produces `signal4_results_train_oof.json` |
| `python fusion/fusion_decomposition_review.py` | Consumes all four score files; produces `fusion_decomposition_review_results.json` |

These are substantial GPU jobs, not quickstart commands. Downloads come from the
original dataset/model sources named in the README. The released S4 checkpoint
is a trained model; it does not contain the five OOF score caches.
Do not substitute in-sample S4 training scores for OOF features.

Additional experiment prerequisites:

| Entry point | Additional inputs |
| --- | --- |
| `signals/signal8_distillation.py` | MiniCheck-7B train/test caches from `signals/minicheck_baseline.py` |
| `evaluation/table41_threshold_audit.py` | NLI, relevance, OOF S4, BERTScore precision, distilled S6, MiniCheck-RoBERTa and MiniCheck-7B train/test score caches; the committed S3 audit explicitly skips an incomplete cache |
| `evaluation/cascade_threshold_reaudit.py`, `evaluation/disagreement_threshold_reaudit.py` | Fusion inputs plus `minicheck_results_test_7b.json`; check with `python research_paths.py --check cascade` |
| `robustness/ragtruth_plusplus_eval_thresholdfix.py` | NLI, relevance, S4 and MiniCheck-7B test caches |
| `robustness/ragtruth_pp_retrain_idfix.py`, `cross_domain/halubench_curve_groupfix.py` | `signal4_model/` checkpoint and source datasets |
| `cross_domain/halubench_per_source_groupfix.py` | `halubench_curve_groupfix/results.json` plus its `per_run_predictions/`; aggregate committed results alone are insufficient |
| `cross_domain/cross_direction_n2240_groupfix.py` | Local original NLI checkpoint at `nli_deberta_v3_base_original/`; use the original `cross-encoder/nli-deberta-v3-base`, not trained S4; also requires the adaptation curve results for its final comparison |
| `evaluation/ece_consistency_audit.py` | All signal caches plus the canonical RAGTruth++ and HaluBench inputs referenced in the script |

Historical per-example caches and all trained checkpoints are not bundled in
Git. Preserve row IDs and the original train/test ordering when supplying them.
Keep the canonical HaluBench split; do not regenerate a random row-level split.
The old RoBERTa baseline runner and other earlier experiments remain in `legacy/`;
the portability change does not claim to make those historical scripts current.

## 4. Efficiency benchmark requires an explicit phase

With the full GPU environment and `signal4_model/` ready:

```bash
python efficiency/efficiency_benchmark.py --phase lightweight
python efficiency/efficiency_benchmark.py --phase minicheck
python efficiency/efficiency_benchmark.py --phase combine
```

Outputs are written to `RAG_WORKSPACE/efficiency/`. Run the model phases
separately to avoid keeping both verifiers in GPU memory.

`legacy/`, the clinical extension, and absolute paths stored inside historical
JSON metadata retain their original provenance. They are not rewritten by this fix.
