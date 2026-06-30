# Hallucination Detection in RAG Using Hybrid External Verification

Code accompanying the master's thesis *Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification* (BHT Berlin).

This repository implements and evaluates a hybrid post-generation verification system for detecting hallucinations in Retrieval-Augmented Generation (RAG) outputs. The system combines five verification signals plus an external baseline (MiniCheck-7B), evaluates fusion strategies, and studies robustness across label noise, task type, generator, and out-of-domain data.

The thesis contribution is **competitive performance with strong analysis across fusion, calibration, robustness, and efficiency**, not a new F1 maximum. No single system dominates all metrics: MiniCheck-7B leads on AUROC, the S2+S4 fusion leads on calibration, and the 30%-escalation cascade leads on F1.

---

## At a glance

| Component | RAGTruth test F1 | AUROC | ECE |
| --- | --- | --- | --- |
| S1 — NLI (DeBERTa) | 0.551 | 0.597 | 0.291 |
| S2 — Relevance (MS-MARCO) | 0.630 | 0.723 | 0.231 |
| S3 — Consistency (Mistral-7B) | 0.526 | 0.573 | 0.221 |
| **S4 — Fine-tuned DeBERTa (184M)** | 0.687 | 0.847 | **0.129** |
| S5 — BERTScore | 0.448 | 0.697 | 0.265 |
| S8 — Distillation (DeBERTa, soft) | 0.643 | 0.794 | — |
| MiniCheck-7B (baseline) | 0.726 | **0.875** | 0.270 |
| **Logreg S2+S4 fusion (+meta)** | 0.710 | 0.862 | **0.105** |
| **Cascade @ 30% escalation** | **0.763** | — | — |

S4 clears all prompting baselines including GPT-4-turbo (62.0 F1 on RAGTruth Table 5 overall response-level) but trails the original RAGTruth-paper fine-tuned Llama-2-13B (78.7). The best cascade configuration (0.763) still trails that 13B by ~2.4 F1 points.

---

## Repository structure

```
signals/                # individual verification signal scoring
fusion/                 # logistic-regression fusion variants
evaluation/             # main metrics and within-RAGTruth breakdowns
robustness/             # RAGTruth++ label-noise, granularity, error overlap
cross_domain/           # HaluBench transfer, adaptation curve, bidirectional study
cascade/                # cascaded verifier (lightweight → MiniCheck escalation)
efficiency/             # latency, throughput, memory benchmark
clinical_extension/     # MERLIN-DDx V1 rationale faithfulness (in progress)
results/                # per-experiment metric JSONs
figures/                # plots used in the thesis
model_card.md           # HuggingFace model card for the released S4 checkpoint
```

Per-example score files (~15k train + 2.7k test + 14k HaluBench rows) are intentionally not in this repository — they are large and not needed to verify the headline results. Aggregate metric JSONs are included under `results/`.

---

## Datasets

**This repository does not redistribute any dataset.** Each experiment loads data directly from its original source.

| Dataset | Source | Use |
| --- | --- | --- |
| RAGTruth | [`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed) | Primary benchmark (15,090 train / 2,700 test) |
| HaluBench | [`PatronusAI/HaluBench`](https://huggingface.co/datasets/PatronusAI/HaluBench) | Cross-domain evaluation (filter `source_ds != 'RAGTruth'` → 14,000 examples) |
| RAGTruth++ | [`blue-guardrails/ragtruth-plus-plus`](https://huggingface.co/datasets/blue-guardrails/ragtruth-plus-plus) | Label-noise robustness re-annotation |

RAGTruth uses JSON-format strings; parse with `json.loads`, not `ast.literal_eval`. RAGTruth++ comes as two CSV files joined by `message_stable_id`, and its assistant outputs are linked back to RAGTruth by text-prefix match rather than index (193 duplicate 100-char prefixes; first-occurrence wins, verified at 200 chars).

For HaluBench transfer, the held-out test set is a fixed 8,000-example proportionally-stratified subsample seeded at 42, with the remaining 6,000 forming the train pool. Use the same split for any reproduction (`cross_domain/halubench_curve.py` writes `test_train_pool_indices.json`).

---

## Environment

Tested on a Kubernetes GPU pod (Tesla V100S-PCIE-32GB, 32 GB VRAM) with the persistent volume mounted at `/workspace`.

Pinned versions matter — the combination below is the only one verified end-to-end with vLLM 0.4.3, the MiniCheck-7B baseline, and the saved S4 checkpoint loading cleanly:

```bash
pip install -r requirements.txt --break-system-packages
python -c "import nltk; nltk.download('punkt_tab')"
```

Known cross-version traps documented in the thesis notes:

- `vllm==0.4.3` pulls `torch` back to `2.3.0+cu121`; this is intentional. Do not upgrade.
- `transformers` must stay at `4.44.0`. Newer versions fail to load the S4 checkpoint with a `register_fake` error; some intermediate versions silently upgrade and break this on rerun.
- `xformers==0.0.26.post1` is required for V100 (compute capability 7.0); newer releases raise `NotImplementedError` on this GPU.
- MiniCheck-7B is ~15 GB and the container overlay fills up fast. Move the model cache to `/workspace` and symlink it into `/root/.cache/huggingface/hub/`.

The full setup walkthrough including error-by-error fixes is preserved in `docs/INFRASTRUCTURE.md` (if you choose to add it from the thesis notes — kept here as a stub since it is environment-specific).

---

## Reproducing the headline results

The scripts below assume the working directory contains the relevant data and that intermediate scoring outputs (`signal4_results_*.json`, etc.) are kept alongside. Each script writes its outputs to the current directory.

### Single signals (RAGTruth)

```bash
python signals/relevance_verifier_full_v2.py    # S2 — best unsupervised
python signals/signal4_finetune.py              # S4 — fine-tune DeBERTa on RAGTruth
python signals/signal4_score_train.py           # S4 — score train split
python signals/minicheck_baseline.py            # MiniCheck-7B external baseline
```

### Fusion (the final system)

```bash
python fusion/fusion_logreg_s2s4.py
```

This is the logistic-regression S2+S4 fusion with task-type and generator one-hot metadata. It assumes S2 and S4 scoring outputs already exist on disk.

### Robustness

```bash
python evaluation/leave_one_task_out.py         # AUROC 0.88–0.95 across held-out tasks
python evaluation/leave_one_generator_out.py    # AUROC 0.88–0.93 across held-out generators
python robustness/ragtruth_plusplus_eval.py     # first-pass scoring under re-annotation
python robustness/ragtruth_pp_retrain.py        # 5-fold retraining study (~hours on V100)
python robustness/sentence_level_s4.py          # granularity ablation (negative result)
python robustness/disagreement_analysis.py      # fusion vs MiniCheck error overlap
```

### Cross-domain

```bash
python cross_domain/halubench_eval.py           # zero-shot transfer
python cross_domain/halubench_fewshot.py        # 1,120-example adaptation
python cross_domain/halubench_curve.py          # full adaptation curve (5 sizes × 3 seeds)
python cross_domain/per_source_breakdown.py     # per-source AUROC heterogeneity
python cross_domain/cross_direction.py          # bidirectional from-scratch (RT↔HB)
```

### Cascade and efficiency

```bash
python cascade/cascaded_verifier.py             # in-domain (RAGTruth) — sweet spot at 30% escalation
python cascade/cascaded_verifier_halubench.py   # out-of-domain (no sweet spot)
python efficiency/efficiency_benchmark.py       # latency, throughput, memory
```

---

## Key empirical findings

1. **S2+S4 is the minimal effective fusion.** Adding S1, S3, or S5 changes AUROC by less than 0.001.
2. **The fusion is the best-calibrated system** (ECE 0.105 on RAGTruth test), beating all individual signals and MiniCheck-7B.
3. **Within-RAGTruth generalization is strong:** AUROC 0.88–0.95 in leave-one-task and leave-one-generator splits.
4. **Cross-domain transfer is fundamentally hard but fixable with adaptation.** S4 zero-shot on HaluBench is near-chance (AUROC 0.50); at N=2240 it reaches 0.96 aggregate — but that aggregate is dominated by halueval and DROP. FinanceBench, CovidQA, and PubMedQA plateau at 0.55–0.75 even when ~80% of available source-specific examples are used. MiniCheck-7B shows the opposite pattern, with the two verifiers exhibiting complementary domain coverage.
5. **Cascading lightweight fusion → MiniCheck-7B at 30% escalation produces F1 0.763** on RAGTruth, beating both endpoints (lightweight 0.710, MiniCheck alone 0.726) at ~4× lightweight cost, well below MiniCheck's 11× cost.
6. **Cascade gain comes from complementary specialization, not redundancy.** MiniCheck rescues false positives on faithful long-context summaries (75% are subtype=none, 49% Summary task); fusion catches real hallucinations on shorter QA and Data2txt outputs.
7. **RAGTruth++ drop is calibration shift, not granularity or representation.** Retraining on RAGTruth++ labels improves AUROC only marginally over retraining on original labels with the same examples (+0.034, one fold negative); sentence-level scoring is uniformly worse than response-level on both label sets. Pos rate moves from 16% to 75% under re-annotation, so the optimal threshold shifts substantially while ranking is largely preserved.
8. **Cross-benchmark transfer from scratch is zero.** Training from the NLI cross-encoder backbone (no S4 init) reaches val AUROC 0.85+ in-domain on HaluBench but test AUROC 0.46–0.55 on RAGTruth across all training sizes and seeds. The earlier "few-shot HaluBench" success at N=1120 only works because S4 was already pretrained on 15k RAGTruth examples — cheap adaptation requires expensive pretraining.

---

## Released artifacts

- **S4 checkpoint** (fine-tuned DeBERTa, 184M params, RAGTruth) — see [HuggingFace link to be added on release]. Model card in `model_card.md`.
- **Aggregate results JSONs** under `results/`.
- **Thesis plots** under `figures/`.

Not released (deliberately):

- The MiniCheck-7B weights (not ours to redistribute; available from [`bespokelabs/Bespoke-MiniCheck-7B`](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)).
- Per-example score files (~ tens of MB each; available on request).
- The HaluBench few-shot S4 checkpoint (released conditionally — see the model card).
- MERLIN-DDx data (clinical pipeline collaboration; the V1 sheet builder is public, the data is not).

---

## Citing

If this work is useful to you, please cite the thesis:

```bibtex
@mastersthesis{thesis2026hallucination,
  title  = {Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification},
  author = {Tharun Johny},
  school = {BHT Berlin},
  year   = {2026}
}
```

---

## License

This code is released under the MIT License — see `LICENSE`. The released S4 checkpoint inherits its license from the base model (`cross-encoder/nli-deberta-v3-base`) and is intended for research use; see `model_card.md` for details.

The datasets used here are governed by their own licenses (see their HuggingFace pages). MiniCheck-7B is released under its own terms — see the [Bespoke Labs model card](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B).
