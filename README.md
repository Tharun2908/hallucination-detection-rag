# Hallucination Detection in RAG Using Hybrid External Verification

Code accompanying the master's thesis *Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification* (BHT Berlin, 2026).

This repository studies **post-generation, response-level faithfulness verification** for Retrieval-Augmented Generation (RAG). Given a generated answer and its retrieved evidence, the goal is to detect unsupported or contradictory content without modifying the retriever, generator, or decoding process.

The thesis evaluates heterogeneous verification signals, a compact learned fusion model, calibration, multiple forms of robustness, and cost-aware escalation to a stronger external verifier. The central result is:

> **Strong in-domain hallucination detection does not imply robust verification under distribution shift.**

On RAGTruth, the lightweight S2+S4+metadata fusion nearly matches MiniCheck-7B in discrimination while being much better calibrated. Under severe zero-shot transfer to HaluBench, however, the RAGTruth-trained lightweight verifier falls close to chance while MiniCheck-7B remains substantially stronger. Target-domain supervision can recover the lightweight architecture strongly, but recovery is highly source-dependent and direct cross-benchmark transfer remains near chance in both directions.

---

## Final thesis results at a glance

### RAGTruth — in-distribution verification

| System | F1 | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: |
| S4 — fine-tuned DeBERTa | 0.7024 | 0.8470 | 0.7724 | 0.1289 |
| MiniCheck-7B | 0.7260 | **0.8754** | **0.8055** | 0.2696 |
| **S2+S4+metadata fusion** | **0.7262** | 0.8749 | 0.7959 | **0.0583** |

All threshold-dependent values above use operating points selected from **training-side information only**. Test labels are not used for threshold selection.

The final fusion therefore has essentially the same RAGTruth discrimination as MiniCheck-7B, but substantially lower calibration error. Paired bootstrap intervals show no stable F1/AUROC/AUPRC advantage between fusion and MiniCheck-7B on RAGTruth, while the ECE difference strongly favors the lightweight fusion.

### HaluBench — zero-shot cross-benchmark transfer

The HaluBench numbers below use the corrected **group-disjoint fixed 8,000-example test set**.

| System | F1 | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: |
| RAGTruth-trained S4 | 0.4148 | 0.5272 | 0.5025 | 0.2894 |
| RAGTruth-trained metadata-free S2+S4 fusion | 0.3622 | 0.5319 | 0.5142 | 0.2500 |
| **MiniCheck-7B** | **0.7283** | **0.7974** | **0.8359** | **0.1784** |

This is the strongest robustness result in the thesis: the lightweight systems that are highly competitive on RAGTruth do **not** transfer reliably to the HaluBench benchmark distribution.

### RAGTruth cascade

MiniCheck-7B has a measured median-latency ratio of **10.77×** relative to the lightweight S2+S4 pipeline. Sequential cascade cost is:

```text
C(r) = 1 + 10.77r
```

where `r` is the fraction escalated to MiniCheck-7B.

The submitted-thesis cascade uses the final train-selected thresholds. The strongest observed test-set trade-off is a **post-hoc 20–30% escalation region**:

| Escalation | F1 | Relative sequential cost |
| ---: | ---: | ---: |
| 0% | 0.7262 | 1.00× |
| 20% | 0.7656 | 3.15× |
| 30% | **0.7659** | 4.23× |
| 100% | 0.7260 | 11.77× |

The 20–30% region was identified by inspecting the test curve and is **not** presented as a validation-selected deployment optimum. The fixed 20% point has a bootstrap-supported F1 gain over both standalone endpoints. On HaluBench, the same interior advantage disappears: more escalation progressively moves performance toward MiniCheck-7B, and no intermediate rate exceeds the MiniCheck endpoint.

---

## Verification methods

The final thesis evaluates six verification signals plus two MiniCheck baselines:

| Thesis name | Method | Main idea |
| --- | --- | --- |
| S1 | NLI entailment | Sentence-level entailment of answer content from retrieved evidence |
| S2 | Relevance | Weak-link semantic relevance using an MS MARCO cross-encoder |
| S3 | Cross-model consistency | Agreement between the candidate answer and fixed alternative generations |
| S5 | BERTScore precision | Token-level semantic coverage of answer content by evidence |
| S4 | Supervised verifier | DeBERTa-v3 fine-tuned directly on RAGTruth response-level labels |
| S6 | Distilled verifier | Compact DeBERTa student trained on continuous MiniCheck-7B teacher scores |
| MiniCheck-RoBERTa | External verifier | Compact MiniCheck grounded-verification baseline |
| MiniCheck-7B | External verifier | Strong grounded-verification baseline and cascade second stage |

The final in-distribution fusion is logistic regression over **S2 + S4 + task-type metadata + generator metadata**.

> **Legacy naming note:** some repository filenames still use `signal8` for the distilled MiniCheck student. The submitted thesis calls this experiment **S6**. Those legacy filenames refer to the same distillation line of work and are retained to avoid breaking historical experiment paths.

### Final standalone RAGTruth results

| Method | F1 | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: |
| S1 NLI | 0.5512 | 0.5965 | 0.3827 | 0.2921 |
| S2 Relevance | 0.6269 | 0.7234 | 0.4879 | 0.2314 |
| S3 Cross-model consistency | 0.5262 | 0.5727 | 0.3744 | 0.2206 |
| **S4 Fine-tuned DeBERTa** | **0.7024** | **0.8470** | **0.7724** | **0.1289** |
| S5 BERTScore precision | 0.6007 | 0.7537 | 0.5472 | 0.2568 |
| S6 Distilled MiniCheck | 0.6432 | 0.7942 | 0.6931 | 0.2650 |
| MiniCheck-RoBERTa | 0.5187 | 0.6213 | 0.4406 | 0.1700 |
| **MiniCheck-7B** | **0.7260** | **0.8754** | **0.8055** | 0.2696 |

S4 is the strongest lightweight standalone verifier; MiniCheck-7B has the strongest standalone discrimination.

---

## What the fusion result actually means

The final thesis decomposes the fusion rather than attributing all gains to “combining S2 and S4.”

| Configuration | F1 | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: |
| Raw S4 | 0.7024 | 0.8470 | 0.7724 | 0.1289 |
| **LogReg S4 only (no S2, no metadata)** | **0.7024** | **0.8470** | **0.7724** | **0.0443** |
| S4 + metadata | **0.7308** | 0.8710 | 0.7934 | 0.0610 |
| S2 + S4 | 0.7065 | 0.8494 | 0.7664 | 0.0547 |
| S2 + S4 + metadata | 0.7262 | **0.8749** | **0.7959** | 0.0583 |

The supported interpretation is:

- **LogReg S4 only isolates the calibration effect:** F1, AUROC, and AUPRC remain unchanged relative to raw S4, while ECE drops from **0.1289 to 0.0443**. This shows that most of the final pipeline's calibration improvement comes from logistic remapping rather than from adding S2 or metadata;
- **task/generator metadata** contributes most of the additional in-distribution discrimination over S4;
- **S2** adds a smaller additional ranking benefit;
- adding still more signals produces only small neighboring changes, so those ablations are not treated as evidence that a larger feature set is meaningfully better without additional uncertainty analysis.

---

## Protocol integrity and final audit decisions

Several review cycles focused specifically on preventing optimistic or ambiguous evaluation. The submitted thesis uses the following canonical protocols.

### 1. Train-side threshold selection

All reported standalone RAGTruth operating thresholds are selected using training-side predictions and then applied unchanged to the held-out test set. For S4, the threshold is selected from out-of-fold training predictions. The final S4 hallucination threshold is **0.55**.

The fusion threshold is also selected from **out-of-fold meta-model predictions** rather than from RAGTruth test labels.

### 2. Leakage-resistant stacking

Because S4 is itself supervised on RAGTruth, using in-sample S4 train predictions as fusion features would leak label information into the meta-classifier. Fusion training therefore uses **five-fold out-of-fold S4 scores**: every RAGTruth training example is scored by an S4 fold model that did not train on that example.

### 3. Exact RAGTruth++ alignment

RAGTruth++ is aligned with RAGTruth through:

```text
meta.original_id -> RAGTruth id
```

not by response-prefix matching. The final integrity audit resolves **408 / 408** examples exactly, with exact response text and generator-model agreement. All 408 originate from the original RAGTruth **test** split.

Under the revised labels:

```text
Original RAGTruth positive rate: 15.93%
RAGTruth++ positive rate:        74.75%
Changed labels:                  240 / 408
```

### 4. Group-disjoint HaluBench evaluation

After excluding HaluBench rows derived from RAGTruth, 14,000 examples remain. Related rows are grouped using:

```text
source_ds + normalized question + normalized passage
```

and kept in the same partition. The final split contains:

```text
adaptation pool: 6,000
fixed test set:  8,000
train/test group overlap: 0
```

Inner train/validation splits used during target-domain adaptation are also group-disjoint. The corrected adaptation curve was rerun for all **5 train sizes × 3 seeds = 15 runs**.

### 5. Fusion-layer holdouts are not full end-to-end holdouts

Leave-one-task-type-out and leave-one-generator-out experiments retrain the **fusion layer** using only the original RAGTruth training split after excluding the held-out group, then evaluate on the corresponding RAGTruth test subset.

The underlying S4 verifier remains fixed and may have encountered that task or generator during its original supervised training. These experiments therefore measure **fusion-layer transfer**, not fully end-to-end unseen-task or unseen-generator generalization.

### 6. Bootstrap uncertainty

The main RAGTruth and HaluBench comparisons include paired bootstrap confidence intervals. These intervals measure **test-sample uncertainty for fixed trained models**; they do not include retraining uncertainty and are not presented as a comprehensive formal hypothesis-testing framework.

---

## Robustness findings

### RAGTruth++: annotation-scheme shift

RAGTruth++ changes the target definition and positive prevalence dramatically while keeping the underlying responses aligned.

At the original RAGTruth operating points, S4 retains moderate ranking information on RAGTruth++ (`AUROC = 0.6837`) but has high precision and low recall, yielding `F1 = 0.4268`. This is an operating-point failure under annotation shift, not a sudden disappearance of all ranking information.

The 5-fold retraining experiment uses all 408 matched examples:

| Condition | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: |
| Baseline S4, no subset-specific retraining | 0.7734 ± 0.0425 | 0.6870 ± 0.0349 | 0.8559 ± 0.0329 |
| **Retrain on RAGTruth++ labels** | **0.8486 ± 0.0210** | **0.7290 ± 0.0545** | **0.8763 ± 0.0484** |
| Retrain on original labels | 0.7645 ± 0.0634 | 0.7036 ± 0.0656 | 0.8671 ± 0.0355 |

For this CV table, F1 uses a **fold-specific threshold selected on an inner validation split using RAGTruth++ labels for each condition**; AUROC and AUPRC remain threshold-free. The more defensible evidence of adaptation is therefore the ranking change, not the raw jump in F1. The corrected-label retraining condition has a mean AUROC gain of `+0.0421` over the baseline and `+0.0254` over the same-example original-label control, but the five-fold variation is large enough that this should be interpreted cautiously.

### HaluBench: target-domain adaptation

The corrected group-disjoint adaptation curve is:

| HaluBench train examples | AUROC mean ± std | F1 mean ± std |
| ---: | ---: | ---: |
| 112 | 0.5008 ± 0.0091 | 0.6432 ± 0.0252 |
| 280 | 0.5477 ± 0.0178 | 0.6325 ± 0.0179 |
| 560 | 0.5865 ± 0.0276 | 0.6422 ± 0.0270 |
| 1120 | 0.8332 ± 0.0429 | 0.7685 ± 0.0285 |
| 2240 | **0.9616 ± 0.0019** | **0.8814 ± 0.0029** |

The aggregate recovery is strong but **not uniform across sources**. At N=2240, AUROC is 0.9905 on HaluEval and 0.9346 on DROP, but only 0.7456 on PubMedQA, 0.6583 on CovidQA, and 0.5560 on FinanceBench.

A matched initialization control shows that strong target-domain learning does not depend on the RAGTruth-trained S4 checkpoint. Starting from the common base NLI checkpoint and training on HaluBench at N=2240 reaches `AUROC = 0.9460 ± 0.0032`; starting from RAGTruth S4 reaches `0.9616 ± 0.0019`. Prior RAGTruth supervision therefore adds a smaller benefit, while access to target-domain labels drives most of the recovery.

### Bidirectional cross-benchmark transfer

Using the same base NLI initialization and N=2240 in both directions:

| Train | Evaluate | AUROC | AUPRC |
| --- | --- | ---: | ---: |
| RAGTruth | HaluBench | 0.5036 ± 0.0415 | 0.5034 ± 0.0320 |
| HaluBench | RAGTruth | 0.4836 ± 0.0543 | 0.3279 ± 0.0216 |

Direct transfer is therefore approximately chance-level in **both directions**, despite strong in-domain learnability. The thesis distinguishes target-domain learnability from benchmark-independent generalization rather than treating them as the same property.

---

## Cascade and error complementarity

The final train-threshold disagreement audit on the 2,700-example RAGTruth test set is:

| Outcome | Examples | Share |
| --- | ---: | ---: |
| Both correct | 1806 | 66.89% |
| Both wrong | 243 | 9.00% |
| Fusion correct, MiniCheck-7B wrong | 327 | 12.11% |
| MiniCheck-7B correct, fusion wrong | 324 | 12.00% |

At least one verifier is correct on **91.0%** of the test set. The nearly symmetric unique-win counts show that MiniCheck-7B is not simply a strict replacement for the lightweight verifier; the two systems make materially different errors. This complementarity helps explain why selective escalation can improve F1 on RAGTruth.

Under HaluBench shift, however, the first stage itself becomes weak, so uncertainty routing no longer produces an interior performance peak. Cascading is therefore a **distribution-dependent design pattern**, not a universally beneficial deployment recipe.

---

## Repository structure

```text
signals/                # individual verification signal scoring/training
fusion/                 # logistic-regression fusion and decomposition
evaluation/             # threshold audits, bootstrap CIs, holdouts, disagreement
robustness/             # RAGTruth++ alignment/retraining and auxiliary analyses
cross_domain/           # HaluBench group-disjoint transfer and adaptation
cascade/                # historical/initial cascade implementations
efficiency/             # latency, throughput, memory benchmark
clinical_extension/     # MERLIN-DDx V1 extension code (not a core thesis result)
results/                # aggregate/canonical experiment outputs
figures/                # thesis plots
model_card.md           # model card for the S4 checkpoint
```

Per-example score files (~15k RAGTruth train + 2.7k test + 14k HaluBench rows) are not stored in this repository because they are large intermediate artifacts. Aggregate result files are included under `results/`. Full end-to-end reproduction requires regenerating the per-example scores or obtaining them separately.

---

## Datasets

**This repository does not redistribute the datasets.** Experiments load them from their original sources.

| Dataset | Source | Use |
| --- | --- | --- |
| RAGTruth | [`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed) | Primary benchmark: 15,090 train / 2,700 test |
| RAGTruth++ | [`blue-guardrails/ragtruth-plus-plus`](https://huggingface.co/datasets/blue-guardrails/ragtruth-plus-plus) | 408 exactly aligned examples for annotation-shift analysis |
| HaluBench | [`PatronusAI/HaluBench`](https://huggingface.co/datasets/PatronusAI/HaluBench) | Cross-benchmark transfer/adaptation after removing RAGTruth-derived rows |

RAGTruth++ assistant responses are linked to RAGTruth using the canonical `meta.original_id` field, **not text-prefix matching**.

HaluBench evaluation uses the saved group-disjoint split under `results/cross_domain/halubench_groupfix/`. Reproduction should reuse that split rather than creating a new row-level random split.

---

## Environment

The main thesis experiments were run on a Kubernetes GPU pod with a **Tesla V100S-PCIE-32GB** and a persistent volume mounted at `/workspace`.

```bash
pip install -r requirements.txt --break-system-packages
python -c "import nltk; nltk.download('punkt_tab')"
```

The verified environment has several version-sensitive dependencies:

- `vllm==0.4.3` pins `torch` to the compatible 2.3.x CUDA stack used for the MiniCheck runs.
- `transformers==4.44.0` is the verified version for loading the saved S4 checkpoint in this environment.
- `xformers==0.0.26.post1` is the version used on the V100 (compute capability 7.0).
- MiniCheck-7B is large enough that Hugging Face cache placement matters on small container overlays.

Additional cluster and cache details are documented in [`docs/INFRASTRUCTURE.md`](docs/INFRASTRUCTURE.md).

---

## Reproducing the submitted-thesis results

The commands below highlight the **final/corrected** entry points. Some older scripts remain in the repository for provenance; see the canonical-artifact note below before using their outputs as thesis numbers.

### Standalone signals and threshold audit

```bash
python signals/nli_verifier_full_v2.py
python signals/relevance_verifier_full_v2.py
python signals/signal4_finetune.py
python signals/signal4_oof_train_scores.py
python signals/signal5_bertscore_v2_precision.py
python signals/signal8_distillation.py          # legacy filename; S6 in the thesis
python signals/minicheck_baseline.py

python evaluation/table41_threshold_audit.py   # final train-side operating-point audit
```

### Fusion

```bash
python fusion/fusion_logreg_s2s4.py
python fusion/fusion_decomposition_review.py
```

Fusion training uses out-of-fold S4 train features; test-time S4 scores come from the final S4 checkpoint trained on the complete RAGTruth training split.

### RAGTruth++

```bash
python robustness/ragtruth_plusplus_eval_thresholdfix.py
python robustness/ragtruth_pp_retrain_idfix.py
```

These are the corrected exact-ID / final-threshold paths. Do not use the old response-prefix matching interpretation for thesis reproduction.

### Fusion-layer task and generator transfer

```bash
python evaluation/leave_one_strict_review.py
```

This is the stricter review protocol that fits the fusion layer only on the original RAGTruth training split after removing the held-out group.

### HaluBench group-disjoint transfer and adaptation

```bash
python cross_domain/halubench_curve_groupfix.py
python cross_domain/halubench_per_source_groupfix.py
python cross_domain/halubench_groupfix_thresholdfix.py
python cross_domain/cross_direction_n2240_groupfix.py
```

The `groupfix` experiments use the corrected 6,000/8,000 group-disjoint split and group-disjoint inner train/validation partitions.

### Cascade, disagreement, bootstrap and efficiency

```bash
python evaluation/cascade_threshold_reaudit.py
python evaluation/disagreement_threshold_reaudit.py
python evaluation/bootstrap_ragtruth_main.py
python cross_domain/bootstrap_halubench_groupfix_thresholdfix.py
python efficiency/efficiency_benchmark.py
```

---

## Canonical result artifacts

The repository contains historical outputs from earlier experiment iterations. For the **submitted thesis**, prefer the following files:

```text
results/evaluation/table41_threshold_audit_results.json
results/fusion/fusion_decomposition_review_results.json

results/robustness/ragtruth_pp_idfix/
  ragtruth_plusplus_results_thresholdfix.json
  full/results.json
  full/summary.txt

results/evaluation/leave_one_strict_review_results.json

results/cross_domain/halubench_groupfix/
  halubench_group_split.json
  halubench_split_integrity_audit_results.json
  halubench_groupfix_thresholdfix_results.json
  results.json
  per_source_results_groupfix.json

results/cross_domain/cross_direction_n2240_groupfix/
  results.json
  summary.txt

results/evaluation/cascade_threshold_reaudit_results.json
results/evaluation/disagreement_threshold_reaudit_results.json
results/evaluation/disagreement_threshold_reaudit_summary.txt

results/bootstrap/bootstrap_ragtruth_main_results.json
results/cross_domain/halubench_groupfix/bootstrap_halubench_groupfix_thresholdfix_results.json

results/efficiency/combined.json
```

> **Historical-output warning:** older files without the relevant `idfix`, `groupfix`, `thresholdfix`, `strict_review`, or re-audit qualification may reflect superseded matching, split, or threshold protocols. They are retained for provenance but should not be copied as the submitted-thesis numbers without checking against the canonical artifacts above.

---

## Key empirical findings

1. **Direct supervision is the strongest lightweight standalone strategy in-domain.** S4 reaches AUROC 0.8470 on RAGTruth, well above the generic entailment, relevance, similarity, and cross-model consistency signals.

2. **The final fusion nearly matches MiniCheck-7B in RAGTruth discrimination but is much better calibrated.** Fusion AUROC is 0.8749 versus 0.8754 for MiniCheck-7B, while ECE is 0.0583 versus 0.2696.

3. **Fusion gains have different causes.** Logistic regression supplies most of the calibration improvement; task/generator metadata supplies most of the additional in-distribution discrimination; S2 contributes a smaller ranking increment.

4. **RAGTruth++ is a large annotation/operating-point shift, not ordinary random label noise.** The positive rate changes from 15.93% to 74.75%, 240/408 labels change, and original operating thresholds transfer poorly even when ranking remains partially useful.

5. **Zero-shot cross-benchmark robustness strongly favors MiniCheck-7B.** On group-disjoint HaluBench, S4/fusion AUROC is about 0.53 while MiniCheck-7B reaches 0.7974.

6. **Target-domain supervision can recover the lightweight architecture, but recovery is source-dependent.** Adapted S4 reaches aggregate AUROC 0.9616 ± 0.0019 at N=2240, driven especially by HaluEval and DROP; FinanceBench, CovidQA, and PubMedQA remain substantially harder.

7. **The recovery is mostly target-supervision driven rather than inherited from RAGTruth.** At N=2240, the common base NLI initialization already reaches HaluBench AUROC 0.9460 ± 0.0032; RAGTruth-S4 initialization raises this to 0.9616 ± 0.0019.

8. **Strong in-domain learnability does not imply direct transfer.** Matched-size bidirectional transfer from a common base remains near chance: 0.5036 AUROC for RAGTruth→HaluBench and 0.4836 for HaluBench→RAGTruth.

9. **Cascading is useful only when the first stage remains informative and complementary.** On RAGTruth, 20–30% escalation gives the strongest observed post-hoc F1/cost region; on HaluBench, the interior advantage disappears.

10. **The two RAGTruth verifier stages have complementary rather than nested errors.** Fusion uniquely wins on 327 examples and MiniCheck-7B on 324, while both are wrong on 243 of 2,700 examples.

---

## Released artifacts

- **Aggregate result JSONs** under `results/`.
- **Thesis plots** under `figures/`.
- **S4 model card** in [`model_card.md`](model_card.md); the checkpoint link can be added if/when the model is released publicly.

Not redistributed here:

- MiniCheck-7B weights (available from [`bespokelabs/Bespoke-MiniCheck-7B`](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)).
- Original benchmark datasets.
- Large per-example score files used as intermediate artifacts.
- MERLIN-DDx patient-derived data from the separate clinical extension.

---

## Citing

If this work is useful to you, please cite the thesis:

```bibtex
@mastersthesis{mekala2026hallucination,
  title  = {Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification},
  author = {Tharun Johny Mekala},
  school = {Berliner Hochschule fuer Technik (BHT)},
  year   = {2026}
}
```

---

## License

The code is released under the MIT License; see [`LICENSE`](LICENSE).

The datasets used by the experiments remain governed by their original licenses. MiniCheck-7B is released under its own terms; see the [Bespoke Labs model card](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B).
