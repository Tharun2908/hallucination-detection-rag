---
language: en
license: apache-2.0
library_name: transformers
tags:
- text-classification
- hallucination-detection
- rag
- faithfulness
- deberta
datasets:
- wandb/RAGTruth-processed
base_model: cross-encoder/nli-deberta-v3-base
metrics:
- f1
- accuracy
- roc_auc
pipeline_tag: text-classification
---

# S4 — Fine-tuned DeBERTa for RAG Hallucination Detection

A 184M-parameter DeBERTa-v3 cross-encoder fine-tuned on RAGTruth for binary hallucination detection in retrieval-augmented generation outputs. Prepared as part of the master's thesis *Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification* (BHT Berlin).

This is the **S4** signal from the thesis. It is the strongest single supervised signal in the system on the RAGTruth benchmark and is the most calibrated of the individual signals (test ECE 0.129).

## Model description

- **Base model:** [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
- **Parameters:** 184M
- **Architecture:** 2-class classification head over DeBERTa-v3-base
- **Input format:** `answer [SEP] context`, `truncation=True`, `max_length=512`
- **Output:** probability that the answer is hallucinated (label 1). Higher = more likely hallucinated. **No inversion needed.**

## Intended use

This model is intended for **research use** on the post-generation hallucination detection task, defined as: given a generated answer and a context passage, predict whether the answer contains claims unsupported by or contradicting the context.

It is the right tool for:

- Reproducing the thesis results on RAGTruth.
- Combining with relevance signals (S2) and/or MiniCheck-7B in a fusion or cascade.
- As a starting point for cross-domain adaptation on related hallucination datasets (see "Limitations" for caveats).

It is **not** intended for:

- Standalone production use without calibration on the target domain. ECE is reasonable in-domain but degrades out-of-domain.
- Determining factual correctness in an open-world sense. The model judges support *given the provided context only*; it has no knowledge of correctness beyond that context.
- Medical, legal, or financial decision-making.

## Training

- **Dataset:** [`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed), full train split (15,090 examples).
- **Validation:** 10% stratified split off the train pool.
- **Loss:** weighted cross-entropy with class weights derived from the train distribution.
- **Optimizer:** AdamW, learning rate 2e-5.
- **Batch size:** 16.
- **Max epochs:** 5 with patience 2 (early stopping on validation F1).
- **Best checkpoint:** epoch 3 (val F1 0.759, val AUROC 0.864).
- **Hardware:** single Tesla V100S-PCIE-32GB.

## Evaluation on RAGTruth test (n=2,700)

The final audited operating threshold is selected from out-of-fold RAGTruth training predictions and then applied unchanged to the held-out test set.

| Metric | Value |
| --- | ---: |
| F1 | 0.7024 |
| Precision | 0.6607 |
| Recall | 0.7497 |
| AUROC | 0.8470 |
| AUPRC | 0.7724 |
| ECE | 0.1289 |
| Hallucination threshold | 0.55 |

For comparison within the thesis system, MiniCheck-7B reaches F1 0.7260, AUROC 0.8754, AUPRC 0.8055, and ECE 0.2696 on the same RAGTruth test set. The metadata-free S2+S4 logistic-regression fusion reaches F1 0.7065, AUROC 0.8494, AUPRC 0.7664, and ECE 0.0547. The benchmark-aware S2+S4+task/generator-metadata fusion reaches F1 0.7262, AUROC 0.8749, AUPRC 0.7959, and ECE 0.0583.

## Usage

After training or obtaining the S4 checkpoint, load the saved model directory with Transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "/path/to/signal4_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

answer = "The treaty was signed in 1815 in Vienna."
context = "The Congress of Vienna concluded in June 1815..."

inputs = tokenizer(
    answer,
    context,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)

hallucination_prob = probs[0, 1].item()
print(f"P(hallucination) = {hallucination_prob:.3f}")
```

Higher class-1 probability means a higher estimated probability of hallucination. The final audited RAGTruth operating threshold is 0.55; for other domains, the threshold should be revalidated rather than transferred blindly.

## Limitations and biases

**Domain.** The model is fine-tuned only on RAGTruth, covering three task types (Summary, QA, Data2txt) and six generator models. Direct cross-benchmark transfer is weak: on the corrected group-disjoint HaluBench test set, zero-shot S4 reaches AUROC 0.5272. Target-domain supervision substantially improves performance: with 1,120 HaluBench training examples, mean AUROC reaches 0.8332 ± 0.0429, and with 2,240 examples it reaches 0.9616 ± 0.0019. This aggregate recovery is highly source-dependent, however: at N=2240, source-level AUROC ranges from 0.9905 on HaluEval and 0.9346 on DROP to 0.7456 on PubMedQA, 0.6583 on CovidQA, and 0.5560 on FinanceBench. The model should therefore not be treated as benchmark-independent without target-domain validation or adaptation.

**Calibration.** ECE is reasonable in-domain (0.129) but should be re-calibrated for any out-of-domain use. The thesis includes a calibration-only ablation showing this.

**Annotation shift.** The training labels are the original RAGTruth annotations. In the exactly aligned 408-example RAGTruth++ subset, 240 examples change label and the positive rate rises from 15.93% to 74.75%. At the original RAGTruth operating point, S4 retains moderate ranking information (AUROC 0.6837) but its F1 falls to 0.4268 because the transferred threshold has high precision and low recall under the revised label definition. In 5-fold retraining experiments on the matched subset, training with RAGTruth++ labels reaches mean AUROC 0.7290 ± 0.0545, compared with 0.6870 ± 0.0349 for the no-retraining baseline. These results show that both the annotation policy and the operating threshold materially affect apparent verifier performance.

**Absence claims.** Like other entailment-based verifiers, the model treats absence claims (e.g. "the document does not mention X") as a difficult case. In the clinical extension to MERLIN-DDx, absence rationales were observed to score higher hallucination than presence rationales across all verifiers, including this one. This is a known structural limitation.

**Subgroups.** Per-generator and per-task analyses are in the thesis. No specific protected-attribute fairness evaluation has been done; this model is intended for research and not for any decision affecting individuals.


## Reproducibility

The checkpoint training code is provided in `signals/signal4_finetune.py`. That script uses a stratified 90/10 train/validation split of the RAGTruth training set, selects the checkpoint by validation F1 with early stopping, and saves the resulting S4 model.

The final thesis evaluation should not be taken directly from the training script's original validation-threshold output. The submitted-thesis metrics use the audited train-side protocol: out-of-fold S4 predictions are generated with `signals/signal4_oof_train_scores.py`, and the final operating point is verified by `evaluation/table41_threshold_audit.py`. Under that protocol, the hallucination threshold is 0.55.

The companion repository also contains the fusion, robustness, cross-domain, bootstrap, and cascade audits used in the final thesis analysis.

## Citation

```bibtex
@mastersthesis{mekala2026hallucination,
  title  = {Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification},
  author = {Tharun Johny Mekala},
  school = {Berliner Hochschule fuer Technik (BHT)},
  year   = {2026}
}
```

## Acknowledgements

Built on [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base). Evaluated against [Bespoke MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) as an external baseline.
