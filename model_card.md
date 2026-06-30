---
language: en
license: mit
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

A 184M-parameter DeBERTa-v3 cross-encoder fine-tuned on RAGTruth for binary hallucination detection in retrieval-augmented generation outputs. Released alongside the master's thesis *Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification* (BHT Berlin).

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

| Metric | Value |
| --- | --- |
| F1 | 0.687 |
| Precision | 0.601 |
| Recall | 0.802 |
| AUROC | 0.847 |
| AUPRC | 0.772 |
| ECE | 0.129 |

For comparison within the thesis system: MiniCheck-7B (7B params) reaches AUROC 0.875 and ECE 0.270 on the same test set, while the logistic-regression S2+S4 fusion reaches AUROC 0.862 with ECE 0.105.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Tharun2908/s4-hallucination-detector")
model = AutoModelForSequenceClassification.from_pretrained(
    "Tharun2908/s4-hallucination-detector",
    ignore_mismatched_sizes=True,   # required for the size-mismatched head
)
model.eval()

answer  = "The treaty was signed in 1815 in Vienna."
context = "The Congress of Vienna concluded in June 1815..."

inputs = tokenizer(
    answer, context,
    truncation=True, max_length=512,
    return_tensors="pt",
)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)

hallucination_prob = probs[0, 1].item()   # higher = more likely hallucinated
print(f"P(hallucination) = {hallucination_prob:.3f}")
```

## Limitations and biases

**Domain.** Fine-tuned only on RAGTruth: three task types (Summary, QA, Data2txt) over six generators (gpt-3.5-turbo, gpt-4, llama-2-7b/13b/70b-chat, mistral-7b-instruct). Cross-domain transfer to other hallucination benchmarks is weak: on HaluBench zero-shot, AUROC drops to roughly 0.53. Fine-tuning on ~1,120 target examples brings AUROC up considerably (~0.80 on the HaluBench mix), but performance varies substantially by source: DROP and halueval adapt fast, biomedical (PubMedQA, CovidQA) and FinanceBench plateau at 0.55–0.75 even when most available source-specific examples are used.

**Calibration.** ECE is reasonable in-domain (0.129) but should be re-calibrated for any out-of-domain use. The thesis includes a calibration-only ablation showing this.

**Label noise.** The training labels are RAGTruth's original human annotations. Re-annotation work (RAGTruth++) suggests roughly 59% of items have their label flipped, with the positive rate moving from 16% to 75%. The model's discrimination is largely preserved under re-annotation (AUROC drops ~0.04 on a matched subset after retraining), but the optimal operating threshold shifts substantially.

**Absence claims.** Like other entailment-based verifiers, the model treats absence claims (e.g. "the document does not mention X") as a difficult case. In the clinical extension to MERLIN-DDx, absence rationales were observed to score higher hallucination than presence rationales across all verifiers, including this one. This is a known structural limitation.

**Subgroups.** Per-generator and per-task analyses are in the thesis. No specific protected-attribute fairness evaluation has been done; this model is intended for research and not for any decision affecting individuals.

## Reproducibility

The full training script (`signal4_finetune.py`), scoring scripts, and evaluation pipeline are in the companion code repository.

## Citation

```bibtex
@mastersthesis{thesis2026hallucination,
  title  = {Hallucination Detection in Retrieval-Augmented Generation Using Hybrid External Verification},
  author = {Tharun Johny},
  school = {BHT Berlin},
  year   = {2026}
}
```

## Acknowledgements

Built on [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base). Evaluated against [Bespoke MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) as an external baseline.
