#!/usr/bin/env python3
"""
score_v1_pod.py  —  RUN ON THE GPU POD (/workspace), not in the chat container.

Fills the three verifier-score columns in merlin_v1_labeling_sheet.csv:
  minicheck_hall : 1 - MiniCheck-7B support prob          (higher = more hallucination)
  s4_hall        : Signal-4 finetuned-DeBERTa score        (higher = more hallucination)
  fusion_hall    : Logreg S2+S4 (+metadata) hallucination prob

Inputs  (copy these up to the pod, e.g. /workspace/merlin/):
  merlin_v1_items.jsonl          (item_id, answer, context, predicted_value, ...)
  merlin_v1_labeling_sheet.csv   (the human sheet; we only write the 3 score cols)

Reuses the SAME logic as:
  - minicheck_baseline.py        (MiniCheck-7B via vLLM)
  - disagreement_analysis.py     (S2 raw_min_relevance + S4 + fusion logreg refit)

------------------------------------------------------------------------------
SETUP ON A FRESH POD (from thesis_notes.md, condensed):
  mkdir -p /root/.cache/huggingface/hub
  ln -s /workspace/models--bespokelabs--Bespoke-MiniCheck-7B \
        /root/.cache/huggingface/hub/models--bespokelabs--Bespoke-MiniCheck-7B   # if still cached
  pip install vllm==0.4.3 transformers==4.44.0 xformers==0.0.26.post1 \
              sentence-transformers sentencepiece scikit-learn fsspec \
              "minicheck @ https://github.com/Liyan06/MiniCheck/archive/refs/heads/main.zip" \
              --break-system-packages
  python -c "import nltk; nltk.download('punkt_tab')"
  # NOTE: MiniCheck-7B was deleted for disk space at some point — re-download (~30 min) if gone.

RUN (MiniCheck needs the __main__ guard for vLLM multiprocessing):
  nohup python -u /workspace/merlin/score_v1_pod.py \
        --merlin_dir /workspace/merlin \
        > /workspace/merlin/score_v1.log 2>&1 &
------------------------------------------------------------------------------
"""
import os, json, argparse
import numpy as np
import pandas as pd

# ---- S2 normalization constants (from thesis_notes.md; TRAIN stats, no leakage) ----
S2_MIN, S2_MAX = -11.430, 10.641
def norm_s2(val):
    return float(max(0.0, min(1.0, (val - S2_MIN) / (S2_MAX - S2_MIN))))

# Manifestation phrasing -> answer is already built in the JSONL ('answer' field).

def load_items(merlin_dir):
    items = []
    with open(os.path.join(merlin_dir, "merlin_v1_items.jsonl")) as f:
        for line in f:
            items.append(json.loads(line))
    return items


# ============================ SIGNAL 2 (relevance) ============================
def score_s2(items, device="cuda"):
    """cross-encoder/ms-marco-MiniLM-L-6-v2, per-answer-sentence best vs context
    sentences, take MIN across answer sentences (raw_min_relevance), then norm+invert.
    Mirrors relevance_verifier_full_v2.py aggregation used for fusion."""
    from sentence_transformers import CrossEncoder
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt"); nltk.download("punkt_tab")

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    out = {}
    for it in items:
        a_sents = sent_tokenize(it["answer"]) or [it["answer"]]
        c_sents = sent_tokenize(it["context"]) or [it["context"]]
        per_ans_best = []
        for a in a_sents:
            pairs = [[a, c] for c in c_sents]
            scores = ce.predict(pairs, show_progress_bar=False)
            per_ans_best.append(float(np.max(scores)))
        raw_min = float(np.min(per_ans_best))     # raw_min_relevance
        s2_norm = norm_s2(raw_min)
        out[it["item_id"]] = {"raw_min_relevance": raw_min,
                              "s2_hall": 1.0 - s2_norm}   # invert: low rel = hallucination
    return out


# ============================ SIGNAL 4 (finetuned) ============================
def score_s4(items, model_dir="/workspace/signal4_model", device="cuda"):
    """Finetuned DeBERTa. Input format: answer [SEP] context, truncation=True.
    signal4_score = P(hallucination) ; higher = hallucination (NO inversion)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, ignore_mismatched_sizes=True).to(device).eval()
    out = {}
    with torch.no_grad():
        for it in items:
            enc = tok(it["answer"], it["context"], truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits
            prob_hall = torch.softmax(logits, dim=-1)[0, 1].item()
            out[it["item_id"]] = {"s4_hall": float(prob_hall)}
    return out


# ============================ FUSION (logreg S2+S4+meta) ======================
def score_fusion(items, s2, s4, merlin_dir):
    """Refit the S2+S4(+metadata) logistic regression on RAGTruth train EXACTLY as
    disagreement_analysis.py does, then apply to the MERLIN items.

    metadata one-hots: task_type and model. MERLIN has neither, so we set the
    metadata block to zeros (out-of-vocab) — the model falls back to the S2/S4
    coefficients. This matches the 'no reliable metadata' regime and is the honest
    choice for an external dataset. We log this clearly.
    """
    from sklearn.linear_model import LogisticRegression
    # ---- load RAGTruth train fusion features (produced earlier in thesis) ----
    # Expected: /workspace/fusion_train_features.json with rows of
    #   {s2_hall, s4_hall, task_type, model, label}
    feat_path = os.path.join("/workspace", "fusion_train_features.json")
    if not os.path.exists(feat_path):
        print(f"[fusion] {feat_path} not found — refit from raw signal results instead.")
        return _fusion_from_raw(items, s2, s4)

    train = pd.read_json(feat_path)
    task_types = sorted(train["task_type"].unique())
    models = sorted(train["model"].unique())

    def feats(s2v, s4v, tt=None, md=None):
        x = [s2v, s4v]
        x += [1.0 if tt == t else 0.0 for t in task_types]
        x += [1.0 if md == m else 0.0 for m in models]
        return x

    Xtr = [feats(r.s2_hall, r.s4_hall, r.task_type, r.model)
           for r in train.itertuples()]
    ytr = train["label"].astype(int).tolist()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)

    out = {}
    for it in items:
        i = it["item_id"]
        x = feats(s2[i]["s2_hall"], s4[i]["s4_hall"], tt=None, md=None)  # zero metadata
        p = clf.predict_proba([x])[0, 1]
        out[i] = {"fusion_hall": float(p)}
    return out

def _fusion_from_raw(items, s2, s4):
    """Fallback: simple equal-weight mean of s2_hall and s4_hall if no train
    features file is present. Logged as APPROXIMATE."""
    print("[fusion] WARNING: using equal-weight S2+S4 mean (approximate, not the "
          "trained logreg). Provide fusion_train_features.json for the exact system.")
    out = {}
    for it in items:
        i = it["item_id"]
        out[i] = {"fusion_hall": 0.5 * (s2[i]["s2_hall"] + s4[i]["s4_hall"])}
    return out


# ============================ MINICHECK-7B (vLLM) =============================
def score_minicheck(items):
    """MiniCheck-7B support prob; minicheck_hall = 1 - support. Mirrors
    minicheck_baseline.py. MUST be called under __main__ for vLLM mp."""
    from minicheck.minicheck import MiniCheck
    scorer = MiniCheck(model_name="Bespoke-MiniCheck-7B", enable_prefix_caching=False)
    docs = [it["context"] for it in items]
    claims = [it["answer"] for it in items]
    pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)
    out = {}
    for it, prob in zip(items, raw_prob):
        out[it["item_id"]] = {"minicheck_hall": float(1.0 - prob)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merlin_dir", default="/workspace/merlin")
    ap.add_argument("--skip_minicheck", action="store_true",
                    help="score only the lightweight signals first")
    args = ap.parse_args()

    items = load_items(args.merlin_dir)
    print(f"loaded {len(items)} items")

    print("scoring S2 (relevance)...");  s2 = score_s2(items)
    print("scoring S4 (finetuned)...");  s4 = score_s4(items)
    print("scoring fusion...");          fus = score_fusion(items, s2, s4, args.merlin_dir)

    mc = {}
    if not args.skip_minicheck:
        print("scoring MiniCheck-7B (vLLM)...")
        mc = score_minicheck(items)

    # ---- write scores back into the sheet ----
    sheet_path = os.path.join(args.merlin_dir, "merlin_v1_labeling_sheet.csv")
    sheet = pd.read_csv(sheet_path)
    sheet["s4_hall"]     = sheet["item_id"].map(lambda i: round(s4[i]["s4_hall"], 4))
    sheet["fusion_hall"] = sheet["item_id"].map(lambda i: round(fus[i]["fusion_hall"], 4))
    if mc:
        sheet["minicheck_hall"] = sheet["item_id"].map(lambda i: round(mc[i]["minicheck_hall"], 4))
    sheet.to_csv(sheet_path, index=False)
    print(f"wrote scores to {sheet_path}")

    # also dump raw per-signal scores for the thesis analysis
    raw = {str(it["item_id"]): {
            "predicted_value": it["predicted_value"],
            **s2[it["item_id"]], **s4[it["item_id"]], **fus[it["item_id"]],
            **(mc.get(it["item_id"], {}))} for it in items}
    with open(os.path.join(args.merlin_dir, "merlin_v1_raw_scores.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print("wrote merlin_v1_raw_scores.json")


if __name__ == "__main__":
    main()
