#!/usr/bin/env python
"""
Efficiency benchmarking for thesis chapter on efficiency vs robustness trade-off.

Measures, for each verification component:
  - Per-example latency (median, p95, p99) at batch=1
  - Throughput (examples/sec) at batch=32 (or vLLM-batched for MiniCheck-7B)
  - Peak GPU memory
  - Disk size

Components:
  - S2: cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-pair aggregation)
  - S4: finetuned DeBERTa at /workspace/signal4_model
  - MiniCheck-7B: bespokelabs/Bespoke-MiniCheck-7B (vLLM)
  - Fusion (S2+S4) and Cascade @ 30%: derived from per-example component medians

Methodology:
  - Fixed sample of 500 RAGTruth test examples (seed=42), reused across all components
  - 20-call warmup discarded before any timing
  - 200 examples timed at batch=1 for per-example latency
  - torch.cuda.synchronize() before/after every timing block
  - Median + IQR + p95/p99 reported (mean is misleading for tail-sensitive systems)

Usage (run on GPU pod):
    # Phase 1: S2 + S4 (~10-15 min)
    nohup python /workspace/efficiency_benchmark.py --phase lightweight \\
        > /workspace/efficiency_lightweight.log 2>&1 &

    # Phase 2: MiniCheck-7B (~30-45 min including model load)
    # NOTE: vLLM holds most of GPU memory — run after lightweight is done.
    nohup python /workspace/efficiency_benchmark.py --phase minicheck \\
        > /workspace/efficiency_minicheck.log 2>&1 &

    # Phase 3: combine + print summary table (instant)
    python /workspace/efficiency_benchmark.py --phase combine

Outputs:
    /workspace/efficiency/lightweight.json
    /workspace/efficiency/minicheck.json
    /workspace/efficiency/combined.json
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


# =============================================================================
# Config
# =============================================================================
N_SAMPLE = 500            # sample of RAGTruth test for benchmarking
N_WARMUP = 20             # warmup calls discarded before timing
N_LATENCY = 200           # examples timed at batch=1
BATCH_THROUGHPUT = 32     # batch size for throughput measurement
SEED = 42

S2_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
S4_MODEL_DIR = "/workspace/signal4_model"
MINICHECK_MODEL = "bespokelabs/Bespoke-MiniCheck-7B"

OUT_DIR = Path("/workspace/efficiency")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def sample_examples():
    """Deterministic sample of N_SAMPLE examples from RAGTruth test."""
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    rng = np.random.RandomState(SEED)
    indices = sorted(rng.choice(len(ds), size=N_SAMPLE, replace=False).tolist())
    samples = []
    for i in indices:
        ex = ds[int(i)]
        samples.append({
            "idx": int(i),
            "context": ex["context"],
            "output": ex["output"],
            "query": ex.get("query", ""),
        })
    return samples


def stat_summary(times_ms):
    arr = np.array(times_ms, dtype=float)
    return {
        "n": int(len(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "iqr_ms": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def disk_size_mb(path_or_repo):
    """Disk size of a directory or HF cache for a repo id, in MB."""
    def _walk_size(root):
        total = 0
        for dirpath, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        return total

    if os.path.isdir(path_or_repo):
        return _walk_size(path_or_repo) / (1024 ** 2)

    folder = "models--" + path_or_repo.replace("/", "--")
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/workspace",  # symlink targets we moved here
    ]
    for c in candidates:
        full = os.path.join(c, folder)
        if os.path.isdir(full):
            return _walk_size(full) / (1024 ** 2)
    return None


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def split_sentences(text):
    """Sentence split with nltk if available, else regex fallback."""
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        import re
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s]


# =============================================================================
# S2: cross-encoder relevance with sentence-pair aggregation
# =============================================================================
def benchmark_s2(samples):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"\n{'=' * 60}\nS2: {S2_MODEL}\n{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(S2_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(S2_MODEL).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())

    def score_one(answer, context):
        """Realistic S2 per-example: split, score all (a_sent, c_sent) pairs in one batch, take min over best-per-answer-sentence."""
        a_sents = split_sentences(answer)
        c_sents = split_sentences(context)
        if not a_sents or not c_sents:
            return 0.0
        pairs_a = [a for a in a_sents for _ in c_sents]
        pairs_c = [c for _ in a_sents for c in c_sents]
        with torch.no_grad():
            inputs = tokenizer(
                pairs_a, pairs_c, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to("cuda")
            logits = model(**inputs).logits.squeeze(-1)
        scores = logits.view(len(a_sents), len(c_sents))
        best_per_a = scores.max(dim=1).values
        return float(best_per_a.min().item())

    # Warmup
    print(f"Warmup ({N_WARMUP} calls)...")
    for i in range(N_WARMUP):
        s = samples[i % len(samples)]
        _ = score_one(s["output"], s["context"])
    free_gpu()

    # Per-example latency (batch=1, end-to-end including sentence splitting)
    print(f"Latency b=1 ({N_LATENCY} examples)...")
    reset_peak_mem()
    times_ms = []
    for i in range(N_LATENCY):
        s = samples[i % len(samples)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = score_one(s["output"], s["context"])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    lat_b1 = stat_summary(times_ms)
    mem_b1 = peak_mem_gb()

    # "Throughput" for S2: each example is internally batched (all sentence pairs).
    # We measure: total time to score N_SAMPLE examples sequentially.
    print(f"Sequential throughput on {N_SAMPLE} examples...")
    reset_peak_mem()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in samples:
        _ = score_one(s["output"], s["context"])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem_seq = peak_mem_gb()

    result = {
        "label": "S2",
        "model": S2_MODEL,
        "n_params_M": n_params / 1e6,
        "disk_MB": disk_size_mb(S2_MODEL),
        "latency_b1": lat_b1,
        "peak_mem_b1_GB": mem_b1,
        "sequential": {
            "n_examples": N_SAMPLE,
            "elapsed_s": elapsed,
            "examples_per_sec": N_SAMPLE / elapsed,
            "per_example_ms": (elapsed / N_SAMPLE) * 1000,
        },
        "peak_mem_seq_GB": mem_seq,
        "per_example_ms_list": times_ms,
        "note": "S2 per-example cost includes sentence splitting + all (a_sent, c_sent) pair scoring",
    }
    del model, tokenizer
    free_gpu()
    return result


# =============================================================================
# S4: finetuned DeBERTa (single forward pass per example)
# =============================================================================
def benchmark_s4(samples):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"\n{'=' * 60}\nS4: {S4_MODEL_DIR}\n{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(S4_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        S4_MODEL_DIR, ignore_mismatched_sizes=True
    ).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())

    # Warmup
    print(f"Warmup ({N_WARMUP} calls)...")
    for i in range(N_WARMUP):
        s = samples[i % len(samples)]
        with torch.no_grad():
            inputs = tokenizer(
                s["output"], s["context"], return_tensors="pt",
                truncation=True, max_length=512
            ).to("cuda")
            _ = model(**inputs)
    free_gpu()

    # Latency batch=1
    print(f"Latency b=1 ({N_LATENCY} examples)...")
    reset_peak_mem()
    times_ms = []
    for i in range(N_LATENCY):
        s = samples[i % len(samples)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            inputs = tokenizer(
                s["output"], s["context"], return_tensors="pt",
                truncation=True, max_length=512
            ).to("cuda")
            _ = model(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    lat_b1 = stat_summary(times_ms)
    mem_b1 = peak_mem_gb()

    # Throughput batch=32
    print(f"Throughput b={BATCH_THROUGHPUT}...")
    reset_peak_mem()
    bsz = BATCH_THROUGHPUT
    n_batches = len(samples) // bsz
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in range(n_batches):
            batch = samples[b * bsz:(b + 1) * bsz]
            inputs = tokenizer(
                [s["output"] for s in batch],
                [s["context"] for s in batch],
                return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to("cuda")
            _ = model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_ex = n_batches * bsz
    mem_b32 = peak_mem_gb()

    result = {
        "label": "S4",
        "model": S4_MODEL_DIR,
        "n_params_M": n_params / 1e6,
        "disk_MB": disk_size_mb(S4_MODEL_DIR),
        "latency_b1": lat_b1,
        "peak_mem_b1_GB": mem_b1,
        "throughput_b32": {
            "n_examples": n_ex,
            "elapsed_s": elapsed,
            "examples_per_sec": n_ex / elapsed,
            "per_example_ms": (elapsed / n_ex) * 1000,
        },
        "peak_mem_b32_GB": mem_b32,
        "per_example_ms_list": times_ms,
    }
    del model, tokenizer
    free_gpu()
    return result


# =============================================================================
# MiniCheck-7B (vLLM)
# =============================================================================
def benchmark_minicheck(samples):
    print(f"\n{'=' * 60}\nMiniCheck-7B (vLLM)\n{'=' * 60}")
    from minicheck.minicheck import MiniCheck

    scorer = MiniCheck(
        model_name="Bespoke-MiniCheck-7B",
        enable_prefix_caching=False,
        cache_dir=None,
    )

    docs = [s["context"] for s in samples]
    claims = [s["output"] for s in samples]

    # Warmup
    print(f"Warmup ({N_WARMUP} examples)...")
    _ = scorer.score(docs=docs[:N_WARMUP], claims=claims[:N_WARMUP])

    # Per-example latency: call score with single (doc, claim) each time.
    # This is the realistic interactive-deployment number.
    print(f"Latency b=1 ({N_LATENCY} examples)...")
    times_ms = []
    for i in range(N_LATENCY):
        d = docs[i % len(docs)]
        c = claims[i % len(claims)]
        t0 = time.perf_counter()
        _ = scorer.score(docs=[d], claims=[c])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
    lat_b1 = stat_summary(times_ms)

    # vLLM batched throughput: pass the full sample at once
    print(f"vLLM batched throughput on {N_SAMPLE} examples...")
    t0 = time.perf_counter()
    _ = scorer.score(docs=docs, claims=claims)
    elapsed = time.perf_counter() - t0

    # vLLM reserves most of GPU upfront — peak mem from torch.cuda is misleading.
    # We report total GPU memory and note vLLM's typical reservation.
    mem_total = None
    if torch.cuda.is_available():
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    result = {
        "label": "MiniCheck-7B",
        "model": MINICHECK_MODEL,
        "n_params_M": 7000,
        "disk_MB": disk_size_mb(MINICHECK_MODEL),
        "latency_b1": lat_b1,
        "throughput_batched": {
            "n_examples": N_SAMPLE,
            "elapsed_s": elapsed,
            "examples_per_sec": N_SAMPLE / elapsed,
            "per_example_ms": (elapsed / N_SAMPLE) * 1000,
        },
        "gpu_total_GB": mem_total,
        "note": "vLLM allocates ~85-90% of GPU memory upfront for KV cache; torch.cuda peak is not meaningful here.",
        "per_example_ms_list": times_ms,
    }
    return result


# =============================================================================
# Combine + summary table
# =============================================================================
def combine_results():
    lw_path = OUT_DIR / "lightweight.json"
    mc_path = OUT_DIR / "minicheck.json"
    if not lw_path.exists() or not mc_path.exists():
        raise FileNotFoundError(
            f"Need both {lw_path} and {mc_path}. "
            f"Run --phase lightweight and --phase minicheck first."
        )

    with open(lw_path) as f:
        lw = json.load(f)
    with open(mc_path) as f:
        mc = json.load(f)

    s2 = lw["S2"]
    s4 = lw["S4"]
    minicheck = mc["MiniCheck-7B"]

    s2_med = s2["latency_b1"]["median_ms"]
    s4_med = s4["latency_b1"]["median_ms"]
    mc_med = minicheck["latency_b1"]["median_ms"]
    fusion_med = s2_med + s4_med  # S2 + S4 sequential at deployment

    # Cascade end-to-end estimate from per-example medians.
    # (For a more accurate number, instrument cascaded_verifier.py with timing.)
    cascade_estimates = {}
    for r in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]:
        ms = fusion_med + r * mc_med
        cascade_estimates[f"{int(r * 100)}%"] = {
            "median_ms_per_example": ms,
            "rel_to_lightweight": ms / fusion_med,
            "rel_to_minicheck": ms / mc_med,
        }

    summary = {
        "S2": {
            "params_M": s2["n_params_M"],
            "disk_MB": s2["disk_MB"],
            "latency_b1_median_ms": s2_med,
            "latency_b1_p95_ms": s2["latency_b1"]["p95_ms"],
            "throughput_seq_eps": s2["sequential"]["examples_per_sec"],
            "peak_mem_b1_GB": s2["peak_mem_b1_GB"],
        },
        "S4": {
            "params_M": s4["n_params_M"],
            "disk_MB": s4["disk_MB"],
            "latency_b1_median_ms": s4_med,
            "latency_b1_p95_ms": s4["latency_b1"]["p95_ms"],
            "throughput_b32_eps": s4["throughput_b32"]["examples_per_sec"],
            "peak_mem_b1_GB": s4["peak_mem_b1_GB"],
        },
        "Fusion_S2+S4": {
            "params_M": s2["n_params_M"] + s4["n_params_M"],
            "latency_b1_median_ms": fusion_med,
            "throughput_estimate_eps": 1000.0 / fusion_med,
        },
        "MiniCheck-7B": {
            "params_M": minicheck["n_params_M"],
            "disk_MB": minicheck["disk_MB"],
            "latency_b1_median_ms": mc_med,
            "latency_b1_p95_ms": minicheck["latency_b1"]["p95_ms"],
            "throughput_batched_eps": minicheck["throughput_batched"]["examples_per_sec"],
            "gpu_total_GB": minicheck.get("gpu_total_GB"),
        },
        "cascade_estimates": cascade_estimates,
    }

    # Pretty-print
    print("\n" + "=" * 80)
    print("EFFICIENCY SUMMARY  (median per-example latency at batch=1)")
    print("=" * 80)
    print(f"{'Component':<22}{'Params (M)':>12}{'Disk (MB)':>12}{'Med (ms)':>12}{'p95 (ms)':>12}")
    print("-" * 70)
    rows = [
        ("S2 (relevance)", s2["n_params_M"], s2["disk_MB"], s2_med, s2["latency_b1"]["p95_ms"]),
        ("S4 (DeBERTa ft)", s4["n_params_M"], s4["disk_MB"], s4_med, s4["latency_b1"]["p95_ms"]),
        ("Fusion (S2+S4)", s2["n_params_M"] + s4["n_params_M"], None, fusion_med, None),
        ("MiniCheck-7B", minicheck["n_params_M"], minicheck["disk_MB"], mc_med, minicheck["latency_b1"]["p95_ms"]),
    ]
    for name, p, disk, med, p95 in rows:
        disk_s = f"{disk:.0f}" if disk else "—"
        p95_s = f"{p95:.1f}" if p95 else "—"
        print(f"{name:<22}{p:>12.1f}{disk_s:>12}{med:>12.2f}{p95_s:>12}")

    print("\nThroughput (batched):")
    print(f"  S2 sequential:        {s2['sequential']['examples_per_sec']:.1f} ex/sec")
    print(f"  S4 batch=32:          {s4['throughput_b32']['examples_per_sec']:.1f} ex/sec")
    print(f"  MiniCheck-7B batched: {minicheck['throughput_batched']['examples_per_sec']:.1f} ex/sec")

    print("\nCascade end-to-end (estimated from per-example medians, b=1):")
    print(f"{'Escalation':<14}{'Med ms/ex':>14}{'vs Lightweight':>18}{'vs MiniCheck':>18}")
    for rate, vals in cascade_estimates.items():
        print(f"{rate:<14}{vals['median_ms_per_example']:>14.2f}"
              f"{vals['rel_to_lightweight']:>17.2f}x"
              f"{vals['rel_to_minicheck']:>17.2f}x")

    out_path = OUT_DIR / "combined.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["lightweight", "minicheck", "combine"])
    args = parser.parse_args()

    if args.phase in ("lightweight", "minicheck"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: "
              f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB")

    if args.phase == "combine":
        combine_results()
        return

    samples = sample_examples()
    print(f"Sampled {len(samples)} examples from RAGTruth test (seed={SEED})")

    if args.phase == "lightweight":
        s2_result = benchmark_s2(samples)
        s4_result = benchmark_s4(samples)
        out = {"S2": s2_result, "S4": s4_result}
        out_path = OUT_DIR / "lightweight.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved {out_path}")

    elif args.phase == "minicheck":
        mc_result = benchmark_minicheck(samples)
        out = {"MiniCheck-7B": mc_result}
        out_path = OUT_DIR / "minicheck.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
