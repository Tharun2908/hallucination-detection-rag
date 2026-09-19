# Post-thesis synthetic smoke observations — 2026-09-19

These observations are **not submitted-thesis or benchmark results**. They are
transcribed from the operator's cluster console outputs; the raw private
journals and full environment snapshots remain on the cluster and were not
independently inspected for this report.

Code: `067f91cace07a67dfaa66aaba607f9c02abe8527`. Prompt:
`faithfulness-development-v1`. Profile: `qwen3_32b_h200_development_v1`;
Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, BF16, non-thinking,
temperature 0, seed 0, one H200. Model settings were unchanged between runs.

| Synthetic input | First run | Fresh repeat |
| --- | ---: | ---: |
| numeric-supported | 0.0 | 0.0 |
| numeric-contradiction | 1.0 | 1.0 |
| addition-supported | 0.0 | 0.0 |
| addition-unsupported | 0.5 | 0.5 |
| absence-supported | 0.0 | 0.0 |
| absence-unsupported | 0.0 | 0.0 |

The first run ID was `qwen3-synthetic-smoke-v1`: 6/6 valid scores, six new
attempts. Repeating that same ID reported zero new attempts and identical scores.
The independent run `qwen3-synthetic-smoke-repeat-v1` reported six new attempts
and the same six scores. This establishes repeatability only for these inputs
and two executions, not universal deterministic inference.

The absence pair tied at zero: the answer claimed that the passage did not
report a release date even when the context explicitly supplied one. This is a
missed contradiction despite the prompt's existing absence-claim instruction.
Two of three pairwise direction checks passed; that is not a benchmark accuracy.
The unsupported-addition case moved upward to 0.5. Neither six examples nor their
endpoint-heavy scores establish calibration or its absence.

First-run reported tokens: 1,621 input and 49 output. Per-attempt seconds:
`[1.8411641418933868, 0.22625684505328536, 0.21180603932589293,
0.2612274610437453, 0.2591475429944694, 0.21303852181881666]`.
These are client-observed whole-attempt timings, not GPU kernel timings; the
first request can include warm-up effects. Repeat-run usage/timing was not
provided in the pasted output. Rental cost remains unknown. Do not report these
short synthetic inputs' timing as benchmark latency.

Working stack reported by the operator: Python 3.12.14, vLLM 0.29.0,
PyTorch 2.13.0+cu130, CUDA runtime/toolkit 13.0, H200, driver 580.173.02.
Startup required a CUDA 13-compatible torch build, a host C/C++ compiler and
the CUDA toolkit's `nvcc`. Model cache was moved from the small `/workspace`
volume to `/root/llm-judge-hf-cache`. The operator reported saving pip/GCC/nvcc/OS
and GPU snapshots under the private environment directory.

Decision before benchmark scoring: retain development prompt v1 for the first
TRAIN pilot and preserve this negative finding. Inspect broader development
evidence before revising the prompt. No RAGTruth/HaluBench judge scores exist
in this report; no test thresholds were selected.
