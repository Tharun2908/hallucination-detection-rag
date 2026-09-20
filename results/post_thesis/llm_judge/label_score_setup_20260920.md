# Post-thesis label-score setup observations — 2026-09-20

Excluded from submitted thesis results. These are operator-supplied console
observations from the restored H200 pod, not a live label-score experiment.
The assistant has not independently read the private cluster report.

- Code revision for the tokenizer check: `e717515375068afd094342593bd737f4dd947791`.
- Prompt: `faithfulness-label-score-v1`, SHA256
  `0b3950a21fb265545d422d1f1ec4b21604d74e715a51ce9261320216ad8fcc3c`.
- Tokenizer: four of four checks passed, A token 32 (`41` in UTF-8 hex),
  B token 33 (`42`). Report SHA256:
  `9965b03b44b6cdfef5e23f1b3c3300a5ba36c423823ae508a6048b0a0661d659`.
- Versions: Python 3.12.14, Transformers 5.17.0, Tokenizers 0.23.2,
  Hugging Face Hub 1.32.0 and Jinja2 3.1.6.
- Tokenizer generation calls: zero; model weights loaded by that check: false.
- The restored runtime reports PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA H200,
  BF16 support and successful vLLM compiled-extension import.
- After installing CUDA development packages, `nvcc` reports 13.0 / V13.0.88
  and `curand.h` is present. The existing default-profile server reached startup
  completion, returned health 200 and advertised `qwen3-32b-judge-9216db5781bf`.
- The live OpenAPI schema advertises explicit `logprob_token_ids`, logprobs and
  token-ID return fields on completion requests. No score request was sent.

The operator then compared four installed vLLM 0.29.0 source files with the
reviewed release blobs; all matched. Exact fingerprints are in the accompanying
[JSON record](label_score_setup_20260920.json) and the new runner's source guard.
The reviewed source computes raw log probabilities before sampling processors,
gathers explicitly requested IDs, and clamps serialized values at -9999. The
planned parser rejects that censoring boundary.

These observations establish setup compatibility only. They do not establish
wire-level score correctness, raw-mode behavior of a particular live request,
semantic performance, calibration or cost. The separate
[bounded synthetic run](../../../post_thesis/llm_judge/LABEL_SCORE_RUN.md) pins an
explicit raw-mode profile and remains unexecuted at the time of this record.
