\# Infrastructure Notes



This project was developed and evaluated on a Kubernetes GPU environment with persistent storage mounted at `/workspace`.



\## Verified environment



\- GPU: Tesla V100S-PCIE-32GB

\- CUDA/PyTorch stack: see `requirements.txt`

\- Main constraints:

&#x20; - MiniCheck-7B requires large model cache storage.

&#x20; - vLLM, torch, xformers, and transformers versions are pinned because later versions caused checkpoint-loading or GPU compatibility issues.

&#x20; - Long-running scoring jobs were executed on GPU pods and wrote intermediate score files to persistent storage.



\## Reproduction notes



The repository includes code and aggregate metric JSONs. Full reproduction requires regenerating intermediate per-example score files, including S2, S4, MiniCheck, fusion, cascade, and HaluBench outputs.



\## Known setup traps



\- Keep `transformers==4.44.0`.

\- Keep `vllm==0.4.3`.

\- Keep `xformers==0.0.26.post1` for V100 compatibility.

\- Move HuggingFace model cache to persistent storage when using MiniCheck-7B.

