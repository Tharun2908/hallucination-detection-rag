# Infrastructure Notes



This project was developed and evaluated on Kubernetes GPU pods with persistent storage mounted at `/workspace`.



## Verified environments



The original S4, MiniCheck, and efficiency experiments were run on a Tesla V100S-PCIE-32GB GPU environment. The later out-of-fold S4 fusion protocol was run on an NVIDIA H200 GPU pod to reduce turnaround time.



## Storage layout



Intermediate per-example score files were written to `/workspace`, including:



\- `/workspace/relevance\_results\_train\_v2.json`

\- `/workspace/relevance\_results\_test\_v2.json`

\- `/workspace/signal4\_results\_train\_oof.json`

\- `/workspace/signal4\_results\_test.json`

\- `/workspace/minicheck\_results\_test.json`



These files are not all stored in Git because they are intermediate artifacts. Aggregate result JSONs are stored under `results/`.



## Dependency notes



The verified stack uses pinned versions because vLLM, PyTorch, Transformers, and xFormers compatibility was fragile on the V100 environment.



Important constraints:



\- `transformers==4.44.0` was used for reliable S4 checkpoint loading.

\- `vllm==0.4.3` was used for MiniCheck-related runs.

\- `xformers==0.0.26.post1` was used for V100 compatibility.

\- HuggingFace model caches should be placed on persistent storage rather than the container overlay.



## Out-of-fold fusion protocol



The final fusion protocol uses out-of-fold S4 predictions for the RAGTruth training split. Each training example is scored by an S4 fold model that did not train on that example. The final RAGTruth test scores still use the full-train S4 checkpoint.



This avoids training the logistic-regression fusion model on in-sample S4 features.

