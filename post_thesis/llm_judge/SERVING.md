# Post-thesis Qwen judge: serving and synthetic checks

This is post-thesis engineering. Nothing here changes submitted thesis results.
The adapter and commands have been tested with offline responses; actual H200
startup, driver compatibility, structured decoding, and judge behavior still
require the following GPU smoke check. No benchmark scores have been collected.

## Development profile

| Setting | Pin |
| --- | --- |
| Checkpoint | `Qwen/Qwen3-32B` |
| Model and tokenizer revision | `9216db5781bf21249d130ec9da846c4624c16137` |
| Serving package | `vllm==0.29.0` |
| HTTP client | `httpx==0.28.1` |
| Hardware | One full H200; launcher records observed name/UUID/memory/driver |
| Precision | BF16; no quantization |
| Tensor parallelism / concurrent sequences | 1 / 1 |
| Context limit | 32,768 including formatted input plus output allowance |
| Output allowance | 128 tokens |
| Thinking / temperature / seed | Disabled / 0 / 0 |
| Sampling | Explicit top-p=1, top-k=-1, min-p=0, neutral penalties |
| Generation defaults | `--generation-config vllm`; no checkpoint generation defaults |
| Prefix caching | Disabled for this initial latency measurement |
| Network | Loopback-bound server; no hosted API or routing fallback |

The model card's general non-thinking sampling recommendation differs from our
zero-temperature development choice. We deliberately begin with greedy decoding
for this short score-only task; pilot checks must assess score behavior and
stability. Neither a fixed seed nor greedy decoding guarantees bitwise identical
GPU results. An H200 has enough nominal memory to make this a reasonable target;
actual startup and peak memory have not yet been measured here.

The profile file is the source of truth for request identity and launch settings.
The adapter hashes it together with its own version, origin, and timeout. Changing
these settings invalidates run identity. Returned model aliases and `/version`
are checked, but **an advertised alias does not prove checkpoint identity**. Use
the pinned launcher, retain the server session records, and keep its environment
snapshot with the run. Do not substitute an independently configured endpoint.

## 1. Apply and commit the patch first

The launch/smoke commands record the Git commit and reject tracked uncommitted
changes. Update the checkout on the GPU host after committing and pushing. Keep
client dependencies separate from the existing thesis training environments.

## 2. Install on the Linux H200 host

From the repository root, in a fresh Python 3.12 environment:

```bash
python3.12 -m venv .venv-judge-serving
source .venv-judge-serving/bin/activate
python -m pip install -r post_thesis/llm_judge/requirements-serving.txt
python -m pip install -r post_thesis/llm_judge/requirements-client.txt
python -m post_thesis.llm_judge.serve --print-command
```

`--print-command` does not launch a server or inspect a GPU. Installation alone
does not establish CUDA/driver compatibility. The server log will expose startup
problems; do not silently change the serving version to get past them. Record a
new reviewed profile if an environment correction is necessary.

Create an environment snapshot under the ignored artifact directory:

```bash
mkdir -p .artifacts/post_thesis/llm_judge/environment
python -m pip freeze > .artifacts/post_thesis/llm_judge/environment/pip-freeze.txt
nvidia-smi > .artifacts/post_thesis/llm_judge/environment/nvidia-smi.txt
```

Package pins specify the primary dependencies; `pip-freeze.txt` captures the
resolved transitive environment for the actual run. If using `RAG_WORKSPACE`,
place snapshots under its `post_thesis/llm_judge/` namespace instead.

## 3. Start the pinned server in terminal A

```bash
python -m post_thesis.llm_judge.serve --device 0
```

This command downloads the pinned weights if uncached and uses a GPU. It launches
in the foreground, binds `127.0.0.1:8000`, and prints a private session directory.
Inspect that directory's `server.log` for readiness or startup errors. The server
must be stopped when the smoke check finishes; the client timeout does not stop
an independently running server.

Optional: supply a known per-GPU hourly rate and currency to produce a clearly
labeled estimate, for example `--hourly-rate 2.50 --currency EUR`. This example
is not a quoted price. With no rate, cost remains unknown. Do not use this profile
on A100/B200 by silently overriding settings; record a separate deployment profile.

## 4. Run six synthetic checks in terminal B

Activate the same environment and run from the same committed checkout:

```bash
python -m post_thesis.llm_judge.smoke --run-id qwen3-synthetic-smoke-v1
```

This explicitly makes model calls. It never reads RAGTruth/HaluBench or sends
labels. Cases cover numeric contradiction, an unsupported addition, and a claim
that the passage lacks information. Each pair holds the answer fixed and changes
only its evidence. Expected score directions are evaluated outside the prompt.

Execution is bounded by six new attempts, one attempt per input, a 60-second
whole-attempt timeout and a 300-second runner deadline. Server preflight has its
own 60-second timeout. Oversized requests are rejected before generation after
counting the full chat-template input. Cancellation/timeout cannot guarantee that
server-side work instantly stops; stop the server after the check.

Inspect `summary.json` and the invocation-specific `client-window-*.json` under
`.artifacts/post_thesis/llm_judge/<run_id>/`. Check:

- Six valid scores and no unexpected reasoning text, refusals or truncation.
- Higher unsupported probability for each unsupported member of a pair.
- Actual input/output tokens and recorded model-call latency.
- Server session revision, GPU, launch command, and environment snapshot.

Exit code zero means all six responses parsed successfully; it does **not** mean
all expected score directions were observed or the judge is scientifically valid.
A false direction is a diagnostic finding to inspect. Scores are verbalized
probabilities, not assumed calibrated.

Repeat the same run ID to test cache reuse (zero new successful calls). Use a
new run ID for an intentional fresh stability check; keep those costs separate.
A terminal failed attempt is not retried by resuming this smoke run. After fixing
an operational issue, use a new run ID and retain the failed run's records.

## 5. Stop the server and read resource reports

Press Ctrl+C in terminal A. The wrapper stops its server process group and writes
`resources.json` in its `server-<session-id>/` directory.

| Report | What is measured | What it does not establish |
| --- | --- | --- |
| Server `resources.json` | Observed GPU identity; wall duration of the supervised process, including loading and idle time; GPU count times that duration | GPU utilization, work before launch/after shutdown, provider rounding or the final bill |
| `client-window-*.json` | Client elapsed time, including preflight/cache/error overhead; an allocated-GPU-hour estimate assuming the declared one-GPU server is dedicated | Actual device allocation on a remote server or the complete serving-session duration |
| Runner `summary.json` | Per-attempt latency and token counts, including failures/retries; cached predictions | Fresh inference timing for cache hits or a measured per-example GPU charge |

Client windows overlap their serving session: **do not add the two together**.
Use the server window for whole-session resource reporting. If the wrapper is
forcibly killed, its record may remain `started` with unknown final duration;
do not interpret this as zero cost. An absent API usage field is also unknown.
Reports and raw messages are private local artifacts, not publication summaries.

## 6. What remains before the benchmark pilot

Review the GPU smoke output, record the working environment and resource usage,
then prepare the source TRAIN pilot manifest with the frozen group/overlap rules.
Audit actual formatted lengths, choose the final timeout and resource limits,
and establish a measured pilot compute budget. These six checks do not authorize
or trigger either full benchmark. Preserve the original development prompt and
record any pilot-driven revisions. The final evaluation prompt is still unfrozen.

## Offline verification

```bash
python -m pip install -r post_thesis/llm_judge/requirements-client.txt
python -m unittest discover -s tests -p test_vllm_backend.py -v
python -S -m unittest discover -s tests -p "test_llm_judge*.py" -v
```

The first suite uses `httpx.MockTransport`, not a live server. CI runs the HTTP
suite on Linux and Windows; the core suite remains standard-library-only. A
passing offline suite does not certify that a particular GPU image boots vLLM.

## Official references used for the pins and contract

- [Pinned Qwen model card](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/README.md)
- [vLLM 0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [Pinned chat request protocol](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/entrypoints/openai/chat_completion/protocol.py)
- [Pinned token-counting protocol](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/entrypoints/serve/tokenize/protocol.py)
- [Pinned structured-output documentation](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/features/structured_outputs.md)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
