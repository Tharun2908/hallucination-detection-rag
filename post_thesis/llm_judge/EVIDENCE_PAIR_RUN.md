# Post-thesis: paired synthetic evidence prompt run

**Adaptive synthetic development only; excluded from the submitted thesis.**
Compare evidence prompts v1 and v2 on exactly the same 26 fixtures. The
[committed plan](configs/evidence_prompt_pair_v1.json) pins both prompt hashes,
the fixture digest, unchanged schema-v3 wire digest and `evidence-v1` profile.
The [completed findings](../../results/post_thesis/llm_judge/evidence_prompt_pair_v1_20260919.md) record all 52 valid outputs and the mixed semantic result. The run completed at revision `6abf9c9eb7e809341c61e5ca62d95c1a3f8fa877`; preserve its artifacts. The commands below are historical and do not provide a new generation allowance at another revision.

The earlier 14 cases are retained alongside 12 new authored controls. References,
expected verdicts and case metadata remain offline: each model request contains
only the fixed prompt and exact answer/context. This comparison does not read
TRAIN manifests, TEST or HaluBench. Its design was informed by the TRAIN pilot;
it is not independent validation. Both prompts use new journals; historical
responses are not relabeled or reused as observations in this paired experiment.

## Fixed limits and execution order

| Limit | Bound |
| --- | ---: |
| Cases per prompt | 26 |
| Generation attempts per case per prompt | 1 |
| Total generation attempts | 52 |
| Concurrent requests | 1 |
| Total client window | 600 seconds |
| HTTP operation timeout | 60 seconds |
| Maximum formatted input per request | 4,096 tokens |
| Maximum aggregate input for generation | 212,992 tokens |
| Output allowance per request | 512 tokens |
| Aggregate output allowance | 26,624 tokens |
| Maximum tokenization requests | 104 |

The input figures are conservative **bounds**, not measured lengths or projected
usage. The runner first audits all 52 formatted requests and saves each count.
If even the last input fails or exceeds its limit, neither prompt generates.
Every generation request is then retokenized and checked against its saved count.
There is no truncation, filtering or substitution. Actual generation usage is
reported separately for each arm; missing usage stays unknown.

Cases follow the frozen fixture order. For case indices 0, 2, 4, ... run v1 then
v2; for indices 1, 3, 5, ... run v2 then v1. Each prompt therefore goes first for
13 cases. The schedule is deterministic and independent of outputs. Prefix
caching remains disabled by the unchanged serving profile. Balanced order does
not eliminate compilation, warmup or other runtime variation; one client window
cannot establish a general latency advantage.

The single 600-second reservation covers preflight, all tokenization, sequential
generation and normal client bookkeeping. Startup and idle server time are
excluded and retained in launcher records. Cleanup can add small wall-clock
overhead beyond a deadline; record actual measured time. Rental cost is unknown.
The server must use the current clean code revision, recorded native-compatible
package versions and unchanged vLLM default auto backend selection.

## Caching, errors and interruption

This synthetic experiment permits **one generation invocation**. Later commands
inspect saved state without HTTP calls, even if the first invocation ended early.
Each attempted case is terminal; failed or interrupted outputs remain missing.
An unfinished window charges the full 600 seconds. It does not yield a new budget
or permission to retry. This deliberately follows the prior synthetic diagnostic
policy; it is not a resumable multi-invocation generation budget.

Overlength or tokenization failures stop the pre-generation audit. Critical model,
token-count, output-limit and selected serving/schema errors halt both arms.
Other per-example failures are retained and the fixed schedule continues. Missing
journals, missing budgets, corrupt records, changed prompts/configuration/code
or request identity mismatches fail closed. Do not delete or rename artifacts to
recover attempts. Schema-valid but wrongly ordered outputs remain valid records
with an independent order mismatch; invalid raw verdicts are never salvaged.

## Historical H200 commands at revision 6abf9c9

Stop the previous launcher with Ctrl+C to finalize its resource record. Restart
the same profile from the newly committed revision in terminal A:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
export CC=/usr/bin/gcc &&
export CXX=/usr/bin/g++ &&
export HF_HOME=/root/llm-judge-hf-cache &&
export CUDA_HOME=/usr/local/cuda-13.0 &&
export PATH="$CUDA_HOME/bin:$PATH" &&
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" &&
python -m post_thesis.llm_judge.serve --device 0 --profile evidence-v1
```

After `Application startup complete`, run in terminal B. Enter the full new
session directory name, **including the `server-` prefix**, when prompted:

```bash
cd /workspace/hallucination-detection-rag &&
source .venv-judge-serving/bin/activate &&
read -r -p "Full current server directory name (server-...): " judge_session &&
python -m post_thesis.llm_judge.evidence_pair --server-session-id "$judge_session"
```

No installation, TRAIN token-audit rerun, manual input editing or model change
is needed. Keep the launcher alive until the comparison completes; then Ctrl+C
finalizes its server resource record. Preserve all three private directories under
`.artifacts/post_thesis/llm_judge/`:

- `qwen3-evidence-prompt-pair-v1/`: shared execution state and `paired_summary.json`.
- `qwen3-evidence-prompt-pair-v1-prompt-v1/`: v1 manifest, journal and arm summary.
- `qwen3-evidence-prompt-pair-v1-prompt-v2/`: v2 manifest, journal and arm summary.

Cache-only inspection from the same committed revision:

```bash
python -m post_thesis.llm_judge.evidence_pair --inspect-only
```

The original offline prompt checker remains a preparation/contract checker; it
still makes no model calls and has no execution plan of its own. This separate
paired runner owns the live plan and budget.

## Review before another experiment

The console summary and paired export were returned and reviewed; see the findings. The private paired summary retains every
case's expected verdict/issue alongside both arms' statuses, predictions and
comparison flags, plus full evidence in each arm report. Missing predictions
have null comparison flags; never drop them from the attempted-case denominator.
The CLI reports valid coverage, verdict and issue matches, field order, and null
context quotes among **valid insufficient-support records** for each arm. It
leaves manual semantic review pending for valid outputs, including supported
answers with null evidence. CLI success means complete valid, correctly ordered
outputs, not that verdicts or explanations are correct.

Review all cases against full fixture inputs, including the prior embedded
absence miss, new paraphrases, support in another passage and nested-record
controls. A correct verdict with an invalid rationale is still a semantic problem.
Report tradeoffs and regressions as well as improvements. More null quotes can
improve structural coverage without improving detection or explanations. Keep
both prompts and the suite fixed through this comparison; no probabilities,
threshold fitting, TRAIN scoring or benchmark freeze is introduced here.
