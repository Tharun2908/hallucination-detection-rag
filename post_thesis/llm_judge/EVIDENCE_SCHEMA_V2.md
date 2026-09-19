# Post-thesis: evidence schema v2 and compatibility checks

**Development only. Excluded from the submitted thesis.** The first
[evidence diagnostic result](../../results/post_thesis/llm_judge/evidence_diagnostic_v1_20260919.md)
has 11/14 valid records. Three supported answers violated the null-field contract;
one accepted unsupported explanation conflated unknown with absence.

Update after this design was committed: the [operator-reported native check](../../results/post_thesis/llm_judge/evidence_schema_v2_native_20260919.md)
passed all 38 cases. A [separate live run plan](EVIDENCE_V2_RUN.md) is now prepared.
The sections below describe the original schema-only patch and its pre-run checks;
statements about that patch having no generation budget are historical.

## Isolated change

`evidence_schema_v2.py` replaces the permissive field combinations with three
mutually exclusive object branches under `anyOf`. Each requires exactly the same
five fields and disallows extra properties:

| Branch | answer_quote | context_quote | explanation |
| --- | --- | --- | --- |
| supported / none | null | null | null |
| unsupported / contradiction | bounded nonempty string | bounded nonempty string | bounded nonempty string |
| unsupported / insufficient_support | bounded nonempty string | bounded nonempty string or null | bounded nonempty string |

This enforces the **field dependencies** already required by the prompt and parser.
It does not enforce exact source membership, semantic correctness, all parser
whitespace rules or whole-response size. Those remain separate checks. The known
misleading explanation is structurally valid under v2 and still needs review.

The [descriptor](evidence_schema_v2.json) records:

- Schema SHA256: `2c933735d69532381a4f45fbe8ad7868a34b03e78478ab8ba0b0e6486bd45807`.
- Request contract: `evidence-diagnostic-request-v2`.
- Schema title: `faithfulness_evidence_verdict_v2`.
- **Unchanged prompt**: `faithfulness-evidence-diagnostic-v1`, SHA256
  `21ba57c9ec668c495a16e4635660c4504f26b5056638d4b17e6424e32ce52f4c`.
- Unchanged parser, evidence character limits, 14 synthetic inputs and model
  candidate. `evidence_request_v2()` preserves messages/configuration and changes
  only the schema and request contract, producing a distinct cache key.

A schema revision is not a prompt-text revision. No v2 inference runner or budget
is enabled by this patch. The original evidence runner still uses schema v1;
**do not rerun it from this reporting commit**. Its completed record is tied to
its scoring revision. Old raw responses are replayed locally only for regression
checks, never changed or counted as newly accepted predictions.

## Independent schema checks

The 38 fixed acceptance fixtures in `schema_v2_checks.py` include positive examples
of every branch, invalid combinations/types/lengths, and all 14 supplied raw
responses. Standard JSON Schema validation accepts all 14 under v1 but only the
same 11 parser-valid records under v2. A 48-combination field-dependency matrix
also agrees with the unchanged parser on the selected valid quote strings.

Tests deliberately show that fabricated source quotes and the misleading
explanation can pass schema validation. A passing schema test does not certify
evidence quality or predict the model's next outputs.

For development tests in a separate CPU/client environment:

```bash
python -m pip install -r post_thesis/llm_judge/requirements-schema.txt
python -m unittest discover -s tests -p test_llm_judge_evidence_schema_v2.py -v
```

CI installs the pinned validator and runs these checks on Linux and Windows.
This requirement file is for the CPU checks; do not change the working GPU
serving environment just to run the separate native check below.

## Native grammar check on the installed cluster environment

[vLLM documents](https://docs.vllm.ai/en/latest/features/structured_outputs/)
that its default structured-output backend is selected automatically. The
previous launcher did not pin a specific structured-output backend, so an
XGrammar test alone cannot attest which backend served the old requests.

The native check uses XGrammar's documented
[compiler](https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/compiler.py),
[tokenizer information](https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/tokenizer_info.py)
and [matcher](https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/matcher.py)
interfaces. These upstream references describe the interfaces; compatibility with
the **actually installed version** must be measured rather than inferred from
current documentation. It is still pending at the time of this patch.

After committing, pushing, verifying CI and pulling on the cluster, run with the
existing serving environment. **A running model server is not needed.**

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.check_evidence_schema
```

The command imports the installed XGrammar package, compiles the exact v2 schema
on CPU, and checks all 38 complete JSON strings with a tiny fixed ASCII byte
vocabulary. It does not load model weights, download tokenizers, make HTTP calls
or generate tokens. A subprocess bounds the check to 60 seconds. It records code
revision, package versions, schema/fixture hashes, individual expected/observed
acceptance and zero generation calls in a new private directory:

`.artifacts/post_thesis/llm_judge/evidence-schema-v2-compatibility-<id>/`.

The result is `passed`, `failed` (acceptance mismatch), or `blocked` (missing or
incompatible native package/API, timeout or other execution failure). A private
worker log is retained. No permissive fallback compiler is substituted, and no
automatic dependency upgrades or model retries occur.

This checks schema compilation and full-string matching with a synthetic
vocabulary. It **does not check the Qwen tokenizer, token-mask generation or the
live vLLM endpoint**. Do not label it end-to-end serving validation. Before a new
bounded model run, record the serving backend choice, matching configuration and
a new budget; then test actual structured decoding with the preserved controls.
If explicit backend pinning changes the prior serving configuration, record that
change as a comparison limitation rather than silently attributing everything
to the schema alone.

## Decision after compatibility evidence

If native checks pass, prepare a separately identified, bounded end-to-end v2
synthetic run, preserving v1. Report contract-valid coverage first, then verdict
and issue matches and manual evidence quality. If the checks fail or are blocked,
inspect the saved compatibility output before modifying the candidate. Neither
outcome changes the historical 11/14 result. No TRAIN/test scoring, threshold
fitting, label changes or final prompt freeze is part of this step.
