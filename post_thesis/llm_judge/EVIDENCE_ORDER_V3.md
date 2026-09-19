# Post-thesis: verdict-first order diagnostic candidate

**Development only; excluded from the submitted thesis. No new model run is
enabled.** The [schema-v2 result](../../results/post_thesis/llm_judge/evidence_diagnostic_v2_20260919.md)
is 14/14 contract-valid records with two false positives and one additional
misleading explanation. Two original response strings put non-null quotes before
the verdict, excluding the supported branch under the v2 constraints.

## Isolated candidate

`evidence_schema_v3.py` builds an ordered serialization of the v2 schema:

1. `verdict`
2. `issue_type`
3. `answer_quote`
4. `context_quote`
5. `explanation`

Only the `properties` member ordering changes in each of the three branches.
All schema values, branch order, required-field arrays, title, constraints,
parser, prompt text and inputs remain identical. The retained schema title
`faithfulness_evidence_verdict_v2` is intentional to avoid changing another
request variable. The request contract is separately versioned as
`evidence-diagnostic-request-v3`; the new schema string also changes its cache
key even without that new contract version.

The prompt still lists the five fields in its original order. This tests schema
order under the fixed prompt, not a rewritten verdict-first instruction. Any
future prompt-list change would need its own recorded comparison.

Canonical JSON sorts dictionary keys, so reordering a Python literal then using
`canonical_json()` would undo this change. V3 serializes the schema without
sorting; the existing HTTP adapter preserves that ordering through JSON loading
and HTTP serialization. Offline tests inspect the actual mock HTTP request body,
not only the Python object before serialization.

The [descriptor](evidence_schema_v3.json) keeps two distinct hashes:

| Identity | SHA256 |
| --- | --- |
| Canonical schema values, identical to v2 | `2c933735d69532381a4f45fbe8ad7868a34b03e78478ab8ba0b0e6486bd45807` |
| Exact serialized schema, including order | `f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33` |
| Native order-check fixtures | `4cab9f343e86b6839b5332a858d39911e9e103b41793570c2999c45f440de792` |

Do not use the canonical schema digest to distinguish these order variants.
Persist the exact serialized schema and its digest in any future execution plan.

## CPU compatibility step

The installed XGrammar checker now accepts `--schema-version v3`; its default
remains v2. It uses the same compiler defaults, tiny ASCII byte vocabulary,
complete-string matching and 60-second worker timeout as the previous check.
No compiler-order option, model tokenizer or live serving backend is substituted.

The 44 fixtures consist of:

- The original 38 valid/invalid payloads serialized verdict-first.
- Four otherwise valid branch examples in the old alphabetical order.
- The two operator-supplied false-positive raw strings in their original order.

The last six must be rejected **for a fixed verdict-first order experiment**.
They are still valid JSON Schema objects: order is a decoding constraint being
tested, not a new semantic validation rule. A compiler accepting both orders
would fail this experiment's order check without necessarily violating JSON
Schema. Do not reinterpret that failure as evidence that v2's parser was wrong.

After commit, push and CI, on the cluster:

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-serving/bin/activate &&
python -m post_thesis.llm_judge.check_evidence_schema --schema-version v3
```

A model server is unnecessary. The check records package versions, code revision,
both schema hashes, fixture hash and individual acceptance results in a new
private `evidence-schema-v3-compatibility-<id>` directory. No generation calls
occur. A passed result would establish the tested native string-order behavior,
not Qwen token-mask correctness, the actual vLLM-selected backend, correct
verdicts or improved evidence quality. Native and live order behavior are both
pending when this candidate is committed.

If this check passes, prepare a separately budgeted 14-example live diagnostic
and inspect the emitted raw field order as well as validity and judgments. If it
fails, inspect the mismatches before changing serialization or compiler settings.
Changing the serving backend or decoding options would be a separate variable.
The current scoring CLI still accepts only v1/v2; there is no v3 generation plan.
Preserve all existing runs. No TRAIN/test calls or prompt freeze are authorized
by this CPU check.
