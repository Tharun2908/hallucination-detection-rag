# Post-thesis: native verdict-first order check

**CPU engineering check only; excluded from the submitted thesis.** The operator
reported 44/44 matching acceptance checks with XGrammar 0.2.7, Python 3.12.14,
PyTorch 2.13.0+cu130 and vLLM 0.29.0. There were zero generation calls.
The [observation JSON](evidence_schema_v3_native_20260919.json) preserves versions,
hashes and the private report location. The full private report and worker log
were not supplied; the instructed revision was
`9a98256c81b2b84484274fe667b4bd0cd1845aa1`, not independently verified from artifacts.

| Identity | SHA256 |
| --- | --- |
| Canonical schema | `2c933735d69532381a4f45fbe8ad7868a34b03e78478ab8ba0b0e6486bd45807` |
| Exact serialized schema | `f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33` |
| Fixtures | `4cab9f343e86b6839b5332a858d39911e9e103b41793570c2999c45f440de792` |

The 44 checks cover 38 valid/invalid payloads serialized verdict-first, plus four
otherwise valid branch objects and two original false-positive responses in the
old field order. Passing includes rejecting those six old-order strings. This
establishes the tested native complete-string behavior using the tiny ASCII byte
vocabulary. It does not establish Qwen tokenizer compatibility, actual live
output ordering, the backend selected by vLLM or judgment quality.

The [next bounded live run](../../../post_thesis/llm_judge/EVIDENCE_V3_RUN.md) keeps
the same prompt, parser, model profile and 14 synthetic inputs. Its separate
run ID, request contract, exact schema digest and raw field-order report make
the comparison inspectable. Live results remain pending. Preserve all previous
runs, including v2's two false positives and the unknown-versus-absence
explanation concern.
