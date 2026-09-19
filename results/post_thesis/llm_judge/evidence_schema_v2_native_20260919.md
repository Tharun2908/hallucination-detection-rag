# Post-thesis: native evidence-schema compatibility check

**Synthetic engineering check only; excluded from the submitted thesis.**
The operator reported that the installed native XGrammar compiler passed all
38 complete-string acceptance checks for evidence schema v2, with zero generation
calls. The [machine-readable observation](evidence_schema_v2_native_20260919.json)
retains the exact versions, hashes and private report location.

| Item | Observed value |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.29.0 |
| XGrammar | 0.2.7 |
| Native check | passed |
| Acceptance matches | 38/38 |
| Generation calls | 0 |

The schema hash is
`2c933735d69532381a4f45fbe8ad7868a34b03e78478ab8ba0b0e6486bd45807`;
the fixture hash is
`85c3e72b532dcfc1664739ad46e03a69a5e1daaa091dcfc3ccbbca961ce7dbc4`.
Both match the committed candidate and acceptance fixtures.

Provenance: operator-pasted console output, following instructions to pull
`7ecd5914e664f535c53265718c75fa3fcb898b2c`. The full private report and worker log
were not supplied, so the actual recorded code revision was not independently
checked. This observation is not a new model result.

The check used a tiny ASCII byte vocabulary, not Qwen's tokenizer. It establishes
native schema compilation and the tested complete-string acceptance behavior.
Token-mask generation, live vLLM decoding, the actual backend selected during
serving and semantic evidence quality remain unverified by this check.

Next: a [separate bounded live synthetic run](../../../post_thesis/llm_judge/EVIDENCE_V2_RUN.md)
with the same 14 inputs, prompt, parser and evidence-v1 model profile. Preserve
the earlier 11/14 valid model outputs and all three terminal failures. No benchmark
examples, continuous scores, threshold fitting or prompt freeze are involved.
