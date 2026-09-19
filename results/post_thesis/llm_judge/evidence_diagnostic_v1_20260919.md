# Post-thesis synthetic evidence diagnostic v1 — 2026-09-19

**Synthetic development only. Excluded from the submitted thesis.** These are
14 manually authored controls, not benchmark performance or evidence of transfer
to the original TRAIN pilot. The official result remains **11/14 valid records**.

## Provenance and execution

The operator supplied the execution summary and all 14 raw responses, preserved
in [the accompanying JSON](evidence_diagnostic_v1_20260919.json). Local replay
against the frozen synthetic inputs and unchanged parser reproduces all statuses
and error codes. The private cluster journal and its hashes were not independently
downloaded; the JSON contains the operator's export, not an independent run audit.

- Run: `qwen3-evidence-synthetic-diagnostic-v1`.
- Scoring code: `15e0bb9437437c72abb1e12df9d407130742f173`.
- Prompt: `faithfulness-evidence-diagnostic-v1`, SHA256
  `21ba57c9ec668c495a16e4635660c4504f26b5056638d4b17e6424e32ce52f4c`.
- Schema SHA256: `ecf37ce6cc1775c9af61d7b30754f5fd84c9c7ccf6a8845907a6347cc8807f12`.
- Synthetic cases SHA256: `2e7aee4037a531ff971c47ade95172c3011b9bc8d874da4cea5c124484df4936`.
- Same pinned Qwen3-32B weights/tokenizer, BF16 H200, non-thinking, temperature zero,
  one sequential request at a time; evidence profile with maximum 512 output tokens.
  See the preserved [execution plan](../../../post_thesis/llm_judge/configs/evidence_diagnostic_v1.json).

Input audit completed; all 14 attempts finished, with no pending inputs. The
operator reported **13,571 input + 933 output = 14,504 tokens**, no unknown usage,
and **39.85682361200452 charged client seconds**. Client time includes preflight,
token checks and bookkeeping, excludes server startup/idle time, and is not a
controlled latency comparison. Rental cost is unknown. No unused allowance is
transferred into another run.

## Coverage and judgments

| Group | Total | Valid records | Invalid records | Valid verdict / issue matches |
| --- | ---: | ---: | ---: | ---: |
| Expected supported | 6 | 3 | 3 | 3 / 3 |
| Expected unsupported | 8 | 8 | 0 | 8 / 8 |
| All | 14 | 11 | 3 | 11 / 11 |

Overall valid coverage is 78.57%, supported-control coverage 50%, and unsupported
coverage 100%. Every valid record matches both expected verdict and issue type.
This conditional agreement is not a 100% end-to-end success claim. The three
invalid results have **no usable verdict or issue type** in the official report.

All 14 raw verdict fields happen to match the manually authored expectations.
That is a post-hoc diagnostic observation only: it does not salvage the rejected
records or change the official coverage. No new accuracy or calibration metric
is calculated by treating invalid raw fields as accepted predictions.

## Three contract failures

All three report `inconsistent_supported_output`:

| ID | Raw output issue |
| --- | --- |
| `attribute-supported-short` | Supported verdict with non-null answer quote, context quote and explanation. |
| `numeric-supported-short` | Supported verdict with a non-null explanation, despite null quotes. |
| `absence-explicit-false-short` | Supported verdict with non-null answer quote, context quote and explanation. |

The frozen prompt and parser require all three fields to be null when verdict is
supported and issue type is none. The model violated that contract; the parser
correctly rejected the records. However, the original JSON schema permits each
field to be null or string independently and does not encode the dependencies.
Independent JSON Schema validation confirms that **all 14 raw objects satisfy
the original schema**. Therefore constrained schema decoding did not prohibit
these three combinations. This is a schema/parser consistency gap, not evidence
that the strict parser should be relaxed.

The short/embedded pattern is observed on these controls only. It does not prove
a causal answer-length effect, nor does it show that adding filler reliably
repairs formatting or grounding.

## Evidence-quality review

This is an unblinded analyst review of supplied responses against the frozen
synthetic fixtures, not independent annotation or access to model reasoning.

- Seven of the eight valid unsupported records quote the relevant target and
  context and give a comparison consistent with the intended fixture.
- `attribute-unknown-embedded` has exact quotes and the correct verdict/issue, but
  its explanation says None indicates **unknown or not offered**. The field
  encodes missing information, not an alternative explicit non-availability
  state. This wording is misleading and remains a semantic-quality concern.
- The three valid supported records correctly use null quote/explanation fields;
  no explanation was emitted to assess. Their verdicts match the controls.
- The three rejected supported records remain contract-invalid even though their
  raw verdicts and supportive statements are consistent with the short controls.
  That observation does not repair their output format.

Quote membership, schema validity, verdict correctness and explanation quality
are distinct. The misleading explanation passes exact-quote and field checks;
structural validation cannot establish that the explanation is faithful.

## Decision

Preserve this v1 run, prompt, schema, parser and synthetic inputs unchanged. Add
[a separate schema-only v2 candidate](../../../post_thesis/llm_judge/EVIDENCE_SCHEMA_V2.md)
that encodes the existing supported/unsupported field dependencies. Keep the
prompt text and parser rules fixed to avoid bundling a prompt edit into this
engineering correction. Use a new request identity and later a new run ID.

First check standard JSON Schema behavior and installed native grammar support.
Do not silently replace the historical schema, replay the completed model run,
relabel failures, or advance to the TRAIN pilot yet. A native grammar check is
not proof of live vLLM behavior; serving backend selection and an eventual bounded
end-to-end check still need to be recorded. Schema v2 is not expected to solve
the observed semantic explanation problem by itself.
