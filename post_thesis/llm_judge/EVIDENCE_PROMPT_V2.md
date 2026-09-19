# Post-thesis: evidence prompt v2 development candidate

**Offline preparation only; excluded from the submitted thesis.** The original
TRAIN evidence pilot produced 37/50 valid records, with quote extraction failures
and incorrect rationales among valid outputs. This candidate responds to those
[recorded findings](../../results/post_thesis/llm_judge/ragtruth_evidence_pilot_v3_20260919.md).
It has not been run on the model or frozen for benchmark evaluation.

## Controlled change

The resolved prompt and checksum are committed in
[`evidence_prompt_v2.json`](evidence_prompt_v2.json). Its version is
`faithfulness-evidence-diagnostic-v2`. It changes instructions to:

- Check every supplied passage before claiming support is absent.
- Respect entity associations in nested records and distinguish reviews from
  universal assertions.
- Accept faithful paraphrases without inventing stronger requirements.
- Distinguish unknown values from explicit false values, and check other evidence
  that can resolve a missing field.
- Copy short contiguous quotes; never reconstruct dictionaries or rewrite answers.
- Prefer the already allowed null context quote for insufficient support; require
  an exact conflict quote for contradiction.
- Make the explanation justify the selected assertion and issue type.

These are a combined prompt revision, not an ablation identifying which sentence
caused any later change. Some rules were already present in v1; clearer wording
is a hypothesis, not evidence of a fix. Preferring null can reduce extraction
errors while yielding less inspectable evidence. Report null-context use among
insufficient-support outputs separately in a future run and still review whether
support actually exists in the full context.

The schema remains **v3**, with identical serialized SHA256
`f86a37ac3bf7ca198663ce266a345ce89018b8d9e52f046d3fb94e7dadf5fa33`,
field order, constraints and parser. The request contract remains
`evidence-diagnostic-request-v3`. Only system-message text and prompt identity
fields change; request keys therefore differ from v1. Historical defaults,
scoring configurations and artifacts remain unchanged. The new candidate is not
registered in existing generation CLIs.

## Synthetic development suite

Keep the original 14 cases verbatim, including the embedded unknown-absence
failure. Add 12 authored cases with new wording and entities:

| Family | New cases |
| --- | ---: |
| Supported paraphrase / conflicting substitution | 2 |
| Same support first, last or absent | 3 |
| Nested review associated with target / different entity | 2 |
| Positive review / unsupported event-hosting addition | 2 |
| Unknown field resolved by another field | 1 |
| Embedded instruction treated as data | 1 |
| Incorrect claim that context omits information | 1 |

The 12 new cases include authored reference outputs and expectations. They are
**not model outputs**. The suite is motivated by TRAIN observations and must be
reported as adaptive synthetic development, not a new holdout or independent
validation set. No benchmark rows, labels, IDs or references are embedded in the
prompt. Example adapters expose only answer/context to requests.

For a later comparison, run **both prompt versions on these same 26 cases** with
identical schema-v3 serialization, `evidence-v1` model profile and serving settings.
That would require a separately recorded paired plan for 52 attempts (one per
case per prompt), fresh token-length checks and independent run identities. The
historical 14-case result is useful provenance but cannot substitute for a matched
26-case comparison. This step adds no live execution plan or scoring budget.

## Review criteria before observing model output

Report all attempted cases, structural validity, exact-quote validity, raw order,
verdict/issue agreement, null-context frequency and raw failures. Preserve missing
outputs; do not salvage verdicts. Review every unsupported rationale against the
full fixture, not just the selected excerpt. Check supported cases against all
claims, and specifically examine paraphrase false positives, cross-passage
omissions, entity leakage, unknown-as-absence errors and the original embedded
absence miss. A schema-valid record or correct verdict alone does not pass the
semantic review. Do not hide regressions behind one aggregate accuracy number.

Hold this candidate and fixture set fixed through that comparison. Record failures
before deciding whether another revision is justified. Do not proceed directly
to TRAIN or full benchmarks on the strength of offline checks. Continuous-score
and calibration design and native-source overlap auditing remain unresolved.

## Offline verification

No server, GPU, dependencies or data downloads are needed:

```bash
python -m post_thesis.llm_judge.check_evidence_prompt_v2
python -m unittest discover -s tests -p test_llm_judge_evidence_prompt_v2.py -v
```

The checker validates recorded identities, prompt-only request differences,
synthetic uniqueness and the 12 authored reference contracts with the existing
parser. It prints `generation_calls: 0`, `model_behavior_tested: false` and
`token_lengths_audited: false`. It does not attest semantic model behavior,
tokenizer/serving compatibility or deployment readiness. Existing CI discovers
the offline tests on Linux and Windows.
