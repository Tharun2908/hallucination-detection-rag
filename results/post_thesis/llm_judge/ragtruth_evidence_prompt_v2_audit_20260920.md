# Post-thesis: evidence prompt-v2 TRAIN token audit

Recorded 2026-09-20 from the operator's H200 console output. The private audit
file was not independently inspected here; the scoring runner verifies its
content checksum and per-input request identities before HTTP.

All 50 original TRAIN development inputs were counted, with no failures,
pending entries or overlength inputs. No generation calls were made.

| Quantity | Observed value |
| --- | ---: |
| Total input tokens | 77,724 |
| Minimum / maximum input tokens | 1,064 / 3,329 |
| Output allowance per input | 512 |
| Largest input plus allowance | 3,841 |
| Model limit | 32,768 |

Prompt `faithfulness-evidence-diagnostic-v2`, verdict-first schema v3 and the
`evidence-v1` serving profile stay fixed. Full hashes and the operator summary
are in the [machine-readable record](ragtruth_evidence_prompt_v2_audit_20260920.json).
Audit SHA256: `06569291a3b0b620884ad1cdd213ffa83febc718c827e5a388bb133a888f92d6`.
Audit revision: `6137fb47b5aa65ace9eb5de78a8bea52202850bc`.

The audit establishes input fit only. It does not authorize scoring or measure
faithfulness. The separate [bounded scoring plan](../../../post_thesis/llm_judge/EVIDENCE_PROMPT_V2_PILOT_RUN.md)
allows at most 50 attempts and 25,600 output tokens: 103,324 input plus requested
output tokens in total, excluding repeated tokenization/preflight overhead.
This is an allowance, not actual generation usage or a rental-cost estimate.
Scoring and manual semantic review remain pending. The original 50 remain
adaptive development data, excluded from submitted thesis results.
