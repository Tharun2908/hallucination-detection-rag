# Post-thesis evidence-v3 TRAIN token audit — 2026-09-19

The operator reported successful tokenization of all 50 original TRAIN pilot
inputs at revision `9bb7dc3a020919493aafe27e5662a50f31f7d8c0`. This record
transcribes console output; the private audit file was not independently
inspected here. The scoring runner checks that file against the pinned digest.

| Observation | Value |
| --- | ---: |
| Counted / selected inputs | 50 / 50 |
| Failed, interrupted, pending or overlength | 0 |
| Total formatted input tokens | 77,574 |
| Minimum / maximum input tokens | 1,061 / 3,326 |
| Output allowance per input | 512 |
| Maximum input plus output allowance | 3,838 |
| Model context limit | 32,768 |
| Generation calls | 0 |

Audit SHA256: `d08eede188c9308e2a77cc9e205e324c0cace58b293b05afa965e24a02f48939`.
The [machine-readable observation](ragtruth_evidence_v3_audit_20260919.json)
retains the prompt, schema and revision identifiers.

All inputs fit unchanged. Tokenization establishes input fit, not faithfulness,
calibration or evidence quality. This audit did not authorize scoring. A separate
[bounded execution plan](../../../post_thesis/llm_judge/EVIDENCE_PILOT_RUN.md)
now records the next 50-example development run. No results from that run exist
in this record. These experiments are excluded from the submitted thesis.
