# Post-thesis reserved TRAIN token audit: local validation

After statistical-design commit `f3caf09251952a2e5320bc51aa79404b385bcbc3`, the
CPU auditor was implemented and run locally on the completed private reservation
with the actual pinned Qwen tokenizer. No synthetic length estimates or tokenizer
substitutions were used for these counts. The four pinned tokenizer/reference
checks passed before auditing. Network access through socket connections was
blocked during this local validation; no inference client or model was constructed.

| Arm | Inputs | Minimum tokens | Maximum tokens | Total input tokens |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 600 | 654 | 2,695 | 705,198 |
| Operating threshold | 600 | 666 | 2,886 | 746,401 |
| Combined | 1,200 | 654 | 2,886 | 1,451,599 |

All inputs fit the pinned 32,768-token model limit with one output token reserved
per example. No input was truncated, dropped or replaced. Counts include the
fixed prompt, answer/context JSON, chat template and non-thinking assistant
boundary. They are tokenization counts, not measured inference usage or a spend
estimate. Later server usage must be reported independently.

The audit saves hashes/lengths after every input, omitting duplicate rendered
text from its cache. Original answer/context remains in the private reservation.
Replay checked the same audit hash with zero new development tokenizations and
no report rewrite. Eight focused tests cover manifest/config drift, strict input
projection, exclusion/membership enforcement, resume across arms, cache identity,
overlength retention and class-token mapping. The existing CPU workflow discovers
them on Windows and Linux.

The [structured record](development_token_audit_local_20260920.json) records exact
identities, tokenizer file hashes, versions and the local audit hash. This is an
uncommitted-implementation check; the cluster result must be recorded separately.
Its full hash will include the eventual commit and software versions. These
counts are not benchmark performance and do not authorize 1,200 scoring calls.
No calibrator or threshold was fitted and no TEST or HaluBench inputs were read.

Next is [cluster reproduction](../../../post_thesis/llm_judge/DEVELOPMENT_TOKEN_AUDIT.md),
then a separate bounded scoring configuration. Preserve the existing reservation;
do not regenerate it under the new commit.
