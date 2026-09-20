# Post-thesis reserved TRAIN token audit: completed cluster result

The cluster run at `9182e27bfdc52b81a1b26f05753bbff5f9a486af` counted all 1,200
reserved inputs. The repeat invocation added zero tokenizations and reported the
same audit SHA256:
`ac101084ab0fa2f76ceaec118eef00ec52c877f5e7dbcc9ee917de1297390fbf`.
Independent reconstruction of the complete report with that revision reproduces
the hash. See the [structured record](development_token_audit_cluster_20260920.json).

| Arm | Inputs | Minimum tokens | Maximum tokens | Total input tokens |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 600 | 654 | 2,695 | 705,198 |
| Operating threshold | 600 | 666 | 2,886 | 746,401 |
| Combined | 1,200 | 654 | 2,886 | 1,451,599 |

Every full prompt fits within 32,768 tokens with one output token reserved.
There are no pending or overlength inputs. No truncation, replacement or altered
membership occurred. The reservation and fitting-protocol hashes match their
frozen records. No model weights were loaded; generation and HTTP requests were
zero. Calibration and threshold selection have not occurred.

Preserve the private audit at its historical revision rather than regenerating
it after a pull. The [bounded scoring plan](../../../post_thesis/llm_judge/DEVELOPMENT_SCORING_RUN.md)
uses this exact audit and two independently budgeted arm runs. Its local preparation
reproduced all 1,200 request fingerprints with the actual pinned tokenizer and
compared every prepared input to the audit. Fourteen focused offline tests passed
for request provenance, arm separation, cumulative budgets, recovery and failures.
That validation made no live inference requests. All work remains post-thesis.
