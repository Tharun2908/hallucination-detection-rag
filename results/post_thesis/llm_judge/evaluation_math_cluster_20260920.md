# Post-thesis evaluation mathematics: cluster check completed

These are artificial-example engineering checks, not thesis or benchmark results.
Evidence is the operator's console output plus independent reconstruction of the
full report hash from the assistant's locally computed artificial-example report.
The cluster's private report file was not transferred to the assistant.

| Observation | Result |
| --- | --- |
| Code revision | `1e876e85e9010ccb6e06f67944f6c8805ce74814` |
| Shared artificial examples | 12 |
| Registered bootstrap replicates | 2,000 |
| Valid AUROC draws | 2,000 for each of the four systems |
| Unready comparison blocked | Yes |
| Initial invocation / replay | New synthetic evaluation true / false |
| Replay report hash | Identical |
| Benchmark predictions read / model calls / fitting | None |

Contract SHA256:
`c092deea43883442914500fedb0627b7938c513ce5a77bc730079161b4425373`.

Report SHA256:
`e9472bbc2a7d1b0d96247430381e5b0a980de099db806bfac94e901217be9752`.

Independent reconstruction substituted the operator's committed code revision
into the locally computed report identity; canonical hashing then matched the
reported full SHA256. This checks agreement of the artificial numerical result,
not any benchmark prediction.

Windows CI initially failed because Git converted checksum-pinned source files
to CRLF. Commit `2e549508479753b7f9f134b8a59dc5c56ea75d70` preserves LF for those
two files. Both Windows and Linux jobs subsequently passed in
[run 35524726341](https://github.com/Tharun2908/hallucination-detection-rag/actions/runs/35524726341).
The frozen mathematical implementation and registered source hashes are unchanged.

Baseline inference provenance remains the next task. The current final S4
checkpoint and fold-1 tokenizer differ; a local CPU compatibility check is now
provided, with its actual checkpoint-loading result still pending.
