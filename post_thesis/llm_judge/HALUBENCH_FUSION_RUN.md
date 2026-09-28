# Post-thesis frozen fusion on HaluBench

This extension is excluded from submitted thesis results. Apply the already
frozen RAGTruth metadata-free S2+S4 model to the fresh canonical 8,000 HaluBench
rows. No model inference, normalization fitting, coefficient fitting, calibration,
threshold selection or TEST performance metrics occur.

## Completed prerequisites

Fresh S2 and S4 reports and their zero-call replays have completed:

| Artifact | SHA256 |
| --- | --- |
| S2 summary | `e71c27c5c62f2b45f8332ee76c8a9bc6ba9d15915970921178a10daa47b9bb35` |
| S4 summary | `2d68d0657d983db0df84b146264652eea00034734e82d09a42126c56e4ba0f2c` |
| Canonical input manifest | `0e307781570c9cef907023d713a31e798be3289e9150d2250fd59aa27b4bcded` |
| Preserved RAGTruth fusion recovery | `1c90a003abbda9044767bb9167b1ecc171f7a8a746b87093a143d833ba4ec0d6` |

These private files remain under `.artifacts/post_thesis/llm_judge/` in their
original run directories. The command recomputes their hashes and checks all
8,000 feature IDs, contiguous TEST indices and input hashes against the canonical
manifest. Labels and source metadata do not enter the numerical application.

## Fixed transfer behavior

Reuse the exact numerical functions from `apply_fresh_fusion.py` and
`recover_fusion.py`, with byte hashes and LF checkout rules. Use Python
`round(float, 4)` for both raw S2 minimum relevance and S4 probability, then:

- Normalize S2 using the fixed RAGTruth TRAIN range `[-11.43, 10.641]`, clipped to `[0, 1]`.
- Apply coefficients `[-1.3590096505049223, 3.155528234679687]` in `[s2, s4]` order and intercept `-1.4168645142264538`.
- Apply sigmoid and the preserved inclusive threshold `>= 0.45`.

The 1,886 S2 empty-pair examples retain raw relevance `0.0` before normalization;
this is a feature fallback, not a support or hallucination probability. S2 had
35 truncated context pairs; S4 had 850 truncated contexts and no truncated answers.
The policy transfers the fresh RAGTruth pipeline unchanged, rather than recovering
or selecting a historical HaluBench feature representation.

## Run on CPU

```bash
cd /workspace/hallucination-detection-rag &&
git pull --ff-only &&
source .venv-judge-fit/bin/activate &&
python -m post_thesis.llm_judge.apply_halubench_fusion
```

Replay without changing the code revision or source artifacts:

```bash
python -m post_thesis.llm_judge.apply_halubench_fusion
```

Expect `8,000 / 8,000` valid scores, `New output: True` then `False`, and an
identical report hash. New output is saved only in
`.artifacts/post_thesis/llm_judge/halubench-fresh-metadata-free-fusion-v1/report.json`.
A conflicting existing report is preserved and rejected, never overwritten.
The GPU server can remain running. No model download or GPU allocation is needed.

## Interpretation and remaining work

Fresh inference links current TEST inputs and features; it does not establish
historical TRAIN cache/checkpoint provenance. No independent fusion probability
calibration is fitted. The canonical split stays unchanged; exact passages link
994 TEST rows to adaptation components, and this zero-shot run uses no adaptation
training. Preserve those limitations in the final comparison.

Fresh MiniCheck inference, frozen judge inference and the cross-domain evaluation
remain subsequent steps. This command does not declare the full comparison ready.
Local tests use artificial features, including all 8,000 alignment positions;
the private pod predictions are not available in the development workspace.
