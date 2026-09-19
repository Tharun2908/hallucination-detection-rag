# Post-thesis experiment: LLM judge for RAG faithfulness

> The experiments below were conducted after thesis submission and are not part
> of the submitted thesis results.

**Status: protocol stage only. No judge implementation, pilot, paid API call, or
new evaluation result is included in this change.** The notice above identifies
the scope of this section; it does not claim completed experiments.

The thesis defence is still pending. Preserve the thesis results and their
provenance throughout this extension.

## Research question

Under a fixed evaluation protocol, when does a prompted LLM judge improve
cross-domain faithfulness detection, and which failures remain shared with
trained verifiers?

Read [PROTOCOL.md](PROTOCOL.md) before implementation. The agreed study design is
recorded there; the model, prompt, budget, and run configuration are not yet
frozen for final evaluation.

## Boundaries and locations

| Content | Location |
| --- | --- |
| Protocol and future judge implementation | This directory, `post_thesis/llm_judge/` |
| New summaries and run manifests | [results/post_thesis/llm_judge/](../../results/post_thesis/llm_judge/) |
| New figures | [figures/post_thesis/llm_judge/](../../figures/post_thesis/llm_judge/) |
| Local request cache and raw responses | `.artifacts/post_thesis/llm_judge/` by default; future runner must preserve `post_thesis/llm_judge/` under any configured artifact root |
| Existing comparison-artifact identities | [baseline_manifest.json](baseline_manifest.json) |

The manifest records the cleaned-up repository baseline and hashes of existing
comparison artifacts. It is **not** a thesis-submission snapshot. The exact
submission commit remains unverified; do not label the current commit as that
snapshot or silently replace historical result files.

Raw benchmark content and API credentials do not belong in committed summaries.
Use IDs, hashes, configuration, and aggregate metrics for public artifacts.

## Sequence

1. Record the protocol and post-thesis boundary — this change.
2. Implement the judge interface, strict parser, offline tests, and resumable runner.
3. Select one model and a bounded API budget; pilot on 50–100 source TRAIN examples.
4. Freeze the prompt and use a separate source development subset for thresholding
   and any declared calibration.
5. Evaluate the frozen system on RAGTruth test and the canonical HaluBench test set.
6. Analyze disagreements and evidence sensitivity, including negative results.
7. Consider a cascade only if the standalone findings justify it.
8. Publish a separately labeled post-thesis report and release.

Model comparison and paid execution follow the protocol stages; there is no
runnable judge command yet.
