# Post-thesis S4 CPU compatibility: passed

Evidence: operator-supplied console output. Actual checkpoint files and GPU
inference were not available to the report author. This records engineering
compatibility, not benchmark or thesis performance.

The full report structure was reconstructed from the supplied values and pinned
code identity, reproducing its SHA256 below. This verifies record consistency;
the author did not independently run the checkpoint's forward passes.

Required report SHA256:
`504dccf973f8c9995a02128332dc21a83d97c80c594cd1cbaee8396b11653077`.

The corresponding pushed code revision is
`4581a26d854c3008a3c2e1f3c6ea8899ea934316`.

| Item | Observation |
| --- | --- |
| Python / PyTorch | 3.12.14 / 2.13.0+cu130 |
| Transformers / tokenizers | 5.17.0 / 0.23.2 |
| safetensors / huggingface-hub | 0.8.0 / 1.32.0 |
| Model class | DebertaV2ForSequenceClassification |
| Tokenizer class / vocabulary | DebertaV2Tokenizer / 128,001 |
| Missing, unexpected, mismatched keys; loading errors | All empty |
| Saved-tokenizer pair encoding matches | 4/4 |
| Device / threads | CPU / 2 |
| Padding and truncation sides | Right / right |
| Synthetic forward calls | 4 |

Loaded tokenizer backend SHA256:
`896dbf60bcf96848ef756a84975905a072b156ac2d17082b7ac0077593afc5fc`.

| Artificial input | Unsupported score | Answer tokens full → kept | Context tokens full → kept |
| --- | ---: | ---: | ---: |
| Supported | 0.014060578308999538 | 6 → 6 | 6 → 6 |
| Contradicted | 0.292743444442749 | 6 → 6 | 8 → 8 |
| Long answer | 0.022107597440481186 | 600 → 503 | 6 → 6 |
| Long context | 0.01845911145210266 | 6 → 6 | 600 → 503 |

The contradiction score is higher than the supported score, but remains below
the frozen historical 0.55 operating threshold. This observation is preserved;
the check did not require correct synthetic classifications or retune a threshold.

The long pairs confirm that the actual 512-token longest-first rule can truncate
either answer or context. Fresh inference must record both sides' visibility.
The completed check supports loading these current files in this runtime; it
does not establish historical training/cache provenance or GPU compatibility.
