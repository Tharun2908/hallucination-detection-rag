# Post-thesis MiniCheck v2 synthetic compatibility — cluster result

Operator-reported successful cached replay on 2026-09-28:

- Report SHA256: `eab310d3c23fdcdb08f23217a01862f8177deb6cbf0949b440f17e23904aa797`.
- Saved-tokenizer encoding checks: 14.
- Synthetic examples / sentence generation requests: 4 / 6.
- Status: `ok`; replay `New model attempt: False`.
- All four expected upstream verdicts matched. The fixed TRAIN threshold also
  classified these four examples as expected.

| Synthetic example | Support score | Expected support |
| --- | ---: | --- |
| single-supported | 0.9524445788000317 | true |
| single-contradicted | 0.00406998656094398 | false |
| multi-supported | 0.904393324044711 | true |
| multi-mixed | 0.005219832157014863 | false |

For the multi-supported answer, sentence scores were
`[0.9464691335881718, 0.904393324044711]`; for the mixed answer,
`[0.9464691335881718, 0.005219832157014863]`. The minimum sentence score
correctly determines the whole-answer score for this one-chunk fixture.

This is an observed excerpt, not a reconstruction of the complete private
report. The TEST runner verifies the original report hash, cached file hashes,
resource fingerprints, runtime settings, and exact synthetic prompt tokens
before preparing benchmark inputs. The failed v1 report remains preserved.

Synthetic success establishes compatibility on these requests. It is not a
benchmark performance estimate, a calibration result or proof of historical
runtime equivalence. Fresh MiniCheck TEST scoring is now prepared separately.
