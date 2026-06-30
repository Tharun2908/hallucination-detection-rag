MERLIN-DDx V1 manual review package

Main file:
- merlin_v1_labeling_sheet.csv

This file contains 180 V1 rationales from 50 MERLIN-DDx cases, scored zero-shot with:
- s4_hall
- fusion_hall
- minicheck_hall

Score interpretation:
- Higher score = more likely unsupported / hallucinated.
- s4_hall = S4 supervised verifier score.
- fusion_hall = metadata-free S2+S4 lightweight fusion score.
- minicheck_hall = MiniCheck-7B hallucination score.

Manual annotation columns to fill:
- evidence_label: supported / unsupported / contradicted / unclear
- rule_label: compliant / non_compliant / unclear
- binary_faithful: 1 iff evidence_label == supported AND rule_label == compliant

Annotation guidance:
- evidence_label should judge whether the generated reason is grounded in the admission note.
- rule_label should judge whether the generated reason correctly justifies the assigned V1 symptom value.
- The automatic verifier scores are expected to align more directly with evidence support than with rule compliance.
