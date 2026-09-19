"""Check the candidate's offline identities and authored fixture contracts only."""

from dataclasses import asdict
import json
from pathlib import Path

from .evidence_contract import EVIDENCE_PROMPT, parse_evidence
from .evidence_prompt_v2 import EVIDENCE_PROMPT_V2, evidence_request_prompt_v2
from .evidence_prompt_v2_cases import case_records, cases_sha256, new_cases
from .evidence_schema_v3 import EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_WIRE_SHA256, evidence_request_v3
from .judge import JudgeConfig, JudgeInput
from .prompts import content_hash

DESCRIPTOR = Path(__file__).with_name("evidence_prompt_v2.json")


def check():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    if (descriptor["prompt"] != asdict(EVIDENCE_PROMPT_V2)
            or descriptor["prompt_sha256"] != EVIDENCE_PROMPT_V2.sha256
            or descriptor["previous_prompt_sha256"] != EVIDENCE_PROMPT.sha256
            or descriptor["schema_serialized_sha256"] != EVIDENCE_V3_WIRE_SHA256
            or descriptor["schema_canonical_sha256"] != content_hash(json.loads(EVIDENCE_SCHEMA_V3_JSON))
            or descriptor["fixtures_sha256"] != cases_sha256()
            or descriptor["generation_enabled"] is not False):
        raise ValueError("candidate descriptor mismatch")
    rows = case_records()
    if len(rows) != 26 or len({r["sample_id"] for r in rows}) != 26:
        raise ValueError("expected the frozen 26 distinct synthetic cases")
    config = JudgeConfig("offline", "offline-fixture-model", "offline-only", max_output_tokens=512)
    for row in rows:
        item = JudgeInput(**row["input"])
        old, new = evidence_request_v3(item, config), evidence_request_prompt_v2(item, config)
        changed = {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]}
        if changed != {"messages", "prompt_version", "prompt_sha256"} or old.key == new.key:
            raise ValueError("candidate must change only prompt fields and isolate request keys")
        if old.messages[1] != new.messages[1]:
            raise ValueError("answer/context changed")
    for row in new_cases():
        parsed = parse_evidence(json.dumps(row["authored_reference"]), JudgeInput(**row["input"]))
        if parsed.verdict != row["expected_verdict"] or parsed.issue_type != row["expected_issue_type"]:
            raise ValueError("authored fixture contract mismatch")
    return {"study_stage": "post_thesis", "prompt_version": EVIDENCE_PROMPT_V2.version,
            "prompt_sha256": EVIDENCE_PROMPT_V2.sha256, "schema_version": "v3",
            "serialized_schema_sha256": EVIDENCE_V3_WIRE_SHA256, "fixtures_sha256": cases_sha256(),
            "synthetic_cases": len(rows), "authored_references_checked": len(new_cases()),
            "generation_calls": 0, "model_behavior_tested": False,
            "token_lengths_audited": False, "live_execution_plan": None}


def main():
    print(json.dumps(check(), indent=2))
    print("Offline contract checks only. Authored references are not model outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
