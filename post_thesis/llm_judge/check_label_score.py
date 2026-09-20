"""Validate the frozen label-score descriptor and arithmetic; stdlib only."""

from dataclasses import asdict
import json
from pathlib import Path

from .binary_contract import BINARY_PROMPT
from .label_score_contract import (CONTRACT_VERSION, LABEL_PROMPT, MODEL, REVISION,
    LABELS, EXTRACTION, NUMERIC_TOLERANCE, ASSISTANT_BOUNDARY, score_from_logprobs)
from .prompts import content_hash

DESCRIPTOR = Path(__file__).with_name("label_score_v1.json")


def descriptor():
    reference = json.loads(DESCRIPTOR.with_name("label_tokenizer_reference_v1.json").read_text(encoding="utf-8"))
    return {"study_stage": "post_thesis", "contract_version": CONTRACT_VERSION,
            "prompt": asdict(LABEL_PROMPT), "prompt_sha256": LABEL_PROMPT.sha256,
            "source_binary_prompt_sha256": BINARY_PROMPT.sha256,
            "model_repository": MODEL, "model_revision": REVISION,
            "tokenizer_revision": REVISION, "labels": LABELS,
            "extraction": EXTRACTION, "mass_tolerance": NUMERIC_TOLERANCE,
            "required_assistant_boundary": ASSISTANT_BOUNDARY,
            "generation_enabled": False, "live_execution_plan": None,
            "token_ids": {k: v["token_id"] for k, v in reference["labels"].items()},
            "tokenizer_reference_sha256": content_hash(reference),
            "tokenizer_compatibility": "assistant_CPU_reference_checked_cluster_pending",
            "live_raw_logprob_compatibility": "unverified", "calibrated": False}


def check():
    saved = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    if saved != descriptor():
        raise ValueError("label-score descriptor mismatch")
    # Artificial numbers, not model scores or fitted parameters.
    result = score_from_logprobs([(10, -1001.0), (20, -1000.0)],
                               supported_token_id=10, unsupported_token_id=20)
    if result["unsupported_log_odds"] != 1.0 or not .73 < result["unsupported_score"] < .74:
        raise ValueError("score arithmetic check failed")
    return {"study_stage": "post_thesis", "contract_version": CONTRACT_VERSION,
            "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
            "descriptor_sha256": content_hash(saved), "offline_contract": "passed",
            "tokenizer_compatibility": "not_checked", "generation_calls": 0,
            "model_behavior_tested": False, "live_execution_plan": None}


def main():
    print(json.dumps(check(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
