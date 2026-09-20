"""Frozen original-50 TRAIN request preparation; metadata stays outside prompts."""

import json
from pathlib import Path

from .audit_label_pilot import MANIFEST_SHA256, SYNTHETIC_REPORT_SHA256, LABELS, summary
from .label_live import PROFILE, SOURCE_BLOBS, completion_payload, request_references
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input
from .prepare_pilot import pilot_examples
from .prompts import content_hash
from .storage import RunConflict
from .vllm_backend import load_profile

RUN_ID = "qwen3-ragtruth-train-pilot-50-label-score-v1"
AUDIT_SHA256 = "b48d1eca31d68032fff55c38f2f82a038138880c72dad29d6f4dec38ebcf51bc"
AUDIT_REVISION = "c964dc8af892c3f5344c5a3b43c91873460de884"
VERSION = "ragtruth-label-score-pilot-50-v1"
PLAN_PATH = Path(__file__).with_name("configs") / "ragtruth_pilot_50_label_score_v1.json"


def plan_descriptor(references):
    return {"study_stage": "post_thesis", "version": VERSION, "run_id": RUN_ID,
            "scope": "reused_TRAIN_development_only", "pilot_manifest_sha256": MANIFEST_SHA256,
            "audit_sha256": AUDIT_SHA256, "audit_code_revision": AUDIT_REVISION,
            "synthetic_report_sha256": SYNTHETIC_REPORT_SHA256,
            "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
            "profile": load_profile(PROFILE), "source_blobs": SOURCE_BLOBS,
            "class_mapping": {"supported": 32, "unsupported": 33},
            "requests": references, "max_attempts": 50, "attempts_per_slot": 1,
            "client_budget_seconds": 600, "request_timeout_seconds": 60,
            "audited_input_tokens": 60574, "audited_min_input_tokens": 721,
            "audited_max_input_tokens": 2986, "max_output_tokens_per_request": 1,
            "max_output_tokens_total": 50, "concurrency": 1, "retries": 0,
            "truncation": "none", "halt_on_any_request_error": True,
            "unknown_window_policy": "charge_full_reserved_remaining_budget",
            "orientation_policy": "primary_only_no_selection_or_averaging"}


def load_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    references = plan["requests"]
    if (plan != plan_descriptor(references) or len(references) != 50
            or len({r["slot"] for r in references}) != 50
            or len({r["request_key"] for r in references}) != 50
            or sum(r["input_tokens"] for r in references) != 60574
            or min(r["input_tokens"] for r in references) != 721
            or max(r["input_tokens"] for r in references) != 2986):
        raise RunConflict("changed TRAIN label-score plan")
    return plan


def validate_inputs(bundle, audited):
    if bundle.get("manifest_sha256") != MANIFEST_SHA256:
        raise RunConflict("expected original TRAIN manifest")
    examples = pilot_examples(bundle)
    state = audited["audit"]
    if audited.get("audit_sha256") != AUDIT_SHA256 or content_hash(state) != AUDIT_SHA256:
        raise RunConflict("TRAIN token audit checksum mismatch")
    identity = state["identity"]
    expected = {"study_stage": "post_thesis", "version": "label-pilot-token-audit-v1",
                "code_revision": AUDIT_REVISION, "pilot_manifest_sha256": MANIFEST_SHA256,
                "synthetic_report_sha256": SYNTHETIC_REPORT_SHA256,
                "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
                "profile": load_profile(PROFILE), "class_mapping": LABELS,
                "output_allowance": 1, "truncation": "none", "scope": "TRAIN_development_only"}
    if len(examples) != 50 or state["status"] != "completed" or any(identity.get(k) != v for k,v in expected.items()):
        raise RunConflict("TRAIN token audit identity mismatch")
    expected_inputs = [{"sample_id": ex.sample_id,
                        "input_sha256": content_hash({"answer": ex.item.answer, "context": ex.item.context})}
                       for ex in examples]
    if identity["inputs"] != expected_inputs or [r["sample_id"] for r in state["rows"]] != [ex.sample_id for ex in examples]:
        raise RunConflict("TRAIN input order or membership changed")
    if state["summary"] != summary(state["rows"], 50, 32768) or not state["summary"]["all_inputs_fit"]:
        raise RunConflict("incomplete or overlength TRAIN token audit")
    return examples


def prepare_requests(examples, state, tokenizer):
    """Regenerate and compare complete tokenized inputs, with no label access."""
    profile = load_profile(PROFILE)
    result = []
    if len(examples) != len(state["rows"]):
        raise RunConflict("audit coverage mismatch")
    for ex, saved in zip(examples, state["rows"]):
        prepared = prepare_tokenized_input(ex.item, tokenizer)
        if saved != {"sample_id": ex.sample_id, "prepared": prepared} or prepared["labels"] != LABELS:
            raise RunConflict("rendered TRAIN input differs from token audit")
        ids = tokenizer.encode(prepared["rendered_prompt"], add_special_tokens=False)
        if len(ids) + 1 > profile["max_model_len"]:
            raise RunConflict("overlength input; truncation forbidden")
        payload = completion_payload(ids, profile)
        slot = "primary:" + ex.sample_id
        mapping = {"supported": 32, "unsupported": 33}
        identity = {"version": VERSION, "slot": slot, "audit_sha256": AUDIT_SHA256,
                    "prompt_version": LABEL_PROMPT.version, "prompt_sha256": LABEL_PROMPT.sha256,
                    "profile": profile, "prepared": prepared, "class_mapping": mapping, "payload": payload}
        result.append({"slot": slot, "sample_id": ex.sample_id, "orientation": "primary",
                       "class_mapping": mapping, "identity": identity,
                       "key": content_hash(identity), "payload": payload})
    return result


def validate_preparation(preparation, requests, plan):
    if (preparation["audit_sha256"] != plan["audit_sha256"]
            or preparation["source"] != {"version": "0.29.0", "git_blob_sha1": SOURCE_BLOBS}
            or request_references(requests) != plan["requests"]):
        raise RunConflict("TRAIN requests, audit or source differ from frozen plan")
