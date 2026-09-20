"""Frozen request preparation for two reserved post-thesis TRAIN arms."""

from dataclasses import asdict
import json
from pathlib import Path

from .audit_development_tokens import (ARMS, FIT_SHA256, MANIFEST_SHA256, LABELS,
    load_fit_protocol, validate_inputs as validate_reservation, summarize as audit_summary)
from .label_live import PROFILE, SOURCE_BLOBS, completion_payload, request_references
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input
from .prompts import content_hash
from .storage import RunConflict
from .vllm_backend import load_profile

AUDIT_SHA256 = "ac101084ab0fa2f76ceaec118eef00ec52c877f5e7dbcc9ee917de1297390fbf"
AUDIT_REVISION = "9182e27bfdc52b81a1b26f05753bbff5f9a486af"
VERSION = "reserved-development-label-scoring-v1"
PLAN_PATH = Path(__file__).parent / "configs" / "development_label_scoring_v1.json"
PLAN_SHA256 = "4aa67f2f0df10d0e6fa7994e11cf5e01a745c06c87e90daf070033b6a36f08de"
COUNTS = {"calibration": (705198, 654, 2695), "operating_threshold": (746401, 666, 2886)}


def plan_descriptor(arm, references):
    total, minimum, maximum = COUNTS[arm]
    return {"study_stage": "post_thesis", "version": VERSION, "arm": arm,
            "run_id": "qwen3-ragtruth-development-" + arm.replace('_', '-') + "-label-v1",
            "scope": "reserved_TRAIN_development_only", "development_manifest_sha256": MANIFEST_SHA256,
            "fit_protocol_sha256": FIT_SHA256, "audit_sha256": AUDIT_SHA256,
            "audit_code_revision": AUDIT_REVISION, "prompt_version": LABEL_PROMPT.version,
            "prompt_sha256": LABEL_PROMPT.sha256, "profile": load_profile(PROFILE),
            "source_blobs": SOURCE_BLOBS, "class_mapping": {"supported": 32, "unsupported": 33},
            "request_references_sha256": content_hash(references), "request_count": 600,
            "max_attempts": 600, "attempts_per_slot": 1, "client_budget_seconds": 1800,
            "request_timeout_seconds": 60, "audited_input_tokens": total,
            "audited_min_input_tokens": minimum, "audited_max_input_tokens": maximum,
            "max_output_tokens_per_request": 1, "max_output_tokens_total": 600,
            "concurrency": 1, "retries": 0, "truncation": "none", "halt_on_any_request_error": True,
            "unknown_window_policy": "charge_full_reserved_remaining_budget",
            "orientation_policy": "primary_only_no_selection_or_averaging",
            "fit_or_threshold_selection": False}


def load_plan(arm):
    if arm not in ARMS:
        raise ValueError("choose a registered development arm")
    plans = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if content_hash(plans) != PLAN_SHA256 or set(plans) != set(ARMS):
        raise RunConflict("changed development scoring plans")
    return plans[arm]


def validate_inputs(bundle, audited):
    examples = validate_reservation(bundle, load_fit_protocol())
    state = audited["audit"]
    if audited.get("audit_sha256") != AUDIT_SHA256 or content_hash(state) != AUDIT_SHA256:
        raise RunConflict("development token audit checksum mismatch")
    identity = state["identity"]
    expected = {"study_stage": "post_thesis", "version": "development-label-token-audit-v1",
                "code_revision": AUDIT_REVISION, "development_manifest_sha256": MANIFEST_SHA256,
                "fit_protocol_sha256": FIT_SHA256, "prompt_version": LABEL_PROMPT.version,
                "prompt_sha256": LABEL_PROMPT.sha256, "profile": load_profile(PROFILE),
                "class_mapping": LABELS, "output_allowance": 1, "truncation": "none",
                "scope": "reserved_TRAIN_development_only"}
    if state["status"] != "completed" or any(identity.get(k) != v for k, v in expected.items()):
        raise RunConflict("development token audit identity mismatch")
    inputs = [{"partition": arm, "sample_id": ex.sample_id, "input_sha256": content_hash(asdict(ex.item))}
              for arm in ARMS for ex in examples[arm]]
    if identity["inputs"] != inputs or len(state["rows"]) != len(inputs):
        raise RunConflict("development token audit membership/order mismatch")
    for expected_input, row in zip(inputs, state["rows"]):
        if any(row[k] != expected_input[k] for k in ("partition", "sample_id")) or row["prepared"]["input_sha256"] != expected_input["input_sha256"]:
            raise RunConflict("development token audit row mismatch")
    if (state["summary"] != audit_summary(state["rows"], examples, 32768)
            or not state["summary"]["all_inputs_fit"]):
        raise RunConflict("incomplete or overlength development token audit")
    return examples


def prepare_requests(examples, state, tokenizer, arm):
    if arm not in ARMS:
        raise ValueError("unknown development arm")
    saved_rows = [r for r in state["rows"] if r["partition"] == arm]
    if len(examples) != len(saved_rows):
        raise RunConflict("arm audit coverage mismatch")
    profile, requests = load_profile(PROFILE), []
    for ex, saved in zip(examples, saved_rows):
        prepared = prepare_tokenized_input(ex.item, tokenizer)
        compact = {k: v for k, v in prepared.items() if k != "rendered_prompt"}
        if saved != {"partition": arm, "sample_id": ex.sample_id, "prepared": compact} or prepared["labels"] != LABELS:
            raise RunConflict("rendered development input differs from frozen token audit")
        ids = tokenizer.encode(prepared["rendered_prompt"], add_special_tokens=False)
        if len(ids) + 1 > profile["max_model_len"]:
            raise RunConflict("overlength input; truncation forbidden")
        payload = completion_payload(ids, profile)
        slot = arm + ":" + ex.sample_id
        mapping = {"supported": 32, "unsupported": 33}
        identity = {"version": VERSION, "arm": arm, "slot": slot,
                    "development_manifest_sha256": MANIFEST_SHA256, "fit_protocol_sha256": FIT_SHA256,
                    "audit_sha256": AUDIT_SHA256, "prompt_version": LABEL_PROMPT.version,
                    "prompt_sha256": LABEL_PROMPT.sha256, "profile": profile,
                    "prepared": compact, "class_mapping": mapping, "payload": payload}
        requests.append({"slot": slot, "sample_id": ex.sample_id, "orientation": "primary",
                         "class_mapping": mapping, "identity": identity,
                         "key": content_hash(identity), "payload": payload})
    return requests


def validate_preparation(preparation, requests, plan):
    refs = request_references(requests)
    if (preparation["audit_sha256"] != plan["audit_sha256"]
            or preparation["source"] != {"version": "0.29.0", "git_blob_sha1": SOURCE_BLOBS}
            or len(refs) != plan["request_count"]
            or len({r["slot"] for r in refs}) != len(refs)
            or len({r["request_key"] for r in refs}) != len(refs)
            or content_hash(refs) != plan["request_references_sha256"]
            or sum(r["input_tokens"] for r in refs) != plan["audited_input_tokens"]
            or any(r["slot"] != plan["arm"] + ":" + r["sample_id"]
                   or r["orientation"] != "primary" for r in requests)):
        raise RunConflict("development requests, audit or source differ from frozen arm plan")
