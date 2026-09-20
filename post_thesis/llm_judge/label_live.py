"""Pinned synthetic class-score transport; no connections or GPU imports here."""

import hashlib
from importlib.metadata import distribution
import math
from pathlib import Path
import json

from .evidence_cases import case_records, cases_sha256
from .judge import JudgeInput
from .label_score_contract import LABEL_PROMPT, prepare_tokenized_input, score_from_logprobs
from .prompts import PromptSpec, content_hash
from .vllm_backend import load_profile

VERSION = "label-score-synthetic-live-v1"
PROFILE = "label-score-v1"
RUN_ID = "qwen3-label-score-synthetic-v1"
PLAN_PATH = Path(__file__).with_name("configs") / "label_score_synthetic_v1.json"
SOURCE_BLOBS = {
    "vllm/config/model.py": "2b69d7ce4ccf82cb7342c6bc5f4259661974b95e",
    "vllm/v1/sample/sampler.py": "3efaf33d980f42f95184923184e2da8c5a69c7d4",
    "vllm/entrypoints/openai/completion/protocol.py": "4edef74ea7fc45f66be759f07f561bffed73e5d8",
    "vllm/entrypoints/openai/completion/serving.py": "2fa01478cee60ad7a1d2307b769c136c70fe0f32",
}
SWAPPED_PROMPT = PromptSpec(version="faithfulness-label-score-swapped-v1",
    system_text=LABEL_PROMPT.system_text.replace("A = supported:", "B = supported:")
                                          .replace("B = unsupported:", "A = unsupported:"))
ATOL, RTOL = 1e-4, 1e-5


def installed_source():
    dist = distribution("vllm")
    if dist.version != "0.29.0":
        raise ValueError("expected installed vLLM 0.29.0")
    observed = {}
    for name, wanted in SOURCE_BLOBS.items():
        data = dist.locate_file(name).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != wanted:
            raise ValueError(f"installed source differs: {name}")
        observed[name] = actual
    return {"version": dist.version, "git_blob_sha1": observed}


def slots():
    result = []
    for orientation in ("primary", "swapped"):
        for case in case_records():
            result.append({"slot": orientation + ":" + case["sample_id"],
                           "orientation": orientation, "temperature": 0.0, "case": case})
    for kind, temperature in (("repeat", 0.0), ("temperature", 1.0)):
        result.append({"slot": kind + ":" + case_records()[0]["sample_id"],
                       "orientation": "primary", "temperature": temperature,
                       "case": case_records()[0]})
    return result


def prepare_requests(tokenizer):
    """Render all inputs before any model calls. Expected labels stay offline."""
    result = []
    profile = load_profile(PROFILE)
    for slot in slots():
        prompt = LABEL_PROMPT if slot["orientation"] == "primary" else SWAPPED_PROMPT
        item = JudgeInput(**slot["case"]["input"])
        prepared = prepare_tokenized_input(item, tokenizer, prompt=prompt)
        ids = tokenizer.encode(prepared["rendered_prompt"], add_special_tokens=False)
        if prepared["labels"] != {
            "supported": {"text": "A", "token_id": 32, "utf8_hex": "41"},
            "unsupported": {"text": "B", "token_id": 33, "utf8_hex": "42"},
        }:
            raise ValueError("class-token reference mismatch")
        if len(ids) > 4096 or len(ids) + 1 > profile["max_model_len"]:
            raise ValueError("synthetic input limit exceeded; no truncation")
        payload = {"model": profile["served_model_name"], "prompt": ids,
                   "max_tokens": 1, "n": 1, "stream": False, "echo": False,
                   "add_special_tokens": False, "temperature": slot["temperature"],
                   "seed": 0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                   "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "repetition_penalty": 1.0, "ignore_eos": True,
                   "skip_special_tokens": False, "logprobs": 2,
                   "logprob_token_ids": [32, 33], "return_tokens_as_token_ids": True,
                   "return_token_ids": True}
        mapping = {"supported": 32, "unsupported": 33}
        if slot["orientation"] == "swapped":
            mapping = {"supported": 33, "unsupported": 32}
            prepared["labels"] = {"supported": prepared["labels"]["unsupported"],
                                  "unsupported": prepared["labels"]["supported"]}
        identity = {"version": VERSION, "slot": slot["slot"],
                    "prompt_version": prompt.version, "prompt_sha256": prompt.sha256, "profile": profile,
                    "prepared": prepared, "class_mapping": mapping, "payload": payload}
        result.append({"slot": slot["slot"], "sample_id": slot["case"]["sample_id"],
                       "orientation": slot["orientation"], "class_mapping": mapping,
                       "expected_unsupported": slot["case"]["expected_unsupported"],
                       "identity": identity, "key": content_hash(identity), "payload": payload})
    return result


def request_references(requests):
    return [{"slot": r["slot"], "request_key": r["key"],
             "input_tokens": len(r["payload"]["prompt"])} for r in requests]


def plan_descriptor(references):
    return {"study_stage": "post_thesis", "version": VERSION, "run_id": RUN_ID,
            "scope": "synthetic_transport_and_development_only", "profile": load_profile(PROFILE),
            "source_blobs": SOURCE_BLOBS, "cases_sha256": cases_sha256(),
            "prompt_sha256": LABEL_PROMPT.sha256, "swapped_prompt_sha256": SWAPPED_PROMPT.sha256,
            "requests": references, "max_attempts": 30, "attempts_per_slot": 1,
            "client_budget_seconds": 600, "request_timeout_seconds": 60,
            "max_input_tokens_per_request": 4096, "max_input_tokens_total": 122880,
            "max_output_tokens_per_request": 1, "max_output_tokens_total": 30,
            "concurrency": 1, "truncation": "none", "retries": 0,
            "numeric_comparison_atol": ATOL, "numeric_comparison_rtol": RTOL,
            "response_clamp_boundary": -9999.0, "halt_on_any_request_error": True,
            "unknown_window_policy": "charge_full_reserved_remaining_budget",
            "orientation_policy": "primary_fixed_swapped_diagnostic_only_no_selection_or_averaging"}


def load_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan != plan_descriptor(plan["requests"]) or len(plan["requests"]) != 30:
        raise ValueError("changed live label-score plan")
    return plan


def known_usage(data):
    raw = data.get("usage") if isinstance(data, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    return {name: raw.get(field) if type(raw.get(field)) is int and raw[field] >= 0 else None
            for name, field in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"))}


def _logprob(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not -9999 < value <= 0:
        raise ValueError("invalid_or_censored_logprob")
    return value


def parse_response(data, request):
    """Fail closed on alignment, missing values, censored values and unknown usage."""
    payload = request["payload"]
    if data.get("model") != payload["model"]:
        raise ValueError("returned_model_mismatch")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected_one_choice")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise ValueError("choice_index_mismatch")
    prompt = choice.get("prompt_token_ids")
    if (not isinstance(prompt, list) or any(type(t) is not int for t in prompt)
            or prompt != payload["prompt"]):
        raise ValueError("prompt_token_alignment_mismatch")
    emitted = choice.get("token_ids")
    if (not isinstance(emitted, list) or len(emitted) != 1
            or type(emitted[0]) is not int or emitted[0] < 0
            or choice.get("finish_reason") not in ("length", "stop")):
        raise ValueError("expected_one_generated_token")
    logs = choice.get("logprobs")
    if not isinstance(logs, dict):
        raise ValueError("missing_logprobs")
    if (logs.get("tokens") != [f"token_id:{emitted[0]}"] or logs.get("text_offset") != [0]
            or any(not isinstance(logs.get(k), list) or len(logs[k]) != 1
                   for k in ("top_logprobs", "token_logprobs"))):
        raise ValueError("generated_logprob_alignment_mismatch")
    candidates = logs["top_logprobs"][0]
    expected = {f"token_id:{t}" for t in (32, 33, emitted[0])}
    if not isinstance(candidates, dict) or set(candidates) != expected:
        raise ValueError("missing_or_unexpected_selected_token")
    values = {int(k.split(":")[1]): _logprob(v) for k, v in candidates.items()}
    sampled = _logprob(logs["token_logprobs"][0])
    if abs(sampled - values[emitted[0]]) > 1e-6:
        raise ValueError("sampled_token_logprob_mismatch")
    if math.fsum(math.exp(v) for v in values.values()) > 1 + 1e-6:
        raise ValueError("returned_probability_mass_exceeds_one")
    usage = known_usage(data)
    if usage != {"input_tokens": len(prompt), "output_tokens": 1}:
        raise ValueError("usage_missing_or_misaligned")
    score = score_from_logprobs([(32, values[32]), (33, values[33])],
                               supported_token_id=request["class_mapping"]["supported"],
                               unsupported_token_id=request["class_mapping"]["unsupported"])
    return {**score, "emitted_token_id": emitted[0], "off_label": emitted[0] not in (32, 33),
            "score_position": len(prompt), "label_logprobs": {"A": values[32], "B": values[33]}}


def comparisons(predictions):
    rows = {r["slot"]: r for r in predictions}
    baseline = rows.get("primary:attribute-supported-short", {}).get("score")
    checks = []
    for kind in ("repeat", "temperature"):
        other = rows.get(kind + ":attribute-supported-short", {}).get("score")
        available = baseline is not None and other is not None
        deltas = {label: abs(baseline["label_logprobs"][label] - other["label_logprobs"][label])
                  for label in ("A", "B")} if available else None
        matches = (all(math.isclose(baseline["label_logprobs"][label], other["label_logprobs"][label],
                                   abs_tol=ATOL, rel_tol=RTOL) for label in ("A", "B")) if available else None)
        if available and kind == "repeat":
            matches = matches and baseline["emitted_token_id"] == other["emitted_token_id"]
        checks.append({"control": kind, "absolute_logprob_differences": deltas, "matches": matches})
    return checks
