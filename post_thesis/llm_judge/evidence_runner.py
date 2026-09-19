"""Durable one-attempt evidence diagnostic journal, isolated from probability runs."""

from dataclasses import asdict
import json
import math
import time

from .evidence_contract import evidence_request, parse_evidence
from .judge import BackendError, BackendResponse, JudgeInput, TokenUsage
from .parse import JudgeParseError
from .prompts import canonical_json
from .runner import run_directory
from .storage import Journal, RunConflict, atomic_json, exclusive_run


def _parsed(response, request):
    if response.outcome != "completed":
        return response.outcome, None, response.outcome
    try:
        item = JudgeInput(**json.loads(request.messages[1].content))
        return "ok", asdict(parse_evidence(response.text, item)), None
    except JudgeParseError as error:
        return "invalid_output", None, error.code


async def evidence_once(request, backend):
    start = time.perf_counter()
    response = None
    try:
        response = await backend.complete(request)
        if not isinstance(response, BackendResponse):
            raise TypeError("backend must return BackendResponse")
        usage = response.usage
        status, evidence, error = _parsed(response, request)
    except BackendError as exc:
        status, evidence, error, usage = "backend_error", None, exc.code, exc.usage
    return {"study_stage": "post_thesis", "request": asdict(request), "status": status,
            "evidence": evidence, "error_code": error,
            "response": asdict(response) if response is not None else None,
            "usage": asdict(usage), "latency_seconds": time.perf_counter() - start}


def _validated(records, requests):
    stored = {}
    for row in records:
        key = row["request_key"]
        if key not in requests or key in stored or row["ordinal"] != 1:
            raise RunConflict("unexpected or repeated evidence diagnostic attempt")
        if row["state"] == "finished":
            value = row["result"]
            try:
                if value["study_stage"] != "post_thesis" or canonical_json(value["request"]) != canonical_json(asdict(requests[key])):
                    raise ValueError
                if "unsupported_probability" in value:
                    raise ValueError
                latency = value["latency_seconds"]
                if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
                    raise ValueError
                usage = TokenUsage(**value["usage"])
                if value["status"] == "backend_error":
                    BackendError(value["error_code"], usage=usage)
                    if value["response"] is not None or value["evidence"] is not None:
                        raise ValueError
                else:
                    data = dict(value["response"])
                    data["usage"] = TokenUsage(**data["usage"])
                    response = BackendResponse(**data)
                    if response.usage != usage or _parsed(response, requests[key]) != (value["status"], value["evidence"], value["error_code"]):
                        raise ValueError
            except (KeyError, TypeError, ValueError):
                raise RunConflict("cached evidence result violates its contract") from None
        stored[key] = row
    return stored


def _report(examples, by_id, stored, new_attempts):
    predictions = []
    usage, unknown = dict(input_tokens=0, output_tokens=0), dict(input_tokens=0, output_tokens=0)
    latencies = []
    for example in examples:
        row = stored.get(by_id[example.sample_id].key)
        value = row["result"] if row and row["state"] == "finished" else None
        predictions.append({"sample_id": example.sample_id,
                            "status": value["status"] if value else "interrupted" if row else "pending",
                            "evidence": value["evidence"] if value else None,
                            "verdict": value["evidence"]["verdict"] if value and value["evidence"] else None,
                            "error_code": value["error_code"] if value else None,
                            "attempts": int(row is not None)})
        if value:
            latencies.append(value["latency_seconds"])
        if row:
            for field in usage:
                count = value["usage"][field] if value else None
                if count is None:
                    unknown[field] += 1
                else:
                    usage[field] += count
    judged = sum(p["status"] == "ok" for p in predictions)
    return {"study_stage": "post_thesis", "output_contract": "evidence_record_not_probability",
            "examples_total": len(examples), "examples_valid": judged,
            "examples_terminal_failure": sum(p["status"] not in ("ok", "pending") for p in predictions),
            "examples_pending": sum(p["status"] == "pending" for p in predictions),
            "attempts_total": len(stored), "new_attempts_this_invocation": new_attempts,
            "known_token_totals": usage, "attempts_with_unknown_tokens": unknown,
            "completed_attempt_latency_seconds": latencies,
            "attempts_with_unknown_latency": len(stored) - len(latencies), "predictions": predictions}


async def run_evidence(examples, *, run_id, config, backend, revision, dataset_revision,
                     artifact_root=None, max_new_attempts=0, halt_codes=()):
    examples = tuple(examples)
    if type(max_new_attempts) is not int or not 0 <= max_new_attempts <= len(examples):
        raise ValueError("invalid attempt cap")
    if not revision or not dataset_revision:
        raise ValueError("explicit code and data revisions required")
    by_id = {e.sample_id: evidence_request(e.item, config) for e in examples}
    requests = {r.key: r for r in by_id.values()}
    if not examples or len(by_id) != len(examples) or len(requests) != len(examples):
        raise ValueError("evidence diagnostic requires unique IDs and inputs")
    manifest = {"study_stage": "post_thesis", "kind": "evidence_diagnostic", "run_id": run_id,
                "code_revision": revision, "dataset_revision": dataset_revision,
                "max_attempts_per_input": 1, "halt_codes": list(halt_codes),
                "examples": [{"sample_id": sid, "request_key": r.key} for sid, r in by_id.items()],
                "requests": {key: asdict(r) for key, r in requests.items()}}
    directory = run_directory(run_id, artifact_root)
    with exclusive_run(directory):
        journal = Journal(directory / "journal.sqlite3", manifest)
        try:
            _validated(journal.records(), requests)
            journal.recover()
            atomic_json(directory / "manifest.json", manifest)
            stored = _validated(journal.records(), requests)
            new = 0
            # A retained critical failure blocks further calls even if invoked directly.
            blocked = any(r["state"] == "finished" and r["result"]["error_code"] in halt_codes for r in stored.values())
            for example in examples:
                request = by_id[example.sample_id]
                if blocked or new >= max_new_attempts:
                    break
                if request.key in stored:
                    continue
                attempt = journal.start(request.key, 1)
                new += 1
                result = await evidence_once(request, backend)
                journal.finish(attempt, result)
                if result["error_code"] in halt_codes:
                    break
            report = _report(examples, by_id, _validated(journal.records(), requests), new)
            atomic_json(directory / "summary.json", report)
            return report
        finally:
            journal.close()
