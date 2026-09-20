"""Bounded post-thesis synthetic class-token transport check and cached replay."""

import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

from .judge import BackendError
from .label_live import (PROFILE, RUN_ID, SOURCE_BLOBS, comparisons, installed_source,
                         known_usage, load_plan, parse_response, prepare_requests, request_references)
from .prompts import content_hash
from .run_pilot import charged_seconds, validate_server
from .runner import run_directory
from .serve import code_revision, resource_totals, server_command
from .storage import Journal, RunConflict, atomic_json, exclusive_run, timestamp
from .vllm_backend import VLLMBackend


class LabelBackend(VLLMBackend):
    def __init__(self, *, transport=None):
        super().__init__(profile_name=PROFILE, timeout_seconds=60, transport=transport)

    async def check_endpoint(self):
        result = await self.preflight()
        api = await self._bounded(self._request("GET", "/openapi.json"))
        fields = api.get("components", {}).get("schemas", {}).get("CompletionRequest", {}).get("properties", {})
        required = ("logprobs", "logprob_token_ids", "return_token_ids", "return_tokens_as_token_ids")
        if any(k not in fields for k in required):
            raise BackendError("missing_selected_logprob_api")
        return {**result, "completion_fields": {k: fields[k] for k in required}}

    async def score(self, request):
        return await self._bounded(self._request("POST", "/v1/completions", request["payload"]))


def _save(path, value):
    atomic_json(path, {"sha256": content_hash(value), "value": value})


def _load(path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["sha256"] != content_hash(saved["value"]):
        raise RunConflict("corrupt label-score artifact")
    return saved["value"]


def validate_records(records, requests):
    for request in requests:
        identity = request["identity"]
        if (content_hash(identity) != request["key"]
                or any(identity[k] != request[k] for k in ("slot", "payload", "class_mapping"))):
            raise RunConflict("prepared request identity mismatch")
    keys, seen = {r["key"] for r in requests}, set()
    for row in records:
        if row["request_key"] not in keys or row["request_key"] in seen or row["ordinal"] != 1:
            raise RunConflict("unexpected or duplicate label-score attempt")
        seen.add(row["request_key"])
        if row["state"] == "finished" and row["result"].get("status") not in ("ok", "failed"):
            raise RunConflict("invalid cached attempt status")


def summarize(requests, records, ledger, *, new_attempts=0):
    validate_records(records, requests)
    by_key = {r["request_key"]: r for r in records}
    predictions, totals, unknown = [], {"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 0, "output_tokens": 0}
    for request in requests:
        stored = by_key.get(request["key"])
        result = stored["result"] if stored and stored["state"] == "finished" else None
        status = result["status"] if result else ("interrupted" if stored else "pending")
        score = result.get("score") if result else None
        if result and status == "ok":
            if parse_response(result["response"], request) != score:
                raise RunConflict("cached score disagrees with its response")
        predictions.append({"slot": request["slot"], "sample_id": request["sample_id"],
                            "orientation": request["orientation"], "status": status,
                            "score": score, "expected_unsupported": request["expected_unsupported"],
                            "error": result.get("error") if result else None})
        if stored:
            usage = result.get("usage", {}) if result else {}
            for key in totals:
                value = usage.get(key)
                if value is None:
                    unknown[key] += 1
                elif type(value) is int and value >= 0:
                    totals[key] += value
                else:
                    raise RunConflict("invalid cached token usage")
    checks = comparisons(predictions)
    valid = sum(p["status"] == "ok" for p in predictions)
    pending = sum(p["status"] == "pending" for p in predictions)
    spent = charged_seconds(ledger["windows"], 600)
    report = {"study_stage": "post_thesis", "run_id": RUN_ID, "plan_sha256": content_hash(load_plan()),
              "code_revision": ledger["identity"]["code_revision"],
              "requests_total": len(requests), "valid_scores": valid, "pending": pending,
              "terminal_failures": len(records) - valid, "new_attempts": new_attempts,
              "halt_reason": ledger.get("halt_reason"), "charged_client_seconds": spent,
              "remaining_client_seconds": max(0, 600 - spent),
              "known_token_totals": totals, "unknown_usage_attempts": unknown,
              "predictions": predictions, "numeric_controls": checks,
              "transport_checks_passed": valid == 30 and all(c["matches"] is True for c in checks),
              "calibrated": False, "benchmark_inputs_read": False,
              "semantic_review": "pending", "estimated_cost": None,
              "cost_basis": "unknown_hourly_rate; serving startup/idle in session resources.json",
              "preparation_windows": ledger.get("preparation_windows", []),
              "execution_windows": ledger["windows"]}
    report["report_sha256"] = content_hash(report)
    return report


def _validate_server(record, backend, revision, source):
    validate_server(record, backend, revision)
    if (record.get("command") != server_command(profile_name=PROFILE)
            or record.get("label_score_source") != source
            or record.get("package_versions", {}).get("vllm") != "0.29.0"
            or record.get("package_versions", {}).get("torch") != "2.13.0+cu130"):
        raise RunConflict("restart using the committed label-score-v1 profile and matching source")


async def execute(*, backend, revision, requests, preparation, server_record=None,
                  artifact_root=None, max_new_attempts=30, preparation_seconds=0):
    plan = load_plan()
    if type(max_new_attempts) is not int or not 0 <= max_new_attempts <= 30:
        raise ValueError("max-new-attempts must be an integer from 0 to 30")
    if request_references(requests) != plan["requests"]:
        raise RunConflict("rendered requests differ from frozen plan")
    validate_records([], requests)
    if preparation["source"] != {"version": "0.29.0", "git_blob_sha1": SOURCE_BLOBS}:
        raise RunConflict("installed source mismatch")
    resource_totals(preparation_seconds, 1)
    identity = {"plan": plan, "code_revision": revision, "preparation": preparation,
                "requests_sha256": content_hash(requests)}
    directory = run_directory(RUN_ID, artifact_root)
    with exclusive_run(directory):
        ledger_path, prepared_path = directory / "budget.json", directory / "prepared.json"
        journal_path = directory / "journal.sqlite3"
        exists = [p.exists() for p in (ledger_path, prepared_path, journal_path)]
        if any(exists) and not all(exists):
            raise RunConflict("incomplete run artifacts; do not reset a run")
        if all(exists):
            ledger = _load(ledger_path)
            if ledger["identity"] != identity or _load(prepared_path) != requests:
                raise RunConflict("changed label-score run identity")
        else:
            ledger = {"identity": identity, "windows": [], "halt_reason": None, "preparation_windows": []}
            _save(prepared_path, requests)
            _save(ledger_path, ledger)
        journal = Journal(journal_path, identity)
        try:
            journal.recover()
            records = journal.records()
            validate_records(records, requests)
            if records and not ledger["windows"]:
                raise RunConflict("attempts without a budget reservation")
            if any(r["state"] == "interrupted" for r in records):
                ledger["halt_reason"] = "interrupted_request"
            if any(r["state"] == "finished" and r["result"]["status"] == "failed" for r in records):
                ledger["halt_reason"] = ledger["halt_reason"] or "previous_request_failure"
            spent = charged_seconds(ledger["windows"], 600)
            done = {r["request_key"] for r in records}
            pending = [r for r in requests if r["key"] not in done]
            before = len(records)
            if pending and max_new_attempts and not ledger["halt_reason"] and spent < 600:
                _validate_server(server_record or {}, backend, revision, preparation["source"])
                ledger["preparation_windows"].append({"at": timestamp(), "wall_seconds": preparation_seconds})
                remaining = 600 - spent
                window = {"id": uuid.uuid4().hex, "status": "started", "reserved_seconds": remaining,
                          "started_at": timestamp(), "elapsed_seconds": None,
                          "server_session_snapshot": server_record}
                ledger["windows"].append(window)
                _save(ledger_path, ledger)  # Reserve before any HTTP, including preflight.
                start = time.monotonic()

                async def work():
                    window["preflight"] = await backend.check_endpoint()
                    _save(ledger_path, ledger)
                    for request in pending[:max_new_attempts]:
                        attempt = journal.start(request["key"], 1)  # Durable before sending.
                        response = None
                        attempt_start = time.monotonic()
                        try:
                            response = await backend.score(request)
                            score = parse_response(response, request)
                            result = {"status": "ok", "score": score, "error": None}
                        except (BackendError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                            result = {"status": "failed", "score": None,
                                      "error": getattr(error, "code", str(error))}
                            ledger["halt_reason"] = result["error"]
                        result.update(response=response, usage=known_usage(response),
                                      latency_seconds=time.monotonic() - attempt_start)
                        journal.finish(attempt, result)
                        _save(ledger_path, ledger)
                        atomic_json(directory / "summary.json", summarize(requests, journal.records(), ledger,
                                    new_attempts=len(journal.records()) - before))
                        if ledger["halt_reason"]:
                            break

                try:
                    await asyncio.wait_for(work(), timeout=remaining)
                except BaseException as error:
                    ledger["halt_reason"] = getattr(error, "code", type(error).__name__)
                    if not isinstance(error, (Exception, asyncio.CancelledError, KeyboardInterrupt)):
                        raise
                finally:
                    window.update(status="finished", finished_at=timestamp(),
                                  elapsed_seconds=time.monotonic() - start)
                    window["resources"] = resource_totals(window["elapsed_seconds"], 1)
                    journal.recover()
                    _save(ledger_path, ledger)
            _save(ledger_path, ledger)
            report = summarize(requests, journal.records(), ledger,
                               new_attempts=len(journal.records()) - before)
            atomic_json(directory / "summary.json", report)
            return report
        finally:
            journal.close()


def inspect_cached(revision, artifact_root=None):
    directory = run_directory(RUN_ID, artifact_root)
    if not directory.is_dir():
        raise RunConflict("no cached label-score run")
    with exclusive_run(directory):
        ledger = _load(directory / "budget.json")
        requests = _load(directory / "prepared.json")
        if (ledger["identity"]["code_revision"] != revision or ledger["identity"]["plan"] != load_plan()
                or ledger["identity"]["requests_sha256"] != content_hash(requests)
                or request_references(requests) != load_plan()["requests"]):
            raise RunConflict("cached identity mismatch")
        if not (directory / "journal.sqlite3").is_file():
            raise RunConflict("missing journal")
        journal = Journal(directory / "journal.sqlite3", ledger["identity"])
        try:
            journal.recover()
            report = summarize(requests, journal.records(), ledger)
            atomic_json(directory / "summary.json", report)
            return report
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-session-id")
    parser.add_argument("--inspect", action="store_true", help="Cached results only; no HTTP or tokenizer loading")
    parser.add_argument("--max-new-attempts", type=int, default=30)
    args = parser.parse_args()
    revision = code_revision()
    directory = run_directory(RUN_ID)
    cached = inspect_cached(revision) if directory.is_dir() else None
    if args.inspect or args.max_new_attempts == 0:
        if cached is None:
            parser.error("no cached run")
        report = cached
    elif cached and (cached["pending"] == 0 or cached["halt_reason"] or cached["remaining_client_seconds"] <= 0
                    or cached["terminal_failures"]):
        report = cached
    else:
        if not args.server_session_id or not args.server_session_id.startswith("server-"):
            parser.error("provide the launcher's full server-... directory name")
        server_path = run_directory(args.server_session_id) / "resources.json"
        server_record = json.loads(server_path.read_text(encoding="utf-8"))
        started = time.monotonic()
        from .check_label_tokenizer import (load_tokenizer, inspect_tokenizer, package_versions, verify_reference)
        versions = package_versions()
        source = installed_source()
        tokenizer, files = load_tokenizer(run_directory("label-tokenizer-cache-v1"))
        reference = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
        requests = prepare_requests(tokenizer)
        preparation = {"versions": versions, "tokenizer_files": files,
                       "tokenizer_reference_sha256": reference, "source": source}
        preparation_seconds = time.monotonic() - started

        async def run():
            async with LabelBackend() as backend:
                return await execute(backend=backend, revision=revision, requests=requests,
                                     preparation=preparation, server_record=server_record,
                                     max_new_attempts=args.max_new_attempts,
                                     preparation_seconds=preparation_seconds)
        report = asyncio.run(run())
    print("Valid synthetic scores:", report["valid_scores"], "/", report["requests_total"])
    print("New attempts:", report["new_attempts"])
    print("Terminal failures / pending:", report["terminal_failures"], "/", report["pending"])
    print("Halt reason:", report["halt_reason"])
    print("Transport checks passed:", report["transport_checks_passed"])
    print("Numeric controls:", json.dumps(report["numeric_controls"]))
    print("Known token totals:", report["known_token_totals"])
    print("Unknown usage attempts:", report["unknown_usage_attempts"])
    print("Client seconds charged / remaining:", report["charged_client_seconds"], "/", report["remaining_client_seconds"])
    print("slot,unsupported_log_odds,unsupported_score,log_class_token_mass,emitted_token_id")
    for row in report["predictions"]:
        score = row["score"]
        print(row["slot"], *(score[k] if score else None for k in
              ("unsupported_log_odds", "unsupported_score", "log_class_token_mass", "emitted_token_id")), sep=",")
    print("Private summary:", directory / "summary.json")
    print("Report SHA256:", report["report_sha256"])
    print("Post-thesis synthetic development only. Uncalibrated scores; no benchmark or threshold fitting.")
    return 0 if report["transport_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
