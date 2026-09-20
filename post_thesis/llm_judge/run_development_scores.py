"""Bounded post-thesis scoring of separate reserved TRAIN development arms."""

import argparse
import asyncio
import json
import time
import uuid

from .audit_development_tokens import RUN_ID as AUDIT_RUN_ID, ARMS
from .judge import BackendError
from .label_diagnose import LabelBackend, _load, _save, _validate_server, validate_records
from .label_live import installed_source, known_usage, parse_response, request_references
from .development_scoring import (load_plan, prepare_requests, validate_inputs, validate_preparation)
from .prompts import content_hash
from .run_pilot import charged_seconds
from .runner import run_directory
from .serve import code_revision, resource_totals
from .storage import Journal, RunConflict, atomic_json, exclusive_run, timestamp

GLOBAL_LOCK_ID = "qwen3-ragtruth-development-scoring-v1-lock"


def summarize(requests, records, ledger, *, new_attempts=0):
    plan = ledger["identity"]["plan"]
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
                            "score": score,
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
    valid = sum(p["status"] == "ok" for p in predictions)
    pending = sum(p["status"] == "pending" for p in predictions)
    spent = charged_seconds(ledger["windows"], plan["client_budget_seconds"])
    report = {"study_stage": "post_thesis", "run_id": plan["run_id"], "arm": plan["arm"], "plan_sha256": content_hash(plan),
              "code_revision": ledger["identity"]["code_revision"],
              "requests_total": len(requests), "valid_scores": valid, "pending": pending,
              "terminal_failures": len(records) - valid, "new_attempts": new_attempts,
              "halt_reason": ledger.get("halt_reason"), "charged_client_seconds": spent,
              "remaining_client_seconds": max(0, plan["client_budget_seconds"] - spent),
              "known_token_totals": totals, "unknown_usage_attempts": unknown,
              "predictions": predictions, "scope": "reserved_TRAIN_development_only",
              "execution_complete": valid == plan["request_count"],
              "calibrated": False, "TRAIN_inputs_read": True, "TEST_inputs_read": False,
              "threshold_fitting": False, "calibration_fitting": False, "audit_sha256": plan["audit_sha256"],
              "development_manifest_sha256": plan["development_manifest_sha256"],
              "fit_protocol_sha256": plan["fit_protocol_sha256"],
              "semantic_review": "pending", "estimated_cost": None,
              "cost_basis": "unknown_hourly_rate; serving startup/idle in session resources.json",
              "preparation_windows": ledger.get("preparation_windows", []),
              "execution_windows": ledger["windows"]}
    report["report_sha256"] = content_hash(report)
    return report


async def execute(*, arm, backend, revision, requests, preparation, server_record=None,
                  artifact_root=None, max_new_attempts=600, preparation_seconds=0):
    # Both arms share one server: enforce the registered sequential execution
    # even if two clients are launched accidentally. Inspection stays arm-local.
    with exclusive_run(run_directory(GLOBAL_LOCK_ID, artifact_root)):
        return await _execute_arm(arm=arm, backend=backend, revision=revision, requests=requests,
            preparation=preparation, server_record=server_record, artifact_root=artifact_root,
            max_new_attempts=max_new_attempts, preparation_seconds=preparation_seconds)


async def _execute_arm(*, arm, backend, revision, requests, preparation, server_record=None,
                       artifact_root=None, max_new_attempts=600, preparation_seconds=0):
    plan = load_plan(arm)
    if type(max_new_attempts) is not int or not 0 <= max_new_attempts <= plan["max_attempts"]:
        raise ValueError("max-new-attempts exceeds the registered arm limit")
    validate_preparation(preparation, requests, plan)
    validate_records([], requests)
    resource_totals(preparation_seconds, 1)
    identity = {"plan": plan, "code_revision": revision, "preparation": preparation,
                "requests_sha256": content_hash(requests)}
    directory = run_directory(plan["run_id"], artifact_root)
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
            spent = charged_seconds(ledger["windows"], plan["client_budget_seconds"])
            done = {r["request_key"] for r in records}
            pending = [r for r in requests if r["key"] not in done]
            before = len(records)
            if pending and max_new_attempts and not ledger["halt_reason"] and spent < plan["client_budget_seconds"]:
                _validate_server(server_record or {}, backend, revision, preparation["source"])
                ledger["preparation_windows"].append({"at": timestamp(), "wall_seconds": preparation_seconds})
                remaining = plan["client_budget_seconds"] - spent
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


def inspect_cached(revision, arm, artifact_root=None):
    plan = load_plan(arm)
    directory = run_directory(plan["run_id"], artifact_root)
    if not directory.is_dir():
        raise RunConflict("no cached label-score run")
    with exclusive_run(directory):
        ledger = _load(directory / "budget.json")
        requests = _load(directory / "prepared.json")
        if (ledger["identity"]["code_revision"] != revision or ledger["identity"]["plan"] != plan
                or ledger["identity"]["requests_sha256"] != content_hash(requests)
                or content_hash(request_references(requests)) != plan["request_references_sha256"]):
            raise RunConflict("cached identity mismatch")
        validate_preparation(ledger["identity"]["preparation"], requests, plan)
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
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--server-session-id")
    parser.add_argument("--inspect", action="store_true", help="Cache only; no HTTP or tokenizer")
    parser.add_argument("--max-new-attempts", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.max_new_attempts <= 600:
        parser.error("max-new-attempts must be from 0 to 600")
    plan = load_plan(args.arm)
    revision = code_revision()
    directory = run_directory(plan["run_id"])
    cached = inspect_cached(revision, args.arm) if directory.is_dir() else None
    if args.inspect or args.max_new_attempts == 0:
        if cached is None:
            parser.error("no cached TRAIN label-score run")
        report = cached
    elif cached and (cached["pending"] == 0 or cached["halt_reason"] or cached["remaining_client_seconds"] <= 0
                    or cached["terminal_failures"]):
        report = cached
    else:
        if not args.server_session_id or not args.server_session_id.startswith("server-"):
            parser.error("provide the full launcher server-... directory name")
        server_record = json.loads((run_directory(args.server_session_id) / "resources.json").read_text(encoding="utf-8"))
        started = time.monotonic()
        bundle = json.loads((run_directory("ragtruth-train-development-reservation-v1") / "manifest.json").read_text(encoding="utf-8"))
        audited = json.loads((run_directory(AUDIT_RUN_ID) / "audit.json").read_text(encoding="utf-8"))
        examples = validate_inputs(bundle, audited)
        from .check_label_tokenizer import load_tokenizer, inspect_tokenizer, package_versions, verify_reference
        versions = package_versions()
        source = installed_source()
        tokenizer, files = load_tokenizer(run_directory("label-tokenizer-cache-v1"), download=False)
        reference = verify_reference(inspect_tokenizer(tokenizer, files=files, versions=versions))
        audit_identity = audited["audit"]["identity"]
        if any(audit_identity[k] != v for k,v in
               (("versions", versions), ("tokenizer_files", files), ("tokenizer_reference_sha256", reference))):
            raise RunConflict("tokenizer environment differs from frozen audit")
        requests = prepare_requests(examples[args.arm], audited["audit"], tokenizer, args.arm)
        preparation = {"versions": versions, "tokenizer_files": files,
                       "tokenizer_reference_sha256": reference, "source": source,
                       "audit_sha256": audited["audit_sha256"]}
        preparation_seconds = time.monotonic() - started
        async def run():
            async with LabelBackend() as backend:
                return await execute(arm=args.arm, backend=backend, revision=revision, requests=requests,
                                     preparation=preparation, server_record=server_record,
                                     max_new_attempts=args.max_new_attempts,
                                     preparation_seconds=preparation_seconds)
        report = asyncio.run(run())
    print("Post-thesis reserved TRAIN scoring; arm:", args.arm)
    print("Valid scores:", report["valid_scores"], "/", report["requests_total"])
    print("New attempts:", report["new_attempts"])
    print("Terminal failures / pending:", report["terminal_failures"], "/", report["pending"])
    print("Halt reason:", report["halt_reason"])
    print("Known token totals:", report["known_token_totals"])
    print("Unknown usage attempts:", report["unknown_usage_attempts"])
    print("Client seconds charged / remaining:", report["charged_client_seconds"], "/", report["remaining_client_seconds"])
    print("Private summary:", directory / "summary.json")
    print("Report SHA256:", report["report_sha256"])
    print("Post-thesis reserved TRAIN only. Uncalibrated scores; no TEST, calibration/threshold fitting or prompt changes.")
    return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
