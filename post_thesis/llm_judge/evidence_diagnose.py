"""One bounded post-thesis synthetic evidence invocation; replay is cache-only."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import time

from .evidence_cases import EVIDENCE_CASE_VERSION, case_records, cases_sha256, examples
from .evidence_contract import EVIDENCE_CONTRACT_VERSION, EVIDENCE_PROMPT, EVIDENCE_SCHEMA_JSON, evidence_request
from .evidence_runner import run_evidence
from .evidence_schema_v2 import EVIDENCE_V2_CONTRACT_VERSION, EVIDENCE_SCHEMA_V2_JSON, evidence_request_v2
from .evidence_schema_v3 import (EVIDENCE_V3_CONTRACT_VERSION, EVIDENCE_SCHEMA_V3_JSON,
                                 EVIDENCE_V3_FIELD_ORDER, evidence_request_v3)
from .judge import BackendError
from .prompts import content_hash
from .run_pilot import HALT_CODES, validate_server
from .runner import run_directory
from .serve import code_revision, resource_totals, server_command
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import VLLMBackend

PLAN_PATH = Path(__file__).with_name("configs") / "evidence_diagnostic_v1.json"
EVIDENCE_HALT_CODES = HALT_CODES + ("diagnostic_input_limit", "invalid_token_count", "input_too_long")
EVIDENCE_V2_HALT_CODES = EVIDENCE_HALT_CODES + ("http_400", "http_422", "http_500")


def load_plan(schema_version="v1"):
    if schema_version not in ("v1", "v2", "v3"):
        raise ValueError("unknown evidence schema version")
    path = PLAN_PATH if schema_version == "v1" else PLAN_PATH.with_name("evidence_diagnostic_" + schema_version + ".json")
    return json.loads(path.read_text(encoding="utf-8"))


def native_observation(schema_version="v2"):
    if schema_version not in ("v2", "v3"):
        raise ValueError("no native observation for this schema")
    path = Path(__file__).resolve().parents[2] / ("results/post_thesis/llm_judge/evidence_schema_" + schema_version + "_native_20260919.json")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v2_server(record, schema_version="v2"):
    # Keep the previous launcher policy (automatic backend selection). Do not
    # infer which compiler served a request from the installed package list.
    if (record.get("package_versions") != native_observation(schema_version)["versions"]
            or record.get("command") != server_command(profile_name="evidence-v1")):
        raise RunConflict("schema " + schema_version + " needs a fresh matching environment and recorded default launcher command")


def comparisons(report):
    by_id = {p["sample_id"]: p for p in report["predictions"]}
    result = []
    for case in case_records():
        prediction = by_id[case["sample_id"]]
        evidence = prediction["evidence"]
        result.append({"sample_id": case["sample_id"], "status": prediction["status"],
                       "expected_verdict": case["expected_verdict"],
                       "observed_verdict": prediction["verdict"],
                       "expected_issue_type": case["expected_issue_type"],
                       "observed_issue_type": evidence["issue_type"] if evidence else None,
                       "verdict_matches_expected": None if evidence is None else evidence["verdict"] == case["expected_verdict"],
                       "issue_matches_expected": None if evidence is None else evidence["issue_type"] == case["expected_issue_type"],
                       "semantic_evidence_review": "pending" if evidence else "not_available"})
    return result


class PrecountedBackend:
    def __init__(self, backend, counts, halt_codes=EVIDENCE_HALT_CODES):
        self.backend, self.counts, self.halt_code = backend, counts, None
        self.halt_codes = halt_codes

    async def complete(self, request):
        row = self.counts.get(request.key)
        if row is None or row["status"] != "ok":
            raise RunConflict("generation requires a completed input audit")
        try:
            return await self.backend.complete_counted(request, expected_input_tokens=row["input_tokens"])
        except BackendError as error:
            if error.code in self.halt_codes:
                self.halt_code = error.code
            raise


def _save(path, state):
    atomic_json(path, {"execution_sha256": content_hash(state), "execution": state})


def validate_plan(plan, backend, cases, schema_version="v1"):
    if schema_version not in ("v1", "v2", "v3"):
        raise RunConflict("unknown evidence schema version")
    contract = {"v1": EVIDENCE_CONTRACT_VERSION, "v2": EVIDENCE_V2_CONTRACT_VERSION,
                "v3": EVIDENCE_V3_CONTRACT_VERSION}[schema_version]
    schema = {"v1": EVIDENCE_SCHEMA_JSON, "v2": EVIDENCE_SCHEMA_V2_JSON,
              "v3": EVIDENCE_SCHEMA_V3_JSON}[schema_version]
    if schema_version in ("v2", "v3"):
        if schema_version == "v2":
            from .schema_v2_checks import checks_sha256
            expected_checks = 38
        else:
            from .schema_v3_checks import checks_sha256
            expected_checks = 44
        observation = native_observation(schema_version)
        if (plan.get("native_observation_sha256") != content_hash(observation)
                or observation["status"] != "passed" or observation["matching_acceptance_checks"] != expected_checks
                or observation["checks_total"] != expected_checks or observation["generation_calls"] != 0
                or observation["schema_sha256"] != content_hash(json.loads(schema))
                or observation["fixture_sha256"] != checks_sha256()
                or plan.get("structured_output_backend_policy") != "unchanged_vllm_default_auto"):
            raise RunConflict("changed native compatibility evidence or serving policy")
    if schema_version == "v3":
        serialized_hash = hashlib.sha256(schema.encode("utf-8")).hexdigest()
        if (plan.get("serialized_schema_sha256") != serialized_hash
                or observation.get("serialized_schema_sha256") != serialized_hash
                or plan.get("expected_response_field_order") != list(EVIDENCE_V3_FIELD_ORDER)
                or plan.get("order_observation_policy") != "report_raw_member_order_separately_from_contract_validity"):
            raise RunConflict("changed serialized schema or response-order observation plan")
    if (plan["study_stage"] != "post_thesis"
            or plan["run_id"] != "qwen3-evidence-synthetic-diagnostic-" + schema_version
            or plan["scope"] != "synthetic_development_only"
            or plan["case_version"] != EVIDENCE_CASE_VERSION or plan["cases_sha256"] != cases_sha256()
            or plan["prompt_version"] != EVIDENCE_PROMPT.version or plan["prompt_sha256"] != EVIDENCE_PROMPT.sha256
            or plan["contract_version"] != contract
            or plan["schema_sha256"] != content_hash(json.loads(schema))
            or plan["profile_name"] != "evidence-v1" or plan["profile_sha256"] != content_hash(backend.profile)
            or plan["examples"] != len(cases) or len(cases) != 14
            or plan["max_attempts_per_input"] != 1 or plan["concurrency"] != 1
            or plan["client_budget_seconds"] != 300 or plan["generation_invocations"] != 1
            or plan["max_input_tokens_per_example"] != 4096
            or plan["max_input_tokens_total"] != 57344
            or plan["max_output_tokens_per_example"] != 512
            or backend.config.max_output_tokens != 512 or plan["max_output_tokens_total"] != 7168
            or plan["request_timeout_seconds"] != backend.timeout_seconds or backend.timeout_seconds != 60
            or plan["max_tokenization_requests"] != 28
            or plan["truncation"] != "none" or plan["halt_on_error_codes"] != list(EVIDENCE_HALT_CODES if schema_version == "v1" else EVIDENCE_V2_HALT_CODES)
            or plan["repeat_policy"] != "cache_inspection_only_after_first_invocation"):
        raise RunConflict("changed evidence diagnostic plan, prompt, cases or profile")


def _validate_state(state, requests):
    try:
        status, audit = state["status"], state["input_audit"]
        if status not in ("ready", "started", "finished") or audit["status"] not in ("pending", "started", "completed", "failed"):
            raise ValueError
        counts = audit["counts"]
        if not isinstance(counts, dict) or not set(counts) <= set(requests):
            raise ValueError
        for row in counts.values():
            if row["status"] not in ("started", "ok", "failed"):
                raise ValueError
            if row["status"] == "ok" and (type(row["input_tokens"]) is not int or not 1 <= row["input_tokens"] <= 4096):
                raise ValueError
        if audit["status"] == "completed" and (set(counts) != set(requests) or any(r["status"] != "ok" for r in counts.values())):
            raise ValueError
        if status == "ready":
            if state["resources"] is not None or audit["status"] != "pending" or counts:
                raise ValueError
        elif type(state["reserved_seconds"]) is not int or state["reserved_seconds"] != 300:
            raise ValueError
        if status == "finished":
            resource_totals(state["resources"]["wall_seconds"], 1)
            if state["outcome"] == "invocation_completed" and audit["status"] != "completed":
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RunConflict("invalid evidence execution or input audit state") from None


async def execute(*, backend, revision, server_record=None, artifact_root=None,
                  inspection_only=False, clock=time.monotonic, schema_version="v1"):
    # Preserve the historical default API and manifest identity.
    plan, cases = (load_plan() if schema_version == "v1" else load_plan(schema_version)), examples()
    validate_plan(plan, backend, cases, schema_version)
    request_factory = {"v1": evidence_request, "v2": evidence_request_v2, "v3": evidence_request_v3}[schema_version]
    requests = {request_factory(ex.item, backend.config).key: request_factory(ex.item, backend.config) for ex in cases}
    directory = run_directory(plan["run_id"], artifact_root)
    control = directory / "execution"
    identity = {"plan": plan, "code_revision": revision, "config": asdict(backend.config)}
    with exclusive_run(control):
        path, journal = control / "budget.json", directory / "journal.sqlite3"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = saved["execution"]
            if saved["execution_sha256"] != content_hash(state) or state["identity"] != identity:
                raise RunConflict("changed or corrupt evidence execution identity")
            _validate_state(state, requests)
            if state["status"] != "ready" and not journal.exists():
                raise RunConflict("missing evidence journal; preserve the run")
        else:
            if journal.exists():
                raise RunConflict("missing evidence budget; preserve the run")
            state = {"identity": identity, "status": "ready", "resources": None,
                     "input_audit": {"status": "pending", "counts": {}}}
            _save(path, state)
        guarded = PrecountedBackend(backend, state["input_audit"]["counts"], tuple(plan["halt_on_error_codes"]))

        async def run(cap):
            return await run_evidence(cases, run_id=plan["run_id"], config=backend.config,
                                     backend=guarded, revision=revision, dataset_revision=EVIDENCE_CASE_VERSION,
                                     artifact_root=artifact_root, max_new_attempts=cap, halt_codes=guarded.halt_codes,
                                     request_factory=request_factory,
                                     expected_response_order=plan.get("expected_response_field_order"))

        report = await run(0)
        before = report["attempts_total"]
        if state["status"] == "ready" and before:
            raise RunConflict("attempts exist without a reserved evidence invocation")
        if before and state["input_audit"]["status"] != "completed":
            raise RunConflict("generation attempts exist without a complete input audit")
        caught = None
        if state["status"] == "ready" and not inspection_only:
            validate_server(server_record or {}, backend, revision)
            if schema_version in ("v2", "v3"):
                validate_v2_server(server_record or {}, schema_version)
            start = clock()
            state.update(status="started", reserved_seconds=300,
                         started_at=datetime.now(timezone.utc).isoformat(), server_session_snapshot=server_record)
            _save(path, state)

            async def work():
                state["server_preflight"] = await backend.preflight()
                state["input_audit"]["status"] = "started"
                _save(path, state)
                # Complete and persist the full audit BEFORE the first generation.
                for key, request in requests.items():
                    row = {"status": "started", "input_tokens": None}
                    guarded.counts[key] = row
                    _save(path, state)
                    try:
                        count = await backend.count_input_tokens(request)
                        if type(count) is not int or count < 1:
                            raise BackendError("invalid_token_count")
                        row["input_tokens"] = count
                        if count > 4096 or count + request.config.max_output_tokens > backend.profile["max_model_len"]:
                            raise BackendError("diagnostic_input_limit")
                    except BackendError as error:
                        row.update(status="failed", error_code=error.code)
                        state["input_audit"]["status"] = "failed"
                        _save(path, state)
                        raise
                    row["status"] = "ok"
                    _save(path, state)
                state["input_audit"]["status"] = "completed"
                _save(path, state)
                return await run(14)

            outcome, error_code = "invocation_completed", None
            try:
                report = await asyncio.wait_for(work(), timeout=300)
                if guarded.halt_code:
                    outcome, error_code = ("halted_on_serving_error" if guarded.halt_code.startswith("http_")
                                           else "halted_on_alignment_error"), guarded.halt_code
            except asyncio.TimeoutError:
                outcome, error_code = "deadline_reached", "evidence_diagnostic_deadline"
            except BackendError as error:
                outcome, error_code = "preflight_or_audit_failed", error.code
            except BaseException as error:
                outcome, error_code = "interrupted_or_error", type(error).__name__
                caught = error
            finally:
                try:
                    report = await run(0)
                finally:
                    state.update(status="finished", outcome=outcome, error_code=error_code,
                                 finished_at=datetime.now(timezone.utc).isoformat(),
                                 resources=resource_totals(clock() - start, 1))
                    _save(path, state)
        charged = (state["resources"]["wall_seconds"] if state["status"] == "finished"
                   else 300 if state["status"] == "started" else 0)
        summary = {"study_stage": "post_thesis", "kind": "synthetic_evidence_diagnostic",
                   "code_revision": revision, "execution_plan_sha256": content_hash(plan),
                   "execution_status": state["status"], "outcome": state.get("outcome"),
                   "error_code": state.get("error_code"), "charged_client_seconds": charged,
                   "generation_invocation_available": state["status"] == "ready",
                   "new_attempts_this_invocation": report["attempts_total"] - before,
                   "input_audit": state["input_audit"], "report": report, "comparisons": comparisons(report),
                   "semantic_evidence_review": "pending", "rental_cost": None,
                   "server_startup_and_idle_excluded": True}
        atomic_json(directory / "evidence_summary.json", summary)
        if caught is not None:
            raise caught
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--server-session-id")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    revision = code_revision()
    record = None
    if args.server_session_id:
        if not args.server_session_id.startswith("server-"):
            parser.error("expected the launcher's server-... directory name")
        record = json.loads((run_directory(args.server_session_id) / "resources.json").read_text(encoding="utf-8"))

    async def start():
        async with VLLMBackend(profile_name="evidence-v1", api_key=os.environ.get("JUDGE_API_KEY")) as backend:
            return await execute(backend=backend, revision=revision, server_record=record, inspection_only=args.inspect_only,
                                 schema_version=args.schema_version)

    result = asyncio.run(start())
    report = result["report"]
    print("Schema version:", args.schema_version)
    print("Prompt:", EVIDENCE_PROMPT.version)
    print(f"Valid evidence records: {report['examples_valid']}/14 synthetic examples")
    print("New attempts:", result["new_attempts_this_invocation"])
    print("Terminal failures:", report["examples_terminal_failure"], "Pending:", report["examples_pending"])
    print("Execution:", result["execution_status"], result["outcome"], result["error_code"])
    print("Input audit:", result["input_audit"]["status"])
    print("Client seconds charged:", result["charged_client_seconds"])
    print("Known token totals:", report["known_token_totals"])
    print("Attempts with unknown token usage:", report["attempts_with_unknown_tokens"])
    print("sample_id,status,verdict,issue_type,verdict_match,issue_match")
    for row in result["comparisons"]:
        print(','.join(str(row[k]) for k in ('sample_id', 'status', 'observed_verdict', 'observed_issue_type',
                                             'verdict_matches_expected', 'issue_matches_expected')))
    if args.schema_version == "v3":
        print("Raw response order:", json.dumps(report["response_order_summary"], sort_keys=True))
        for row in report["predictions"]:
            print("Raw field order:", row["sample_id"], json.dumps(row["response_field_order"]))
    print("Private evidence summary:", run_directory(load_plan(args.schema_version)["run_id"]) / "evidence_summary.json")
    print("Quote membership is not semantic correctness. Manual evidence review pending.")
    print("Post-thesis synthetic development only; no benchmark data or threshold fitting.")
    return 0 if (report["examples_valid"] == 14 and
                 (args.schema_version != "v3" or report["response_order_summary"]["matches"] == 14)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
