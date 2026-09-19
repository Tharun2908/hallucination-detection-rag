"""One bounded synthetic v2 diagnostic invocation; repeats only inspect cache."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .diagnostic_cases import CASE_VERSION, case_records, cases_sha256, comparisons, examples
from .judge import BackendError
from .prompts import DEVELOPMENT_PROMPT_V2, content_hash
from .run_pilot import HALT_CODES, validate_server
from .runner import RetryPolicy, run_directory, run_judge
from .serve import code_revision, resource_totals
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import VLLMBackend

PLAN_PATH = Path(__file__).with_name("configs") / "synthetic_diagnostic_v1.json"
DIAGNOSTIC_HALT_CODES = HALT_CODES + ("diagnostic_input_limit", "invalid_token_count")


def load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


class LimitedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.halt_code = None
        self.counts = {}

    async def complete(self, request):
        async def work():
            count = await self.backend.count_input_tokens(request)
            self.counts[request.key] = count
            if count > 4096:
                raise BackendError("diagnostic_input_limit")
            return await self.backend.complete_counted(request, expected_input_tokens=count)
        try:
            return await asyncio.wait_for(work(), timeout=60)
        except asyncio.TimeoutError:
            raise BackendError("timeout") from None
        except BackendError as error:
            if error.code in DIAGNOSTIC_HALT_CODES:
                self.halt_code = error.code
            raise


def _save(path, state):
    atomic_json(path, {"execution_sha256": content_hash(state), "execution": state})


async def execute(*, backend, revision, server_record=None, artifact_root=None,
                  inspection_only=False, clock=time.monotonic):
    plan = load_plan()
    cases = examples()
    if (plan["run_id"] != "qwen3-v2-synthetic-diagnostic-v1"
            or plan["study_stage"] != "post_thesis" or plan["case_version"] != CASE_VERSION
            or plan["cases_sha256"] != cases_sha256()
            or plan["prompt_version"] != DEVELOPMENT_PROMPT_V2.version
            or plan["prompt_sha256"] != DEVELOPMENT_PROMPT_V2.sha256
            or plan["profile_sha256"] != content_hash(backend.profile)
            or plan["examples"] != len(cases) or len(cases) != 10
            or plan["max_attempts_per_input"] != 1 or plan["concurrency"] != 1
            or plan["client_budget_seconds"] != 300 or plan["generation_invocations"] != 1
            or plan["max_input_tokens_per_example"] != 4096
            or plan["max_output_tokens_per_example"] != backend.config.max_output_tokens
            or backend.config.max_output_tokens != 128
            or plan["request_timeout_seconds"] != backend.timeout_seconds
            or backend.timeout_seconds != 60
            or plan["repeat_policy"] != "cache_inspection_only_after_first_invocation"):
        raise RunConflict("changed synthetic diagnostic plan, prompt, cases or profile")
    directory = run_directory(plan["run_id"], artifact_root)
    control = directory / "execution"
    identity = {"plan": plan, "code_revision": revision, "config": asdict(backend.config)}
    guarded = LimitedBackend(backend)

    async def run(cap):
        return await run_judge(cases, run_id=plan["run_id"], config=backend.config,
                               backend=guarded, code_revision=revision, dataset_revision=CASE_VERSION,
                               prompt=DEVELOPMENT_PROMPT_V2, policy=RetryPolicy(max_attempts=1),
                               artifact_root=artifact_root, max_new_attempts=cap,
                               halt_on_error_codes=DIAGNOSTIC_HALT_CODES)

    with exclusive_run(control):
        path = control / "budget.json"
        journal = directory / "journal.sqlite3"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = saved["execution"]
            if saved["execution_sha256"] != content_hash(state) or state["identity"] != identity:
                raise RunConflict("changed or corrupt diagnostic execution identity")
            if state["status"] not in ("ready", "started", "finished"):
                raise RunConflict("unknown diagnostic execution status")
            if state["status"] != "ready" and not journal.exists():
                raise RunConflict("missing diagnostic journal; preserve the run")
        else:
            if journal.exists():
                raise RunConflict("missing diagnostic budget; preserve the run")
            state = {"identity": identity, "status": "ready", "resources": None}
            _save(path, state)
        report = await run(0)
        before = report["attempts_total"]
        if state["status"] == "ready" and before:
            raise RunConflict("attempts exist without a reserved diagnostic invocation")
        caught = None
        if state["status"] == "ready" and not inspection_only:
            validate_server(server_record or {}, backend, revision)
            start = clock()
            state.update(status="started", started_at=datetime.now(timezone.utc).isoformat(),
                         reserved_seconds=300, server_session_snapshot=server_record)
            _save(path, state)  # Any interrupted invocation consumes its reservation.

            async def work():
                state["server_preflight"] = await backend.preflight()
                return await run(10)

            outcome, error_code = "invocation_completed", None
            try:
                report = await asyncio.wait_for(work(), timeout=300)
                if guarded.halt_code:
                    outcome, error_code = "halted_on_alignment_error", guarded.halt_code
            except asyncio.TimeoutError:
                outcome, error_code = "deadline_reached", "diagnostic_deadline"
            except BackendError as error:
                outcome, error_code = "preflight_failed", error.code
            except BaseException as error:
                outcome, error_code = "interrupted_or_error", type(error).__name__
                caught = error
            finally:
                try:
                    report = await run(0)
                finally:
                    elapsed = clock() - start
                    state.update(status="finished", outcome=outcome, error_code=error_code,
                                 finished_at=datetime.now(timezone.utc).isoformat(),
                                 resources=resource_totals(elapsed, 1), input_counts=guarded.counts)
                    _save(path, state)
        charged = (state["resources"]["wall_seconds"] if state["status"] == "finished"
                   else 300 if state["status"] == "started" else 0)
        summary = {"study_stage": "post_thesis", "kind": "synthetic_diagnostic",
                   "code_revision": revision, "execution_plan_sha256": content_hash(plan),
                   "execution_status": state["status"], "outcome": state.get("outcome"),
                   "error_code": state.get("error_code"), "charged_client_seconds": charged,
                   "generation_invocation_available": state["status"] == "ready",
                   "new_attempts_this_invocation": report["attempts_total"] - before,
                   "report": report, "comparisons": comparisons(report)}
        atomic_json(directory / "diagnostic_summary.json", summary)
        if caught is not None:
            raise caught
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        async with VLLMBackend(api_key=os.environ.get("JUDGE_API_KEY")) as backend:
            return await execute(backend=backend, revision=revision, server_record=record,
                                 inspection_only=args.inspect_only)

    result = asyncio.run(start())
    report = result["report"]
    print(f"Scored {report['examples_scored']}/10 synthetic examples")
    print("New attempts:", result["new_attempts_this_invocation"])
    print("Terminal failures:", report["examples_terminal_failure"], "Pending:", report["examples_pending"])
    print("Execution:", result["execution_status"], result["outcome"], result["error_code"])
    print("Client seconds charged:", result["charged_client_seconds"])
    print("Known token totals:", report["known_token_totals"])
    print("Attempts with unknown token usage:", report["attempts_with_unknown_tokens"])
    expected = {r["sample_id"]: int(r["expected_unsupported"]) for r in case_records()}
    print("sample_id,expected_unsupported,unsupported_probability")
    for prediction in report["predictions"]:
        sid = prediction["sample_id"]
        print(f"{sid},{expected[sid]},{prediction['unsupported_probability']}")
    print("Comparisons:", json.dumps(result["comparisons"], indent=2))
    print("Private records:", run_directory(load_plan()["run_id"]))
    print("Post-thesis synthetic development only; no benchmark data or threshold fitting.")
    return 0 if result["report"]["examples_scored"] == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
