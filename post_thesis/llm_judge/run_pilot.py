"""Score only the frozen 50-example post-thesis TRAIN pilot, within its budget."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid

from .audit_pilot import AUDIT_VERSION
from .judge import BackendError, PROTOCOL_ID, build_request
from .prepare_pilot import DATASET, DATASET_REVISION, TRAIN_ROWS, TRAIN_SHA256, pilot_examples
from .prompts import DEVELOPMENT_PROMPT, content_hash
from .runner import RetryPolicy, run_directory, run_judge
from .serve import code_revision, resource_totals
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import VLLMBackend

PLAN_PATH = Path(__file__).with_name("configs") / "ragtruth_pilot_50_v1.json"
HALT_CODES = ("audited_input_token_mismatch", "input_token_mismatch",
              "returned_model_mismatch", "output_token_limit_exceeded")


def load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def validate_inputs(bundle, audited, plan, backend):
    """Fail closed before model calls or spending the execution allowance."""
    if (plan["study_stage"] != "post_thesis" or plan["protocol_id"] != PROTOCOL_ID
            or plan["execution_plan_version"] != "ragtruth_pilot_50_v1"
            or plan["examples"] != 50 or plan["max_attempts_per_input"] != 1
            or plan["concurrency"] != 1 or plan["truncation"] != "none"
            or plan["request_timeout_seconds"] != 60
            or plan["client_budget_seconds_across_resumes"] != 600
            or plan["halt_on_error_codes"] != list(HALT_CODES)
            or plan["unknown_window_policy"] != "charge_full_reserved_remaining_budget"):
        raise RunConflict("unsupported pilot execution plan")
    if bundle["manifest_sha256"] != plan["pilot_manifest_sha256"]:
        raise RunConflict("pilot manifest differs from the frozen execution plan")
    examples = pilot_examples(bundle)
    manifest = bundle["manifest"]
    dataset = manifest["dataset"]
    if (dataset["repository"] != DATASET or dataset["revision"] != DATASET_REVISION
            or dataset["sha256"] != TRAIN_SHA256 or dataset["rows"] != TRAIN_ROWS
            or manifest["code_revision"] != plan["preparation_code_revision"]):
        raise RunConflict("unexpected TRAIN data provenance")
    prompt = {"version": DEVELOPMENT_PROMPT.version, "sha256": DEVELOPMENT_PROMPT.sha256}
    if (manifest["initial_prompt"] != prompt
            or prompt != {"version": plan["prompt_version"], "sha256": plan["prompt_sha256"]}
            or content_hash(backend.profile) != plan["profile_sha256"]
            or backend.timeout_seconds != plan["request_timeout_seconds"]):
        raise RunConflict("changed development prompt, profile or request timeout")
    state = audited["audit"]
    if (audited["audit_sha256"] != plan["audit_sha256"]
            or content_hash(state) != plan["audit_sha256"]):
        raise RunConflict("token audit checksum mismatch")
    identity = state["identity"]
    if (identity["study_stage"] != "post_thesis" or identity["audit_version"] != AUDIT_VERSION
            or identity["pilot_manifest_sha256"] != plan["pilot_manifest_sha256"]
            or identity["code_revision"] != plan["audit_code_revision"]
            or identity["config"] != asdict(backend.config)
            or identity["profile"] != backend.profile or identity["prompt"] != prompt
            or identity["sample_ids"] != [example.sample_id for example in examples]):
        raise RunConflict("token audit identity mismatch")
    if len(examples) != plan["examples"] or set(state["counts"]) != set(identity["sample_ids"]):
        raise RunConflict("token audit must cover exactly the selected 50 examples")
    expected = {}
    for example in examples:
        request = build_request(example.item, config=backend.config)
        row = state["counts"][example.sample_id]
        count = row["input_tokens"]
        if (row["status"] != "ok" or row["request_key"] != request.key
                or type(count) is not int or count < 1
                or count + backend.config.max_output_tokens > backend.profile["max_model_len"]):
            raise RunConflict("incomplete, overlength or misaligned token audit")
        expected[request.key] = count
    counts = list(expected.values())
    if (len(counts) != len(examples) or sum(counts) != plan["audited_input_tokens"]
            or min(counts) != plan["audited_min_input_tokens"]
            or max(counts) != plan["audited_max_input_tokens"]
            or backend.config.max_output_tokens != plan["max_output_tokens_per_example"]
            or len(counts) * backend.config.max_output_tokens != plan["max_output_tokens_total"]):
        raise RunConflict("audited token totals or output allowance changed")
    return examples, expected


class AuditedBackend:
    def __init__(self, backend, expected):
        self.backend, self.expected = backend, expected
        self.halt_code = None

    async def complete(self, request):
        try:
            return await self.backend.complete_counted(
                request, expected_input_tokens=self.expected[request.key])
        except BackendError as error:
            # An alignment discrepancy stops the entire invocation. Ordinary
            # bounded transport/refusal failures are recorded by the core runner.
            if error.code in HALT_CODES:
                self.halt_code = error.code
            raise


def _save_ledger(path, state):
    atomic_json(path, {"ledger_sha256": content_hash(state), "ledger": state})


def charged_seconds(windows, total):
    charged, seen = 0.0, set()
    for row in windows:
        if row["id"] in seen:
            raise RunConflict("duplicate budget window")
        seen.add(row["id"])
        reserved = row["reserved_seconds"]
        if (type(reserved) not in (int, float) or not math.isfinite(reserved)
                or reserved <= 0 or abs(reserved - max(0, total - charged)) > 1e-6):
            raise RunConflict("invalid budget reservation history")
        if row["status"] == "started":
            charged += reserved  # Unknown elapsed time cannot become a fresh budget.
        elif row["status"] == "finished":
            elapsed = row["elapsed_seconds"]
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise RunConflict("invalid measured budget window")
            charged += elapsed
        else:
            raise RunConflict("unexpected budget window status")
    return charged


def validate_server(record, backend, revision):
    if (record.get("study_stage") != "post_thesis" or record.get("kind") != "serving_session"
            or record.get("status") != "started" or record.get("code_revision") != revision
            or record.get("profile") != backend.profile
            or "H200" not in record.get("gpu", {}).get("name", "")):
        raise RunConflict("start the pinned H200 launcher from this committed checkout")


async def execute(bundle, audited, *, plan, backend, revision, artifact_root=None,
                  server_record=None, max_new_attempts=50, clock=time.monotonic):
    examples, counts = validate_inputs(bundle, audited, plan, backend)
    if type(max_new_attempts) is not int or not 0 <= max_new_attempts <= 50:
        raise ValueError("max_new_attempts must be in [0,50]; it cannot increase the total budget")
    directory = run_directory(plan["run_id"], artifact_root)
    control = directory / "execution"
    identity = {"plan": plan, "plan_sha256": content_hash(plan), "code_revision": revision,
                "config": asdict(backend.config)}
    guarded = AuditedBackend(backend, counts)

    async def run(cap):
        return await run_judge(examples, run_id=plan["run_id"], config=backend.config,
                               backend=guarded, code_revision=revision,
                               dataset_revision=DATASET_REVISION, policy=RetryPolicy(max_attempts=1),
                               artifact_root=artifact_root, max_new_attempts=cap,
                               halt_on_error_codes=HALT_CODES)

    with exclusive_run(control):
        ledger_path = control / "budget.json"
        journal = directory / "journal.sqlite3"
        if ledger_path.exists():
            saved = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger = saved["ledger"]
            if saved["ledger_sha256"] != content_hash(ledger) or ledger["identity"] != identity:
                raise RunConflict("changed or corrupt pilot budget identity")
            if ledger["windows"] and not journal.exists():
                raise RunConflict("pilot attempt journal is missing; do not reset a run")
        else:
            if journal.exists():
                raise RunConflict("pilot budget ledger is missing; do not reset a run")
            ledger = {"identity": identity, "windows": []}
            _save_ledger(ledger_path, ledger)
        total = plan["client_budget_seconds_across_resumes"]
        spent = charged_seconds(ledger["windows"], total)
        # Validate/recover the existing journal and refresh its summary WITHOUT calls.
        report = await run(0)
        before_attempts = report["attempts_total"]
        remaining = max(0, total - spent)
        status = "cached_or_terminal"
        error_code = None
        caught = None
        if report["examples_pending"] and max_new_attempts:
            prior_halt = next((w.get("error_code") for w in ledger["windows"]
                               if w.get("error_code") in HALT_CODES), None)
            if prior_halt:
                status, error_code = "blocked_by_alignment_error", prior_halt
            elif remaining <= 0:
                status = "budget_exhausted_or_unknown"
            else:
                validate_server(server_record or {}, backend, revision)
                start = clock()
                window = {"id": uuid.uuid4().hex, "status": "started",
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          "reserved_seconds": remaining, "elapsed_seconds": None,
                          "server_session_snapshot": server_record, "resources": None}
                ledger["windows"].append(window)
                _save_ledger(ledger_path, ledger)  # Reserve all remaining time before HTTP.

                async def work():
                    window["server_preflight"] = await backend.preflight()
                    return await run(max_new_attempts)

                try:
                    report = await asyncio.wait_for(work(), timeout=remaining)
                    status = "halted_on_alignment_error" if guarded.halt_code else "invocation_completed"
                    error_code = guarded.halt_code
                except asyncio.TimeoutError:
                    status, error_code = "deadline_reached", "cumulative_client_deadline"
                except BackendError as error:
                    status, error_code = "preflight_failed", error.code
                except BaseException as error:
                    status, error_code = "interrupted_or_error", type(error).__name__
                    caught = error
                finally:
                    # Interrupted reservations become terminal attempts. No retry,
                    # fabricated score, or extra request during summary recovery.
                    try:
                        report = await run(0)
                    finally:
                        elapsed = clock() - start
                        window.update(status="finished", outcome=status, error_code=error_code,
                                      finished_at=datetime.now(timezone.utc).isoformat(),
                                      elapsed_seconds=elapsed, resources=resource_totals(elapsed, 1))
                        _save_ledger(ledger_path, ledger)
        elif report["examples_pending"]:
            status = "inspection_only"
        spent = charged_seconds(ledger["windows"], total)
        output = {"study_stage": "post_thesis", "kind": "TRAIN_development_pilot",
                  "execution_plan_sha256": content_hash(plan), "status": status,
                  "error_code": error_code, "code_revision": revision,
                  "pilot_manifest_sha256": plan["pilot_manifest_sha256"],
                  "audit_sha256": plan["audit_sha256"],
                  "new_attempts_this_invocation": report["attempts_total"] - before_attempts,
                  "charged_client_seconds": spent, "remaining_client_seconds": max(0, total - spent),
                  "budget_seconds": total, "budget_scope": "preflight_and_runner_windows_across_resumes",
                  "unknown_window_count": sum(w["status"] == "started" for w in ledger["windows"]),
                  "server_startup_and_idle_excluded": True, "rental_cost": None,
                  "report": report}
        atomic_json(directory / "pilot_summary.json", output)
        if caught is not None:
            raise caught
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-session-id", help="Printed server-... directory name from this launcher")
    parser.add_argument("--max-new-attempts", type=int, default=50,
                        help="Optional smaller invocation cap, 0 for cache inspection; never resets budgets")
    args = parser.parse_args()
    revision = code_revision()
    plan = load_plan()
    bundle = json.loads((run_directory("ragtruth-train-pilot-50-v1") / "manifest.json").read_text(encoding="utf-8"))
    audited = json.loads((run_directory("ragtruth-train-pilot-50-token-audit-v1") / "audit.json").read_text(encoding="utf-8"))
    record = None
    if args.server_session_id:
        if not args.server_session_id.startswith("server-"):
            parser.error("expected the launcher's server-... directory name")
        record = json.loads((run_directory(args.server_session_id) / "resources.json").read_text(encoding="utf-8"))

    async def start():
        async with VLLMBackend(api_key=os.environ.get("JUDGE_API_KEY")) as backend:
            return await execute(bundle, audited, plan=plan, backend=backend, revision=revision,
                                 server_record=record, max_new_attempts=args.max_new_attempts)

    output = asyncio.run(start())
    report = output["report"]
    print(f"Scored {report['examples_scored']}/{report['examples_total']} TRAIN development examples")
    print("New attempts:", output["new_attempts_this_invocation"])
    print("Terminal failures:", report["examples_terminal_failure"], "Pending:", report["examples_pending"])
    print("Execution status:", output["status"])
    print("Client seconds charged / remaining:", output["charged_client_seconds"], "/", output["remaining_client_seconds"])
    print("Known token totals:", report["known_token_totals"])
    print("Attempts with unknown token usage:", report["attempts_with_unknown_tokens"])
    print("Private pilot summary:", run_directory(plan["run_id"]) / "pilot_summary.json")
    print("Post-thesis development only. No test metrics, threshold fitting or prompt changes.")
    return 0 if report["examples_scored"] == plan["examples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
