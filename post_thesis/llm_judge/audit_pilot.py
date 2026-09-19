"""Post-thesis pilot token audit. Calls /tokenize, never generation endpoints."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

from .binary_contract import BINARY_PROMPT, BINARY_SCHEMA_JSON, binary_request
from .evidence_contract import EVIDENCE_PROMPT
from .evidence_schema_v3 import (EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_WIRE_SHA256,
                                 EVIDENCE_V3_CONTRACT_VERSION, EVIDENCE_V3_FIELD_ORDER, evidence_request_v3)
from .judge import BackendError, build_request
from .prepare_pilot import pilot_examples
from .prompts import DEVELOPMENT_PROMPT, DEVELOPMENT_PROMPT_V2, content_hash, get_prompt
from .runner import run_directory
from .serve import code_revision, resource_totals
from .storage import RunConflict, atomic_json, exclusive_run
from .vllm_backend import VLLMBackend

AUDIT_VERSION = "pilot_formatted_lengths_v1"
DEADLINE_SECONDS = 300
EVIDENCE_PILOT_MANIFEST_SHA256 = "ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25"
EVIDENCE_PROFILE_SHA256 = "4d7a4a3b0d534240e2d87c287faa97804b44320e40c2097525615975bdf2aa28"
AUDIT_DIRECTORIES = {
    DEVELOPMENT_PROMPT.version: "ragtruth-train-pilot-50-token-audit-v1",
    DEVELOPMENT_PROMPT_V2.version: "ragtruth-train-pilot-50-token-audit-v2",
    BINARY_PROMPT.version: "ragtruth-train-pilot-50-token-audit-binary-v1",
    EVIDENCE_PROMPT.version: "ragtruth-train-pilot-50-token-audit-evidence-v3",
}


def audit_prompt(version):
    # Diagnostic prompts stay separate from the probability registry/defaults.
    if version == BINARY_PROMPT.version:
        return BINARY_PROMPT
    if version == EVIDENCE_PROMPT.version:
        return EVIDENCE_PROMPT
    return get_prompt(version)


def _save(path, state):
    atomic_json(path, {"audit_sha256": content_hash(state), "audit": state})


def _load(path, identity, requests):
    saved = json.loads(path.read_text(encoding="utf-8"))
    state = saved["audit"]
    if saved["audit_sha256"] != content_hash(state) or state["identity"] != identity:
        raise RunConflict("changed or corrupt token audit; preserve the old run")
    for sample_id, row in state["counts"].items():
        if sample_id not in requests or row["request_key"] != requests[sample_id].key:
            raise RunConflict("token audit request alignment mismatch")
        if row["status"] == "ok":
            if type(row["input_tokens"]) is not int or row["input_tokens"] < 1:
                raise RunConflict("invalid cached token count")
        elif row["status"] not in {"started", "error", "interrupted"}:
            raise RunConflict("unexpected token audit status")
    return state


def summary(state):
    counts = state["counts"]
    good = {key: value["input_tokens"] for key, value in counts.items() if value["status"] == "ok"}
    config = state["identity"]["config"]
    limit = state["identity"]["profile"]["max_model_len"]
    allowance = config["max_output_tokens"]
    over = [key for key, count in good.items() if count + allowance > limit]
    expected = len(state["identity"]["sample_ids"])
    return {"examples_total": expected, "examples_counted": len(good),
            "examples_failed_or_interrupted": len(counts) - len(good),
            "examples_pending": expected - len(counts), "overlength_ids": over,
            "max_input_tokens": max(good.values(), default=None),
            "min_input_tokens": min(good.values(), default=None),
            "total_input_tokens": sum(good.values()),
            "output_allowance_per_example": allowance, "model_limit": limit,
            "all_inputs_fit": len(good) == expected and not over,
            "generation_calls": 0, "scoring_authorized_by_this_audit": False}


async def audit(bundle, *, expected_manifest_sha256, backend, directory, revision,
                prompt=DEVELOPMENT_PROMPT):
    # No network calls or output writes until the immutable input contract passes.
    if bundle.get("manifest_sha256") != expected_manifest_sha256:
        raise RunConflict("this is not the expected pilot manifest")
    examples = pilot_examples(bundle)
    manifest = bundle["manifest"]
    if len(examples) != 50 or manifest["initial_prompt"] != {
            "version": DEVELOPMENT_PROMPT.version, "sha256": DEVELOPMENT_PROMPT.sha256}:
        raise RunConflict("expected the unchanged 50-example pilot prepared with v1")
    if prompt != audit_prompt(prompt.version):
        raise RunConflict("prompt text does not match its recorded version")
    if prompt == EVIDENCE_PROMPT and (
            expected_manifest_sha256 != EVIDENCE_PILOT_MANIFEST_SHA256
            or content_hash(backend.profile) != EVIDENCE_PROFILE_SHA256
            or backend.config.max_output_tokens != 512 or backend.timeout_seconds != 60):
        raise RunConflict("evidence audit requires the original pilot and pinned evidence profile")
    requests = {
        example.sample_id: (evidence_request_v3(example.item, backend.config) if prompt == EVIDENCE_PROMPT
                            else binary_request(example.item, backend.config) if prompt == BINARY_PROMPT
                            else build_request(example.item, config=backend.config, prompt=prompt))
        for example in examples
    }
    identity = {"study_stage": "post_thesis", "audit_version": AUDIT_VERSION,
                "pilot_manifest_sha256": expected_manifest_sha256,
                "code_revision": revision, "config": asdict(backend.config),
                "profile": backend.profile, "prompt": {"version": prompt.version, "sha256": prompt.sha256},
                "sample_ids": list(requests), "deadline_seconds": DEADLINE_SECONDS,
                "max_attempts_per_input": 1, "concurrency": 1}
    if prompt == BINARY_PROMPT:
        identity.update(request_contract="binary-diagnostic-request-v1",
                        response_schema_sha256=content_hash(json.loads(BINARY_SCHEMA_JSON)))
    if prompt == EVIDENCE_PROMPT:
        identity.update(request_contract=EVIDENCE_V3_CONTRACT_VERSION,
                        response_schema_sha256=content_hash(json.loads(EVIDENCE_SCHEMA_V3_JSON)),
                        serialized_schema_sha256=EVIDENCE_V3_WIRE_SHA256,
                        response_schema_json=EVIDENCE_SCHEMA_V3_JSON,
                        expected_response_field_order=list(EVIDENCE_V3_FIELD_ORDER))
    directory = Path(directory)
    with exclusive_run(directory):
        path = directory / "audit.json"
        if prompt == EVIDENCE_PROMPT and not path.exists() and any(directory.glob("client-window-*.json")):
            raise RunConflict("missing evidence audit checkpoint; preserve previous attempts")
        state = (_load(path, identity, requests) if path.exists() else
                 {"identity": identity, "counts": {}})
        for row in state["counts"].values():
            if row["status"] == "started":
                row.update(status="interrupted", error_code="previous_process_interrupted")
        _save(path, state)
        window_path = directory / ("client-window-" + uuid.uuid4().hex + ".json")
        window = {"study_stage": "post_thesis", "kind": "pilot_token_audit",
                  "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
                  "pilot_manifest_sha256": expected_manifest_sha256,
                  "prompt": {"version": prompt.version, "sha256": prompt.sha256},
                  "code_revision": revision, "tokenize_requests_started": 0,
                  "generation_calls": 0, "resources": None,
                  "scope": "client_window_excludes_server_startup_and_other_idle_time",
                  "gpu_count_basis": "declared_profile_assumes_exclusive_H200_server"}
        atomic_json(window_path, window)
        start = time.monotonic()

        async def work():
            # A failed/interrupted attempt is terminal for this audit ID.
            if any(row["status"] != "ok" for row in state["counts"].values()):
                return
            if len(state["counts"]) == len(requests):
                return  # Pure cache replay; no preflight or tokenization requests.
            window["server_preflight"] = await backend.preflight()
            atomic_json(window_path, window)
            for sample_id, request in requests.items():
                if sample_id in state["counts"]:
                    continue
                row = {"request_key": request.key, "status": "started", "input_tokens": None}
                state["counts"][sample_id] = row
                _save(path, state)  # Reserve attempt before issuing HTTP request.
                window["tokenize_requests_started"] += 1
                atomic_json(window_path, window)
                try:
                    count = await backend.count_input_tokens(request)
                    if type(count) is not int or count < 1:
                        raise BackendError("invalid_token_count")
                    row.update(status="ok", input_tokens=count)
                except BackendError as error:
                    row.update(status="error", error_code=error.code)
                    return
                except BaseException:
                    row.update(status="interrupted", error_code="interrupted_or_unexpected_error")
                    raise
                finally:
                    _save(path, state)

        try:
            await asyncio.wait_for(work(), timeout=DEADLINE_SECONDS)
            window["status"] = "completed"
        except (BackendError, asyncio.TimeoutError) as error:
            window.update(status="failed", error_code=getattr(error, "code", "audit_deadline"))
        finally:
            if window["status"] == "started":
                window["status"] = "interrupted_or_unexpected_error"
            window["summary"] = summary(state)
            if window["summary"]["examples_failed_or_interrupted"]:
                window["status"] = "blocked_by_terminal_attempt"
            window["finished_at"] = datetime.now(timezone.utc).isoformat()
            window["resources"] = resource_totals(time.monotonic() - start, 1)
            atomic_json(window_path, window)
        return window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--prompt-version", default=DEVELOPMENT_PROMPT.version,
                        choices=list(AUDIT_DIRECTORIES))
    parser.add_argument("--audit-id", help="Defaults to a separate directory for each prompt version")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    prompt = audit_prompt(args.prompt_version)
    defaults = AUDIT_DIRECTORIES
    audit_id = args.audit_id or defaults[prompt.version]
    if audit_id in defaults.values() and audit_id != defaults[prompt.version]:
        parser.error("use the selected prompt's audit directory; preserve other prompt versions")
    if prompt == EVIDENCE_PROMPT and audit_id != defaults[prompt.version]:
        parser.error("use the fixed evidence-v3 audit directory; preserve its attempts")
    revision = code_revision()
    manifest_path = run_directory("ragtruth-train-pilot-50-v1") / "manifest.json"
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = run_directory(audit_id)

    async def execute():
        async with VLLMBackend(args.base_url, api_key=os.environ.get("JUDGE_API_KEY"),
                               profile_name="evidence-v1" if prompt == EVIDENCE_PROMPT else "default") as backend:
            return await audit(bundle, expected_manifest_sha256=args.expected_manifest_sha256,
                               backend=backend, directory=directory, revision=revision, prompt=prompt)

    result = asyncio.run(execute())
    print(json.dumps(result["summary"], indent=2))
    print("New tokenization requests:", result["tokenize_requests_started"])
    print("Audit status:", result["status"])
    print("Prompt version:", prompt.version)
    print("Prompt SHA256:", prompt.sha256)
    if prompt == EVIDENCE_PROMPT:
        print("Evidence schema version: v3")
        print("Serialized schema SHA256:", EVIDENCE_V3_WIRE_SHA256)
    saved = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    print("Audit SHA256:", saved["audit_sha256"])
    print("Audit code revision:", revision)
    print("Private audit records:", directory)
    print("No scores generated. Pilot scoring budget still requires a recorded configuration.")
    return 0 if result["summary"]["all_inputs_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
