"""Frozen pilot provenance, budget and resume checks with synthetic/mock data."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx
except ImportError:
    httpx = None

from post_thesis.llm_judge.audit_pilot import AUDIT_VERSION
from post_thesis.llm_judge.judge import build_request
from post_thesis.llm_judge.prepare_pilot import build_bundle, pilot_examples
from post_thesis.llm_judge.prompts import content_hash, get_prompt
from post_thesis.llm_judge import run_pilot as pilot
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

P = load_profile()


def records():
    return [{"id": str(i), "context": f"Source {i}: complete evidence.", "output": f"Answer {i}.",
             "model": "NEVER_SEND_GENERATOR", "task_type": "QA", "quality": "good",
             "query": "NEVER_SEND_QUERY", "hallucination_labels_processed":
             {"evident_conflict": i % 2, "baseless_info": 0}} for i in range(52)]


class BudgetTests(unittest.TestCase):
    def test_unknown_window_charges_its_full_reservation(self):
        windows = [{"id": "a", "status": "finished", "reserved_seconds": 600, "elapsed_seconds": 12},
                   {"id": "b", "status": "started", "reserved_seconds": 588, "elapsed_seconds": None}]
        self.assertEqual(pilot.charged_seconds(windows, 600), 600)

    def test_bad_budget_history_is_not_accepted_as_free_compute(self):
        for value in (-1, True, float("nan"), None):
            with self.subTest(value=value), self.assertRaises(RunConflict):
                pilot.charged_seconds([{"id": "a", "status": "finished", "reserved_seconds": 600,
                                        "elapsed_seconds": value}], 600)
        with self.assertRaises(RunConflict):
            pilot.charged_seconds([{"id": "a", "status": "started", "reserved_seconds": 10}], 600)

    def test_committed_plan_records_operator_supplied_checksums_and_budget(self):
        plan = pilot.load_plan()
        self.assertEqual(plan["audit_sha256"], "08dee554dbedde9c53354862af9d2f718c1f52895b33b32379fe4a9971a2b9f0")
        self.assertEqual(plan["pilot_manifest_sha256"], "ef2690b5703ddd67222e4aff2a269f8bab2d47e52a8d3f7f8b8bd608fff69a25")
        self.assertEqual(plan["client_budget_seconds_across_resumes"], 600)
        self.assertEqual(plan["audited_input_tokens"] + plan["max_output_tokens_total"], 55324)


@unittest.skipIf(httpx is None, "HTTP execution tests run after installing the client")
class PilotExecutionTests(unittest.IsolatedAsyncioTestCase):
    pilot_version = "v1"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.count = 100
        self.output_tokens = 8
        self.completion_status = 200
        self.slow = False
        fixture_rows = patch.object(pilot, "TRAIN_ROWS", 52)
        fixture_rows.start()
        self.addCleanup(fixture_rows.stop)

    def backend(self):
        async def handler(request):
            self.calls.append(request)
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": P["vllm_version"]})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": P["served_model_name"], "max_model_len": P["max_model_len"]}]})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": self.count, "max_model_len": P["max_model_len"]})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            if self.slow:
                await asyncio.sleep(10)
            return httpx.Response(self.completion_status, json={"id": "mock-response", "model": P["served_model_name"],
                "usage": {"prompt_tokens": self.count, "completion_tokens": self.output_tokens},
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                    "content": '{"unsupported_probability": 0.25}'}}]})
        return VLLMBackend(transport=httpx.MockTransport(handler))

    def setup_inputs(self, backend):
        self.plan = pilot.load_plan(self.pilot_version)
        self.prompt = get_prompt(self.plan["prompt_version"])
        self.bundle = build_bundle(records(), revision=self.plan["preparation_code_revision"])
        self.plan["pilot_manifest_sha256"] = self.bundle["manifest_sha256"]
        examples = pilot_examples(self.bundle)
        counts = {ex.sample_id: {"status": "ok", "input_tokens": 100,
                  "request_key": build_request(ex.item, config=backend.config, prompt=self.prompt).key} for ex in examples}
        state = {"identity": {"study_stage": "post_thesis", "audit_version": AUDIT_VERSION,
                 "pilot_manifest_sha256": self.bundle["manifest_sha256"],
                 "code_revision": self.plan["audit_code_revision"], "config": asdict(backend.config),
                 "profile": backend.profile, "prompt": {"version": self.prompt.version, "sha256": self.prompt.sha256},
                 "sample_ids": [ex.sample_id for ex in examples]}, "counts": counts}
        self.audited = {"audit": state, "audit_sha256": content_hash(state)}
        self.plan.update(audit_sha256=self.audited["audit_sha256"], audited_input_tokens=5000,
                         audited_min_input_tokens=100, audited_max_input_tokens=100)
        self.server = {"study_stage": "post_thesis", "kind": "serving_session", "status": "started",
                       "profile": backend.profile, "code_revision": "scoring-code", "gpu": {"name": "NVIDIA H200"}}
        self.run_dir = run_directory(self.plan["run_id"], self.root)

    async def execute(self, backend, **kwargs):
        return await pilot.execute(self.bundle, self.audited, plan=self.plan, backend=backend,
                                   revision="scoring-code", artifact_root=self.root,
                                   server_record=self.server, **kwargs)

    def ledger(self):
        path = self.run_dir / "execution" / "budget.json"
        return path, json.loads(path.read_text())["ledger"]

    async def test_fifty_bounded_requests_and_cache_replay_without_network(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            result = await self.execute(backend)
            self.assertEqual(result["report"]["examples_scored"], 50)
            self.assertEqual(result["new_attempts_this_invocation"], 50)
            generations = [r for r in self.calls if r.url.path == "/v1/chat/completions"]
            self.assertEqual(len(generations), 50)
            self.assertEqual(sum(r.url.path == "/tokenize" for r in self.calls), 50)
            for wire in generations:
                payload = json.loads(wire.content)
                self.assertEqual(set(json.loads(payload["messages"][1]["content"])), {"answer", "context"})
                self.assertEqual(payload["max_completion_tokens"], 128)
                self.assertEqual(payload["messages"][0]["content"], self.prompt.system_text)
                self.assertNotIn("NEVER_SEND", wire.content.decode())
            before = len(self.calls)
            self.server["status"] = "stopped"
            cached = await self.execute(backend)
            self.assertEqual(cached["new_attempts_this_invocation"], 0)
            self.assertEqual(before, len(self.calls))
            self.assertEqual(result["charged_client_seconds"], cached["charged_client_seconds"])

    async def test_chunked_resume_accumulates_time_and_skips_successes(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            first = await self.execute(backend, max_new_attempts=1, clock=iter([0, 10]).__next__)
            second = await self.execute(backend, max_new_attempts=1, clock=iter([20, 40]).__next__)
            self.assertEqual(first["charged_client_seconds"], 10)
            self.assertEqual(second["charged_client_seconds"], 30)
            self.assertEqual(second["report"]["examples_scored"], 2)
            self.assertEqual(second["report"]["attempts_total"], 2)
            self.assertEqual(second["remaining_client_seconds"], 570)

    async def test_no_generation_when_audited_count_changes(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.count = 101
            result = await self.execute(backend)
            self.assertEqual(result["error_code"], "audited_input_token_mismatch")
            self.assertEqual(result["status"], "halted_on_alignment_error")
            self.assertNotIn("/v1/chat/completions", [r.url.path for r in self.calls])
            summary = json.loads((self.run_dir / "pilot_summary.json").read_text())
            self.assertEqual(summary["report"]["examples_terminal_failure"], 1)
            self.assertEqual(summary["report"]["examples_pending"], 49)
            before = len(self.calls)
            resumed = await self.execute(backend)
            self.assertEqual(resumed["status"], "blocked_by_alignment_error")
            self.assertEqual(before, len(self.calls))

    async def test_provenance_and_audit_corruption_fail_before_network(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.audited["audit"]["counts"]["0"] = {"status": "ok", "input_tokens": 123}
            with self.assertRaisesRegex(RunConflict, "checksum"):
                await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.run_dir.exists())

    async def test_wrong_server_revision_fails_before_network(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.server["code_revision"] = "other"
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])

    async def test_unknown_previous_window_blocks_new_calls(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            await self.execute(backend, max_new_attempts=0)
            path, ledger = self.ledger()
            ledger["windows"].append({"id": "unclean-process", "status": "started", "reserved_seconds": 600})
            pilot._save_ledger(path, ledger)
            result = await self.execute(backend)
            self.assertEqual(result["status"], "budget_exhausted_or_unknown")
            self.assertEqual(result["remaining_client_seconds"], 0)
            self.assertEqual(result["unknown_window_count"], 1)
            self.assertEqual(self.calls, [])

    async def test_deadline_preserves_interrupted_attempt_and_budget(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            await self.execute(backend, max_new_attempts=0)
            path, ledger = self.ledger()
            ledger["windows"].append({"id": "previous", "status": "finished", "reserved_seconds": 600,
                                      "elapsed_seconds": 599.0})
            pilot._save_ledger(path, ledger)
            self.slow = True
            result = await self.execute(backend)
            self.assertEqual(result["status"], "deadline_reached")
            self.assertEqual(result["report"]["examples_terminal_failure"], 1)
            self.assertEqual(result["report"]["examples_pending"], 49)
            before = len(self.calls)
            resumed = await self.execute(backend)
            self.assertEqual(resumed["new_attempts_this_invocation"], 0)
            self.assertEqual(before, len(self.calls))

    async def test_failed_attempt_is_not_retried(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.completion_status = 503
            first = await self.execute(backend, max_new_attempts=1)
            self.assertEqual(first["report"]["examples_terminal_failure"], 1)
            self.completion_status = 200
            second = await self.execute(backend, max_new_attempts=1)
            self.assertEqual(second["report"]["attempts_total"], 2)
            self.assertEqual(second["report"]["examples_terminal_failure"], 1)
            self.assertEqual(second["report"]["examples_scored"], 1)

    async def test_missing_ledger_or_journal_does_not_reset_run(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            await self.execute(backend, max_new_attempts=1)
            before = len(self.calls)
            path, _ = self.ledger()
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            path.write_bytes(original)
            (self.run_dir / "journal.sqlite3").unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before, len(self.calls))

    async def test_response_output_limit_violation_halts(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.output_tokens = 129
            result = await self.execute(backend)
            self.assertEqual(result["error_code"], "output_token_limit_exceeded")
            self.assertEqual(result["report"]["known_token_totals"]["output_tokens"], 129)
            self.assertEqual(sum(r.url.path == "/v1/chat/completions" for r in self.calls), 1)


class V2PilotExecutionTests(PilotExecutionTests):
    # Exercise every budget, failure, resume and exact wire-input guard for v2 too.
    pilot_version = "v2"

    async def test_v1_and_v2_journals_and_budgets_stay_independent(self):
        async with self.backend() as backend:
            self.pilot_version = "v1"
            self.setup_inputs(backend)
            await self.execute(backend, max_new_attempts=1, clock=iter([0, 10]).__next__)
            v1_dir = self.run_dir
            before = {p.relative_to(v1_dir): p.read_bytes() for p in v1_dir.rglob("*") if p.is_file()}
            self.pilot_version = "v2"
            self.setup_inputs(backend)
            result = await self.execute(backend, max_new_attempts=1, clock=iter([20, 25]).__next__)
            self.assertNotEqual(v1_dir, self.run_dir)
            self.assertEqual(result["charged_client_seconds"], 5)
            self.assertEqual(result["remaining_client_seconds"], 595)
            self.assertEqual(result["new_attempts_this_invocation"], 1)
            self.assertEqual(before, {p.relative_to(v1_dir): p.read_bytes() for p in v1_dir.rglob("*") if p.is_file()})

    async def test_v1_audit_is_rejected_by_v2_before_writes_or_network(self):
        async with self.backend() as backend:
            self.pilot_version = "v1"
            self.setup_inputs(backend)
            old_audit = deepcopy(self.audited)
            self.pilot_version = "v2"
            self.setup_inputs(backend)
            self.audited = old_audit
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.run_dir.exists())

    async def test_v2_cannot_target_v1_directory_or_change_preparation_prompt(self):
        async with self.backend() as backend:
            self.setup_inputs(backend)
            self.plan["run_id"] = "qwen3-ragtruth-train-pilot-50-v1"
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.run_dir.exists())
            self.setup_inputs(backend)
            self.bundle["manifest"]["initial_prompt"] = {"version": self.prompt.version, "sha256": self.prompt.sha256}
            self.bundle["manifest_sha256"] = content_hash(self.bundle["manifest"])
            self.plan["pilot_manifest_sha256"] = self.bundle["manifest_sha256"]
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.run_dir.exists())


class V2PlanTests(unittest.TestCase):
    def test_only_recorded_versions_and_matching_prompt_identity_are_accepted(self):
        self.assertEqual(pilot.load_plan(), pilot.load_plan("v1"))
        with self.assertRaises(ValueError):
            pilot.load_plan("v99")
        plan = pilot.load_plan("v2")
        self.assertEqual(plan["audit_sha256"], "fb28d1bab2bc2cd64d650af26dce5fad8c7374b5997acf8d6afe720d59bbeaae")
        self.assertEqual(plan["audit_code_revision"], "f0645464dbe1803921b8a8545fad980eaae4e424")
        self.assertEqual(plan["audited_input_tokens"] + plan["max_output_tokens_total"], 69074)
        self.assertEqual(pilot.plan_prompt(plan), get_prompt("faithfulness-development-v2"))
        plan["prompt_sha256"] = "wrong"
        with self.assertRaises(RunConflict):
            pilot.plan_prompt(plan)


if __name__ == "__main__":
    unittest.main()
