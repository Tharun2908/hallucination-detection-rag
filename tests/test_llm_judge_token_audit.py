"""Token-only pilot checks: mock HTTP and synthetic TRAIN fixtures."""

import asyncio
from copy import deepcopy
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx
except ImportError:
    httpx = None

from post_thesis.llm_judge.audit_pilot import audit, main
from post_thesis.llm_judge.judge import JudgeConfig, JudgeInput, build_request
from post_thesis.llm_judge.prepare_pilot import build_bundle
from post_thesis.llm_judge.prompts import (
    DEVELOPMENT_PROMPT, DEVELOPMENT_PROMPT_V2, PromptSpec, content_hash, get_prompt,
)
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

P = load_profile()


def pilot():
    return build_bundle([
        {"id": str(i), "context": f"Complete evidence for source {i}.",
         "output": f"Statement {i}.", "task_type": "QA", "model": "hidden-model",
         "quality": "good", "hallucination_labels_processed":
         {"evident_conflict": i % 2, "baseless_info": 0}}
        for i in range(52)], revision="fixture-preparation")


class PromptVersionTests(unittest.TestCase):
    def test_v1_hash_default_and_wire_input_remain_frozen(self):
        self.assertEqual(DEVELOPMENT_PROMPT.sha256,
                         "e9e218764a4c7171f85468fd8f38ed698003f7538766ef692163f21cacc483c5")
        item = JudgeInput("A statement.", "Evidence.")
        config = JudgeConfig("offline", "model", "profile")
        default = build_request(item, config)
        v1 = build_request(item, config, get_prompt("faithfulness-development-v1"))
        v2 = build_request(item, config, get_prompt("faithfulness-development-v2"))
        self.assertEqual(default, v1)
        self.assertNotEqual(v1.key, v2.key)
        self.assertEqual(v1.config, v2.config)
        self.assertEqual(v1.response_schema_json, v2.response_schema_json)
        self.assertEqual(v1.messages[1], v2.messages[1])

    def test_unknown_version_has_no_fallback(self):
        with self.assertRaises(ValueError):
            get_prompt("faithfulness-development-v99")

    def test_cli_blocks_other_prompts_reserved_directory_before_loading_artifacts(self):
        with patch("sys.argv", ["audit_pilot", "--expected-manifest-sha256", "fixture",
                               "--prompt-version", "faithfulness-development-v2",
                               "--audit-id", "ragtruth-train-pilot-50-token-audit-v1"]), \
                patch("post_thesis.llm_judge.audit_pilot.code_revision") as revision, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            revision.assert_not_called()


@unittest.skipIf(httpx is None, "HTTP audit tests run after installing the client")
class TokenAuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "audit"
        self.bundle = pilot()
        self.calls = []

    def backend(self, count=100, fail=False):
        def handler(request):
            self.calls.append(request)
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": P["vllm_version"]})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": P["served_model_name"],
                                                          "max_model_len": P["max_model_len"]}]})
            self.assertEqual(request.url.path, "/tokenize", "audit must never generate")
            if fail:
                return httpx.Response(503)
            return httpx.Response(200, json={"count": count, "max_model_len": P["max_model_len"]})
        return VLLMBackend(transport=httpx.MockTransport(handler))

    async def run_audit(self, backend, bundle=None, expected=None, **kwargs):
        return await audit(bundle or self.bundle,
                           expected_manifest_sha256=expected or self.bundle["manifest_sha256"],
                           backend=backend, directory=self.directory, revision="fixture-audit", **kwargs)

    async def test_v2_audit_uses_new_system_prompt_and_identical_prepared_inputs(self):
        original = deepcopy(self.bundle)
        async with self.backend() as backend:
            result = await self.run_audit(backend, prompt=DEVELOPMENT_PROMPT_V2)
            self.assertTrue(result["summary"]["all_inputs_fit"])
            self.assertEqual(self.bundle, original)
            tokens = [r for r in self.calls if r.url.path == "/tokenize"]
            self.assertEqual(len(tokens), 50)
            for wire, source in zip(tokens, original["manifest"]["pilot_inputs"]):
                messages = json.loads(wire.content)["messages"]
                self.assertEqual(messages[0]["content"], DEVELOPMENT_PROMPT_V2.system_text)
                self.assertEqual(json.loads(messages[1]["content"]), source["input"])
            saved = json.loads((self.directory / "audit.json").read_text())["audit"]
            self.assertEqual(saved["identity"]["prompt"]["version"], DEVELOPMENT_PROMPT_V2.version)
            self.assertEqual(saved["identity"]["pilot_manifest_sha256"], original["manifest_sha256"])

    async def test_v2_cannot_reuse_or_overwrite_v1_counts(self):
        async with self.backend() as backend:
            await self.run_audit(backend)
            path = self.directory / "audit.json"
            original = path.read_bytes()
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.run_audit(backend, prompt=DEVELOPMENT_PROMPT_V2)
            self.assertEqual(len(self.calls), before)
            self.assertEqual(path.read_bytes(), original)

    async def test_changed_text_cannot_impersonate_recorded_prompt_version(self):
        async with self.backend() as backend:
            with self.assertRaises(RunConflict):
                await self.run_audit(backend, prompt=PromptSpec(DEVELOPMENT_PROMPT_V2.version, "changed"))
            self.assertEqual(self.calls, [])
            self.assertFalse(self.directory.exists())

    async def test_fifty_exact_inputs_without_labels_or_generation_and_cached_replay(self):
        async with self.backend() as backend:
            result = await self.run_audit(backend)
            self.assertTrue(result["summary"]["all_inputs_fit"])
            self.assertEqual(result["tokenize_requests_started"], 50)
            self.assertEqual(result["summary"]["total_input_tokens"], 5000)
            tokens = [r for r in self.calls if r.url.path == "/tokenize"]
            for wire, original in zip(tokens, self.bundle["manifest"]["pilot_inputs"]):
                payload = json.loads(wire.content)
                self.assertEqual(json.loads(payload["messages"][1]["content"]), original["input"])
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertTrue(payload["add_generation_prompt"])
                self.assertFalse(payload["add_special_tokens"])
            before = len(self.calls)
            repeat = await self.run_audit(backend)
            self.assertEqual(repeat["tokenize_requests_started"], 0)
            self.assertEqual(before, len(self.calls))
            self.assertEqual(len(list(self.directory.glob("client-window-*.json"))), 2)

    async def test_limit_includes_output_allowance(self):
        async with self.backend(count=P["max_model_len"] - P["max_output_tokens"]) as backend:
            result = await self.run_audit(backend)
            self.assertTrue(result["summary"]["all_inputs_fit"])
        self.directory = Path(self.temporary.name) / "over"
        async with self.backend(count=P["max_model_len"] - P["max_output_tokens"] + 1) as backend:
            result = await self.run_audit(backend)
            self.assertFalse(result["summary"]["all_inputs_fit"])
            self.assertEqual(len(result["summary"]["overlength_ids"]), 50)
            self.assertEqual(result["summary"]["generation_calls"], 0)

    async def test_wrong_or_changed_manifest_fails_before_any_request(self):
        async with self.backend() as backend:
            with self.assertRaises(RunConflict):
                await self.run_audit(backend, expected="different")
            broken = deepcopy(self.bundle)
            broken["manifest"]["pilot_inputs"][0]["input"]["answer"] = "changed"
            with self.assertRaises(RunConflict):
                await self.run_audit(backend, bundle=broken)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.directory.exists())

    async def test_error_stops_and_is_not_retried_by_resume(self):
        async with self.backend(fail=True) as backend:
            result = await self.run_audit(backend)
            self.assertEqual(result["status"], "blocked_by_terminal_attempt")
            self.assertEqual(result["summary"]["examples_pending"], 49)
            self.assertFalse(result["summary"]["all_inputs_fit"])
            before = len(self.calls)
            await self.run_audit(backend)
            self.assertEqual(before, len(self.calls))

    async def test_interrupted_call_is_recorded_and_not_retried(self):
        async with self.backend() as backend:
            async def slow(_):
                await asyncio.sleep(10)
            backend.count_input_tokens = slow
            with patch("post_thesis.llm_judge.audit_pilot.DEADLINE_SECONDS", 0.02):
                result = await self.run_audit(backend)
                self.assertEqual(result["summary"]["examples_failed_or_interrupted"], 1)
                self.assertEqual(result["tokenize_requests_started"], 1)
                again = await self.run_audit(backend)
                self.assertEqual(again["tokenize_requests_started"], 0)
            saved = json.loads((self.directory / "audit.json").read_text())["audit"]
            self.assertEqual(next(iter(saved["counts"].values()))["status"], "interrupted")

    async def test_corrupt_cached_count_rejected_even_with_updated_checksum(self):
        async with self.backend() as backend:
            await self.run_audit(backend)
            path = self.directory / "audit.json"
            saved = json.loads(path.read_text())
            next(iter(saved["audit"]["counts"].values()))["input_tokens"] = True
            saved["audit_sha256"] = content_hash(saved["audit"])
            path.write_text(json.dumps(saved))
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.run_audit(backend)
            self.assertEqual(before, len(self.calls))


if __name__ == "__main__":
    unittest.main()
