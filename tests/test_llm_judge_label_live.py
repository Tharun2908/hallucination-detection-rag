"""Offline transport, provenance and budget regressions; no model calls."""

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import label_live as contract
from post_thesis.llm_judge import label_diagnose as runner
from post_thesis.llm_judge.label_score_contract import LABEL_PROMPT
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import Journal, RunConflict
from post_thesis.llm_judge.vllm_backend import load_profile

try:
    import httpx
except ImportError:
    httpx = None


def requests():
    """Artificial token sequences for transport tests, not the Qwen reference."""
    rows = []
    for slot in contract.slots():
        mapping = {"supported": 32, "unsupported": 33}
        if slot["orientation"] == "swapped":
            mapping = {"supported": 33, "unsupported": 32}
        payload = {"model": load_profile(contract.PROFILE)["served_model_name"],
                   "prompt": [11, 12, 13], "temperature": slot["temperature"],
                   "max_tokens": 1, "logprobs": 2, "logprob_token_ids": [32, 33],
                   "return_token_ids": True, "return_tokens_as_token_ids": True}
        identity = {"slot": slot["slot"], "payload": payload, "class_mapping": mapping}
        rows.append({"slot": slot["slot"], "sample_id": slot["case"]["sample_id"],
                     "orientation": slot["orientation"], "class_mapping": mapping,
                     "expected_unsupported": slot["case"]["expected_unsupported"],
                     "payload": payload, "identity": identity, "key": content_hash(identity)})
    return rows


def response(request, emitted=32):
    values = {"token_id:32": -0.5, "token_id:33": -1.5}
    if emitted not in (32, 33):
        values[f"token_id:{emitted}"] = -3.0
    return {"model": request["payload"]["model"], "id": "offline-fake",
            "usage": {"prompt_tokens": len(request["payload"]["prompt"]), "completion_tokens": 1},
            "choices": [{"index": 0, "text": "A", "finish_reason": "length",
                         "prompt_token_ids": request["payload"]["prompt"], "token_ids": [emitted],
                         "logprobs": {"tokens": [f"token_id:{emitted}"], "text_offset": [0],
                                      "token_logprobs": [values[f"token_id:{emitted}"]],
                                      "top_logprobs": [values]}}]}


class ContractTests(unittest.TestCase):
    def test_frozen_plan_and_diagnostic_orientation(self):
        plan = contract.load_plan()
        self.assertEqual(len(plan["requests"]), 30)
        self.assertEqual(sum(r["input_tokens"] for r in plan["requests"]), 18786)
        self.assertEqual(plan["profile"]["logprobs_mode"], "raw_logprobs")
        self.assertIn("A = supported:", LABEL_PROMPT.system_text)
        self.assertIn("B = supported:", contract.SWAPPED_PROMPT.system_text)
        self.assertIn("A = unsupported:", contract.SWAPPED_PROMPT.system_text)
        self.assertNotIn("--logprobs-mode", server_command())
        self.assertEqual(server_command(profile_name=contract.PROFILE)[-2:], ["--logprobs-mode", "raw_logprobs"])

    def test_valid_score_and_swapped_meaning(self):
        a, b = requests()[0], requests()[14]
        sa = contract.parse_response(response(a), a)
        sb = contract.parse_response(response(b), b)
        self.assertEqual(sa["unsupported_log_odds"], -1)
        self.assertEqual(sb["unsupported_log_odds"], 1)
        self.assertAlmostEqual(sa["unsupported_score"] + sb["unsupported_score"], 1)

    def test_off_label_token_is_retained_not_a_missing_score(self):
        r = requests()[0]
        parsed = contract.parse_response(response(r, emitted=999), r)
        self.assertTrue(parsed["off_label"])
        self.assertEqual(parsed["unsupported_log_odds"], -1)

    def test_model_prefix_position_and_usage_must_match(self):
        r = requests()[0]
        mutations = [lambda x: x.update(model="other"),
                     lambda x: x["choices"][0].update(prompt_token_ids=[11, 12]),
                     lambda x: x["choices"][0].update(token_ids=[32, 33]),
                     lambda x: x["choices"][0]["logprobs"].update(text_offset=[1]),
                     lambda x: x["choices"][0]["logprobs"].update(tokens=["A"]),
                     lambda x: x["usage"].update(completion_tokens=2),
                     lambda x: x.pop("usage"),
                     lambda x: x["choices"][0].update(index=True)]
        for mutate in mutations:
            data = deepcopy(response(r)); mutate(data)
            with self.assertRaises(ValueError):
                contract.parse_response(data, r)

    def test_missing_invalid_censored_or_extra_probabilities_fail(self):
        r = requests()[0]
        for value in (-9999, -10000, float("nan"), float("inf"), 0.1, True, None):
            data = response(r)
            data["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.parse_response(data, r)
        for extra in (False, True):
            data = response(r)
            values = data["choices"][0]["logprobs"]["top_logprobs"][0]
            if extra: values["token_id:444"] = -10
            else: del values["token_id:33"]
            with self.assertRaisesRegex(ValueError, "selected_token"):
                contract.parse_response(data, r)

    def test_sampled_probability_and_mass_are_checked(self):
        r = requests()[0]
        data = response(r); data["choices"][0]["logprobs"]["token_logprobs"] = [-0.2]
        with self.assertRaisesRegex(ValueError, "sampled_token"):
            contract.parse_response(data, r)
        data = response(r)
        data["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"] = -0.1
        with self.assertRaisesRegex(ValueError, "mass"):
            contract.parse_response(data, r)

    def test_source_guard_checks_installed_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.py"; path.write_bytes(b"source")
            wanted = hashlib.sha1(b"blob 6\0source").hexdigest()
            class Dist:
                version = "0.29.0"
                def locate_file(self, name): return Path(root) / name
            with patch.object(contract, "distribution", return_value=Dist()), patch.object(contract, "SOURCE_BLOBS", {"source.py": wanted}):
                self.assertEqual(contract.installed_source()["git_blob_sha1"]["source.py"], wanted)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source differs"):
                    contract.installed_source()


@unittest.skipIf(httpx is None, "isolated HTTP client not installed")
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rows = requests()
        self.plan = contract.plan_descriptor(contract.request_references(self.rows))
        patcher = patch.object(runner, "load_plan", return_value=self.plan)
        patcher.start(); self.addCleanup(patcher.stop)
        self.preparation = {"source": {"version": "0.29.0", "git_blob_sha1": contract.SOURCE_BLOBS}}
        self.server = {"study_stage": "post_thesis", "kind": "serving_session", "status": "started",
                       "code_revision": "test-revision", "profile": load_profile(contract.PROFILE),
                       "gpu": {"name": "NVIDIA H200"}, "command": server_command(profile_name=contract.PROFILE),
                       "package_versions": {"vllm": "0.29.0", "torch": "2.13.0+cu130"},
                       "label_score_source": self.preparation["source"]}
        self.calls = []

    def handler(self, req):
        self.calls.append(req.url.path)
        if req.url.path == "/version": return httpx.Response(200, json={"version": "0.29.0"})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.server["profile"]["served_model_name"], "max_model_len": 32768}]})
        if req.url.path == "/openapi.json":
            return httpx.Response(200, json={"components": {"schemas": {"CompletionRequest": {"properties": {k: {} for k in ("logprobs", "logprob_token_ids", "return_token_ids", "return_tokens_as_token_ids")}}}}})
        self.assertEqual(req.url.path, "/v1/completions")
        payload = json.loads(req.content)
        self.assertNotIn("expected_unsupported", payload)
        return httpx.Response(200, json=response({"payload": payload}))

    async def run_one(self, *, cap=30, handler=None, server=None, preparation=None):
        async with runner.LabelBackend(transport=httpx.MockTransport(handler or self.handler)) as backend:
            return await runner.execute(backend=backend, revision="test-revision", requests=self.rows,
                preparation=preparation or self.preparation, server_record=server or self.server,
                artifact_root=self.temp.name, max_new_attempts=cap)

    async def test_complete_and_replay_use_no_extra_calls(self):
        first = await self.run_one()
        self.assertEqual(first["valid_scores"], 30)
        self.assertTrue(first["transport_checks_passed"])
        self.assertEqual(first["known_token_totals"], {"input_tokens": 90, "output_tokens": 30})
        self.assertEqual(self.calls.count("/v1/completions"), 30)
        calls = len(self.calls)
        second = await self.run_one()
        inspected = runner.inspect_cached("test-revision", self.temp.name)
        self.assertEqual(second["new_attempts"], 0)
        self.assertEqual(inspected["valid_scores"], 30)
        self.assertEqual(len(self.calls), calls)

    async def test_partial_run_resumes_only_pending_slots(self):
        first = await self.run_one(cap=7)
        self.assertEqual(first["valid_scores"], 7)
        second = await self.run_one()
        self.assertEqual(second["new_attempts"], 23)
        self.assertEqual(self.calls.count("/v1/completions"), 30)
        self.assertEqual(len(second["execution_windows"]), 2)

    async def test_invalid_response_halts_and_preserves_usage(self):
        def bad(req):
            result = self.handler(req)
            if req.url.path == "/v1/completions":
                value = result.json(); del value["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"]
                return httpx.Response(200, json=value)
            return result
        result = await self.run_one(handler=bad)
        self.assertEqual(result["terminal_failures"], 1)
        self.assertEqual(result["pending"], 29)
        self.assertEqual(result["known_token_totals"]["output_tokens"], 1)
        self.assertIsNone(result["predictions"][0]["score"])
        await self.run_one()
        self.assertEqual(self.calls.count("/v1/completions"), 1)

    async def test_http_failure_unknown_usage_and_no_retry(self):
        def bad(req):
            if req.url.path == "/v1/completions":
                self.calls.append(req.url.path)
                return httpx.Response(500, text="failed")
            return self.handler(req)
        result = await self.run_one(handler=bad)
        self.assertEqual(result["unknown_usage_attempts"], {"input_tokens": 1, "output_tokens": 1})
        self.assertEqual(result["halt_reason"], "http_500")
        await self.run_one()
        self.assertEqual(self.calls.count("/v1/completions"), 1)

    async def test_duplicate_json_fields_rejected(self):
        def bad(req):
            result = self.handler(req)
            if req.url.path == "/v1/completions":
                return httpx.Response(200, content=b'{"model":"x","model":"y"}')
            return result
        result = await self.run_one(handler=bad)
        self.assertEqual(result["halt_reason"], "invalid_http_json")
        self.assertEqual(result["valid_scores"], 0)

    async def test_wrong_server_command_fails_before_http(self):
        server = deepcopy(self.server); server["command"] = server_command()
        with self.assertRaises(RunConflict): await self.run_one(server=server)
        self.assertFalse(self.calls)

    async def test_preflight_failure_makes_no_generation_calls(self):
        def bad(req):
            self.calls.append(req.url.path)
            return httpx.Response(200, json={"version": "different"})
        result = await self.run_one(handler=bad)
        self.assertEqual(result["halt_reason"], "server_version_mismatch")
        self.assertEqual(result["new_attempts"], 0)
        await self.run_one()
        self.assertNotIn("/v1/completions", self.calls)

    async def test_unknown_window_exhausts_budget_and_does_not_repeat(self):
        await self.run_one(cap=1)
        directory = run_directory(contract.RUN_ID, self.temp.name)
        ledger = runner._load(directory / "budget.json")
        ledger["windows"][0]["status"] = "started"
        ledger["windows"][0]["elapsed_seconds"] = None
        runner._save(directory / "budget.json", ledger)
        before = len(self.calls)
        result = await self.run_one()
        self.assertEqual(result["remaining_client_seconds"], 0)
        self.assertEqual(len(self.calls), before)

    async def test_interrupted_attempt_has_no_score_or_repeat(self):
        await self.run_one(cap=1)
        directory = run_directory(contract.RUN_ID, self.temp.name)
        ledger = runner._load(directory / "budget.json")
        journal = Journal(directory / "journal.sqlite3", ledger["identity"])
        try: journal.start(self.rows[1]["key"], 1)
        finally: journal.close()
        result = await self.run_one()
        self.assertEqual(result["halt_reason"], "interrupted_request")
        self.assertEqual(result["predictions"][1]["status"], "interrupted")
        self.assertIsNone(result["predictions"][1]["score"])
        self.assertEqual(self.calls.count("/v1/completions"), 1)

    async def test_changed_identity_or_missing_journal_fails_closed(self):
        await self.run_one(cap=1)
        with self.assertRaises(RunConflict):
            await self.run_one(preparation={**self.preparation, "changed": True})
        directory = run_directory(contract.RUN_ID, self.temp.name)
        (directory / "journal.sqlite3").unlink()
        with self.assertRaises(RunConflict): await self.run_one()

    async def test_temperature_control_mismatch_is_reported_not_retuned(self):
        def change(req):
            result = self.handler(req)
            if req.url.path == "/v1/completions" and json.loads(req.content)["temperature"] == 1:
                value = result.json()
                value["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:33"] = -2
                return httpx.Response(200, json=value)
            return result
        result = await self.run_one(handler=change)
        self.assertEqual(result["valid_scores"], 30)
        self.assertFalse(result["transport_checks_passed"])
        self.assertFalse(result["numeric_controls"][1]["matches"])

    async def test_cancelled_request_is_missing_and_cannot_repeat(self):
        async def cancelled(req):
            if req.url.path == "/v1/completions":
                self.calls.append(req.url.path)
                raise asyncio.CancelledError()
            return self.handler(req)
        result = await self.run_one(handler=cancelled)
        self.assertEqual(result["terminal_failures"], 1)
        self.assertEqual(result["unknown_usage_attempts"]["output_tokens"], 1)
        self.assertIsNone(result["predictions"][0]["score"])
        await self.run_one()
        self.assertEqual(self.calls.count("/v1/completions"), 1)

    async def test_changed_payload_cannot_reuse_a_frozen_key(self):
        self.rows[0]["payload"]["prompt"] = [91, 92, 93]
        with self.assertRaisesRegex(RunConflict, "identity"):
            await self.run_one()
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
