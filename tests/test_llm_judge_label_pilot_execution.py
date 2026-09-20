"""Bounded TRAIN label-score execution, provenance and recovery; fake HTTP only."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from post_thesis.llm_judge import label_pilot as contract
from post_thesis.llm_judge import run_label_pilot as runner
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import run_directory, Example
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import Journal, RunConflict
from post_thesis.llm_judge.vllm_backend import load_profile
from post_thesis.llm_judge.audit_label_pilot import summary, LABELS
from post_thesis.llm_judge.label_score_contract import prepare_tokenized_input
from test_llm_judge_label_live import response
from test_llm_judge_label_pilot_audit import PinnedLabelToy
try:
    import httpx
except ImportError:
    httpx = None


def requests():
    rows = []
    for i in range(50):
        slot = 'primary:' + str(i)
        mapping = {'supported':32, 'unsupported':33}
        payload = {'model': load_profile(contract.PROFILE)['served_model_name'],
                   'prompt':[11,12,13], 'temperature':0.0, 'max_tokens':1}
        identity = {'slot':slot, 'payload':payload, 'class_mapping':mapping}
        rows.append({'slot':slot,'sample_id':str(i),'orientation':'primary',
                     'class_mapping':mapping,'payload':payload,'identity':identity,'key':content_hash(identity)})
    return rows

@unittest.skipIf(httpx is None, "isolated HTTP client not installed")
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rows = requests()
        self.plan = contract.plan_descriptor(contract.request_references(self.rows))
        patcher = patch.object(runner, "load_plan", return_value=self.plan)
        patcher.start(); self.addCleanup(patcher.stop)
        self.preparation = {"audit_sha256": contract.AUDIT_SHA256, "source": {"version": "0.29.0", "git_blob_sha1": contract.SOURCE_BLOBS}}
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

    async def run_one(self, *, cap=50, handler=None, server=None, preparation=None):
        async with runner.LabelBackend(transport=httpx.MockTransport(handler or self.handler)) as backend:
            return await runner.execute(backend=backend, revision="test-revision", requests=self.rows,
                preparation=preparation or self.preparation, server_record=server or self.server,
                artifact_root=self.temp.name, max_new_attempts=cap)

    async def test_complete_and_replay_use_no_extra_calls(self):
        first = await self.run_one()
        self.assertEqual(first["valid_scores"], 50)
        self.assertTrue(first["execution_complete"])
        self.assertEqual(first["known_token_totals"], {"input_tokens": 150, "output_tokens": 50})
        self.assertEqual(self.calls.count("/v1/completions"), 50)
        calls = len(self.calls)
        second = await self.run_one()
        inspected = runner.inspect_cached("test-revision", self.temp.name)
        self.assertEqual(second["new_attempts"], 0)
        self.assertEqual(inspected["valid_scores"], 50)
        self.assertEqual(len(self.calls), calls)

    async def test_partial_run_resumes_only_pending_slots(self):
        first = await self.run_one(cap=7)
        self.assertEqual(first["valid_scores"], 7)
        second = await self.run_one()
        self.assertEqual(second["new_attempts"], 43)
        self.assertEqual(self.calls.count("/v1/completions"), 50)
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
        self.assertEqual(result["pending"], 49)
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


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PinnedLabelToy()
        self.examples = [Example(str(i), JudgeInput('A venue.', 'A venue exists.')) for i in range(50)]
        self.state = {'rows':[{'sample_id':e.sample_id,'prepared':prepare_tokenized_input(e.item,self.tokenizer)} for e in self.examples]}
    def test_plan_pins_original_audit_and_bounds(self):
        plan = contract.load_plan()
        self.assertEqual(plan['audit_sha256'], 'b48d1eca31d68032fff55c38f2f82a038138880c72dad29d6f4dec38ebcf51bc')
        self.assertEqual(plan['max_attempts'],50)
        self.assertEqual(plan['max_output_tokens_total'],50)
        self.assertEqual(sum(r['input_tokens'] for r in plan['requests']),60574)
    def test_prepared_payload_excludes_metadata_and_constraints(self):
        rows = contract.prepare_requests(self.examples,self.state,self.tokenizer)
        for row in rows:
            self.assertNotIn('expected_unsupported', row)
            for field in ('label','task','sample_id','generator','allowed_token_ids','logit_bias','guided_json'):
                self.assertNotIn(field,row['payload'])
            self.assertEqual(row['class_mapping'],{'supported':32,'unsupported':33})
            self.assertEqual(row['payload']['max_tokens'],1)
            self.assertEqual(row['payload']['temperature'],0)
        changed = deepcopy(self.state); changed['rows'][0]['prepared']['input_tokens'] += 1
        with self.assertRaises(RunConflict): contract.prepare_requests(self.examples,changed,self.tokenizer)
        with self.assertRaises(RunConflict): contract.prepare_requests(self.examples[::-1],self.state,self.tokenizer)
    def test_audit_hash_and_identity_are_independent_guards(self):
        inputs = [{'sample_id':e.sample_id,'input_sha256':content_hash({'answer':e.item.answer,'context':e.item.context})} for e in self.examples]
        identity = {'study_stage':'post_thesis','version':'label-pilot-token-audit-v1',
                    'code_revision':contract.AUDIT_REVISION,'pilot_manifest_sha256':contract.MANIFEST_SHA256,
                    'synthetic_report_sha256':contract.SYNTHETIC_REPORT_SHA256,
                    'prompt_version':contract.LABEL_PROMPT.version,'prompt_sha256':contract.LABEL_PROMPT.sha256,
                    'profile':load_profile(contract.PROFILE),'class_mapping':LABELS,
                    'output_allowance':1,'truncation':'none','scope':'TRAIN_development_only','inputs':inputs}
        state = {**self.state,'identity':identity,'status':'completed','summary':summary(self.state['rows'],50,32768)}
        digest = content_hash(state); audited = {'audit_sha256':digest,'audit':state}
        bundle = {'manifest_sha256':contract.MANIFEST_SHA256}
        with patch.object(contract,'pilot_examples',return_value=self.examples), patch.object(contract,'AUDIT_SHA256',digest):
            self.assertEqual(contract.validate_inputs(bundle,audited),self.examples)
            bad = deepcopy(audited); bad['audit']['identity']['code_revision'] = 'other'
            with self.assertRaisesRegex(RunConflict,'checksum'): contract.validate_inputs(bundle,bad)
            altered = content_hash(bad['audit']); bad['audit_sha256'] = altered
            with patch.object(contract,'AUDIT_SHA256',altered), self.assertRaisesRegex(RunConflict,'identity'):
                contract.validate_inputs(bundle,bad)
    def test_changed_audit_rejected_before_execution(self):
        preparation = {'audit_sha256':'changed','source':{'version':'0.29.0','git_blob_sha1':contract.SOURCE_BLOBS}}
        with self.assertRaises(RunConflict): contract.validate_preparation(preparation,requests(),contract.load_plan())


if __name__ == '__main__':
    unittest.main()
