"""Binary output isolation, strict parsing and bounded diagnostic execution."""

import asyncio
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

from post_thesis.llm_judge import binary_diagnose as diagnose
from post_thesis.llm_judge.binary_contract import BINARY_PROMPT, BINARY_SCHEMA_JSON, binary_request, parse_verdict
from post_thesis.llm_judge.parse import JudgeParseError
from post_thesis.llm_judge.diagnostic_cases import case_records, cases_sha256, comparisons, examples, FACTS
from post_thesis.llm_judge.judge import build_request
from post_thesis.llm_judge.prompts import DEVELOPMENT_PROMPT_V2
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

P = load_profile()


class ContractTests(unittest.TestCase):
    def test_strict_verdict_parser(self):
        for value in ("supported", "unsupported"):
            self.assertEqual(parse_verdict(json.dumps({"verdict": value})), value)
        invalid = [None, "", '[]', 'null', '{"verdict":true}', '{"verdict":1}',
                   '{"verdict":"SUPPORTED"}', '{"verdict":"unknown"}',
                   '{"verdict":"supported","verdict":"unsupported"}',
                   '{"verdict":"supported","explanation":"extra"}',
                   '{"unsupported_probability":0.5}',
                   '```json\n{"verdict":"supported"}\n```',
                   '{"verdict":NaN}', '{"verdict":["supported"]}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(JudgeParseError):
                parse_verdict(text)

    def test_prompt_and_schema_are_distinct_but_inputs_are_identical(self):
        from post_thesis.llm_judge.judge import JudgeConfig
        config = JudgeConfig("offline", "model", "version")
        self.assertEqual(DEVELOPMENT_PROMPT_V2.sha256,
                         "7771610009b40b5cce476fa4abda9ecb6bd025ecf71d635d7d3e7feb3c70135a")
        self.assertEqual(BINARY_PROMPT.sha256,
                         "e6b7d43ec5f34b07de69d11dacae6212d8c59e97b69b7375375ea8f289544683")
        for e in examples():
            probability = build_request(e.item, config, DEVELOPMENT_PROMPT_V2)
            binary = binary_request(e.item, config)
            self.assertEqual(probability.messages[1], binary.messages[1])
            self.assertEqual(probability.config, binary.config)
            self.assertNotEqual(probability.key, binary.key)
            self.assertNotEqual(probability.contract_version, binary.contract_version)
            self.assertEqual(set(json.loads(binary.response_schema_json)["properties"]), {"verdict"})
        self.assertNotIn("Return the probability", BINARY_PROMPT.system_text)
        self.assertNotIn("the probability.", BINARY_PROMPT.system_text)

    def test_missing_verdict_is_not_false_or_supported(self):
        preds = [{"sample_id": r["sample_id"], "verdict": None} for r in case_records()]
        summary = diagnose.comparisons({"predictions": preds})
        self.assertEqual(len(summary), 10)
        self.assertTrue(all(r["matches_expected"] is None for r in summary))


@unittest.skipIf(httpx is None, 'HTTP diagnostic tests need the isolated client')
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = run_directory(diagnose.load_plan()['run_id'], self.root)
        self.calls = []
        self.count = 900
        self.output_tokens = 9
        self.slow = False
        self.http_status = 200
        self.output_text = '{"verdict": "supported"}'
        self.finish_reason = "stop"
        self.server = {'study_stage': 'post_thesis', 'kind': 'serving_session', 'status': 'started',
                       'code_revision': 'diagnostic-code', 'profile': P, 'gpu': {'name': 'NVIDIA H200'}}

    def backend(self):
        async def handler(request):
            self.calls.append(request)
            if request.url.path == '/version':
                return httpx.Response(200, json={'version': P['vllm_version']})
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': P['served_model_name'], 'max_model_len': P['max_model_len']}]})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'count': self.count, 'max_model_len': P['max_model_len']})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            if self.slow:
                await asyncio.sleep(10)
            return httpx.Response(self.http_status, json={'id': 'mock', 'model': P['served_model_name'],
                'usage': {'prompt_tokens': self.count, 'completion_tokens': self.output_tokens},
                'choices': [{'finish_reason': self.finish_reason, 'message': {'role': 'assistant', 'content': self.output_text}}]})
        return VLLMBackend(transport=httpx.MockTransport(handler))

    async def execute(self, backend, **kwargs):
        return await diagnose.execute(backend=backend, revision='diagnostic-code', server_record=self.server,
                                      artifact_root=self.root, **kwargs)

    def state(self):
        path = self.directory / 'execution' / 'budget.json'
        return path, json.loads(path.read_text())['execution']

    async def test_exact_ten_requests_no_expected_labels_and_zero_call_replay(self):
        async with self.backend() as backend:
            result = await self.execute(backend, clock=iter([0, 12]).__next__)
            self.assertEqual(result['report']['examples_judged'], 10)
            self.assertEqual(result['new_attempts_this_invocation'], 10)
            self.assertEqual(result['charged_client_seconds'], 12)
            wires = [r for r in self.calls if r.url.path == '/v1/chat/completions']
            self.assertEqual(len(wires), 10)
            for wire, example in zip(wires, examples()):
                payload = json.loads(wire.content)
                self.assertEqual(payload['messages'], [asdict(m) for m in binary_request(example.item, backend.config).messages])
                self.assertEqual(set(json.loads(payload['messages'][1]['content'])), {'answer', 'context'})
                self.assertEqual(payload['max_completion_tokens'], 128)
                self.assertEqual(payload['response_format']['json_schema']['name'], 'faithfulness_verdict')
                self.assertEqual(payload['response_format']['json_schema']['schema'], json.loads(BINARY_SCHEMA_JSON))
            before = len(self.calls)
            self.server['status'] = 'stopped'
            cached = await self.execute(backend)
            self.assertEqual(cached['new_attempts_this_invocation'], 0)
            self.assertEqual(len(self.calls), before)
            self.assertEqual(cached['charged_client_seconds'], 12)

    async def test_inspection_then_execution_and_wrong_server(self):
        async with self.backend() as backend:
            result = await self.execute(backend, inspection_only=True)
            self.assertEqual(self.calls, [])
            self.assertTrue(result['generation_invocation_available'])
            self.server['code_revision'] = 'wrong'
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.server['code_revision'] = 'diagnostic-code'
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_judged'], 10)

    async def test_unknown_window_cannot_be_restarted(self):
        async with self.backend() as backend:
            await self.execute(backend, inspection_only=True)
            path, state = self.state()
            state.update(status='started', reserved_seconds=300)
            diagnose._save(path, state)
            result = await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertEqual(result['charged_client_seconds'], 300)
            self.assertFalse(result['generation_invocation_available'])
            self.assertEqual(result['report']['examples_pending'], 10)

    async def test_overlength_blocks_generation_and_no_new_calls_on_repeat(self):
        async with self.backend() as backend:
            self.count = 4097
            result = await self.execute(backend)
            self.assertEqual(result['error_code'], 'diagnostic_input_limit')
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            self.assertEqual(result['report']['examples_pending'], 9)
            self.assertFalse(any(r.url.path == '/v1/chat/completions' for r in self.calls))
            before = len(self.calls)
            self.count = 900
            await self.execute(backend)
            self.assertEqual(len(self.calls), before)

    async def test_output_violation_preserves_known_usage_and_halts(self):
        async with self.backend() as backend:
            self.output_tokens = 129
            result = await self.execute(backend)
            self.assertEqual(result['error_code'], 'output_token_limit_exceeded')
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 129)
            self.assertEqual(result['report']['examples_pending'], 9)

    async def test_failed_attempts_never_become_scores_or_retries(self):
        async with self.backend() as backend:
            self.http_status = 503
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_terminal_failure'], 10)
            self.assertTrue(all(r['verdict'] is None for r in result['report']['predictions']))
            before = len(self.calls)
            self.http_status = 200
            result = await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'], 0)
            self.assertEqual(len(self.calls), before)

    async def test_deadline_recovers_interrupted_attempt_without_new_allowance(self):
        original = asyncio.wait_for
        async def bounded(operation, timeout):
            return await original(operation, timeout=0.5 if timeout == 300 else timeout)
        async with self.backend() as backend:
            self.slow = True
            with patch.object(diagnose.asyncio, 'wait_for', bounded):
                result = await self.execute(backend)
            self.assertEqual(result['outcome'], 'deadline_reached')
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            self.assertEqual(result['report']['examples_pending'], 9)
            before = len(self.calls)
            result = await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'], 0)
            self.assertEqual(len(self.calls), before)

    async def test_missing_budget_or_journal_and_changed_code_are_rejected(self):
        async with self.backend() as backend:
            await self.execute(backend)
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await diagnose.execute(backend=backend, revision='changed-code', artifact_root=self.root)
            path, _ = self.state()
            raw = path.read_bytes()
            path.unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            path.write_bytes(raw)
            (self.directory / 'journal.sqlite3').unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(len(self.calls), before)

    async def test_changed_plan_fails_before_writes_or_network(self):
        plan = diagnose.load_plan()
        plan['cases_sha256'] = 'changed'
        async with self.backend() as backend:
            with self.assertRaises(RunConflict), patch.object(diagnose, 'load_plan', return_value=plan):
                await self.execute(backend)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.directory.exists())

    async def test_probability_output_is_not_coerced_to_binary(self):
        async with self.backend() as backend:
            self.output_text = '{"unsupported_probability":1.0}'
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_terminal_failure'], 10)
            self.assertEqual(result['report']['examples_judged'], 0)
            self.assertTrue(all(r['verdict'] is None for r in result['report']['predictions']))
            self.assertNotIn('unsupported_probability', json.dumps(result))
            before = len(self.calls)
            self.output_text = '{"verdict":"unsupported"}'
            await self.execute(backend)
            self.assertEqual(before, len(self.calls))

    async def test_truncated_binary_json_is_not_accepted(self):
        async with self.backend() as backend:
            self.finish_reason = 'length'
            result = await self.execute(backend)
            self.assertTrue(all(r['status'] == 'incomplete' and r['verdict'] is None for r in result['report']['predictions']))
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 90)

    async def test_cached_verdict_must_match_raw_response(self):
        from contextlib import closing
        import sqlite3
        from post_thesis.llm_judge.prompts import canonical_json, content_hash
        async with self.backend() as backend:
            await self.execute(backend)
            path = self.directory / 'journal.sqlite3'
            # The transaction context commits but does not close the connection.
            # Explicit closure also releases the file handle before Windows cleanup.
            with closing(sqlite3.connect(path)) as db, db:
                row_id, text = db.execute('SELECT id, result FROM attempts ORDER BY id LIMIT 1').fetchone()
                result = json.loads(text)
                result['verdict'] = 'unsupported'
                db.execute('UPDATE attempts SET result=?,result_hash=? WHERE id=?',
                           (canonical_json(result), content_hash(result), row_id))
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before, len(self.calls))


if __name__ == '__main__':
    unittest.main()
