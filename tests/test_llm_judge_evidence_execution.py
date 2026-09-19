"""Bounded evidence execution with synthetic HTTP responses; no GPU or network."""

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

from post_thesis.llm_judge import evidence_diagnose as diagnose
from post_thesis.llm_judge.evidence_contract import evidence_request, EVIDENCE_SCHEMA_JSON
from post_thesis.llm_judge.evidence_cases import examples, case_records
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.prompts import content_hash

P = load_profile('evidence-v1')
SUPPORTED = dict(answer_quote=None, context_quote=None, issue_type='none', explanation=None, verdict='supported')


class ProfileTests(unittest.TestCase):
    def test_default_profile_preserved_and_new_profile_explicit(self):
        old = load_profile()
        self.assertEqual(content_hash(old), '634eb8385c47ba3495c7568029520d1e9471dcd0b190c7931a9c87d614b9f9d5')
        self.assertEqual(old['max_output_tokens'], 128)
        self.assertEqual(P['max_output_tokens'], 512)
        self.assertEqual({k for k in old if old[k] != P[k]},
                         {'profile_version', 'served_model_name', 'max_output_tokens'})
        self.assertIn(P['served_model_name'], server_command(profile_name='evidence-v1'))
        self.assertIn(old['served_model_name'], server_command())
        with self.assertRaises(ValueError):
            load_profile('unreviewed')


@unittest.skipIf(httpx is None, 'HTTP diagnostic tests need the isolated client')
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = run_directory(diagnose.load_plan()['run_id'], self.root)
        self.calls = []
        self.tokenization_calls = 0
        self.token_overrides = {}
        self.count = 900
        self.output_tokens = 9
        self.slow = False
        self.generation_started = asyncio.Event()
        self.output_factory = None
        self.http_status = 200
        self.output_text = json.dumps(SUPPORTED)
        self.finish_reason = "stop"
        self.server = {'study_stage': 'post_thesis', 'kind': 'serving_session', 'status': 'started',
                       'code_revision': 'evidence-code', 'profile': P, 'gpu': {'name': 'NVIDIA H200'}}

    def backend(self):
        async def handler(request):
            self.calls.append(request)
            if request.url.path == '/version':
                return httpx.Response(200, json={'version': P['vllm_version']})
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': P['served_model_name'], 'max_model_len': P['max_model_len']}]})
            if request.url.path == '/tokenize':
                self.tokenization_calls += 1
                count = self.token_overrides.get(self.tokenization_calls, self.count)
                return httpx.Response(200, json={'count': count, 'max_model_len': P['max_model_len']})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            self.generation_started.set()
            if self.slow:
                await asyncio.sleep(14)
            output = self.output_factory(json.loads(request.content)) if self.output_factory else self.output_text
            return httpx.Response(self.http_status, json={'id': 'mock', 'model': P['served_model_name'],
                'usage': {'prompt_tokens': self.count, 'completion_tokens': self.output_tokens},
                'choices': [{'finish_reason': self.finish_reason, 'message': {'role': 'assistant', 'content': output}}]})
        return VLLMBackend(transport=httpx.MockTransport(handler), profile_name="evidence-v1")

    async def execute(self, backend, **kwargs):
        return await diagnose.execute(backend=backend, revision='evidence-code', server_record=self.server,
                                      artifact_root=self.root, **kwargs)

    def state(self):
        path = self.directory / 'execution' / 'budget.json'
        return path, json.loads(path.read_text())['execution']

    async def test_exact_fourteen_requests_no_expected_labels_and_zero_call_replay(self):
        async with self.backend() as backend:
            result = await self.execute(backend, clock=iter([0, 12]).__next__)
            self.assertEqual(result['report']['examples_valid'], 14)
            self.assertEqual(result['new_attempts_this_invocation'], 14)
            self.assertEqual(result['charged_client_seconds'], 12)
            wires = [r for r in self.calls if r.url.path == '/v1/chat/completions']
            self.assertEqual(len(wires), 14)
            self.assertEqual(self.tokenization_calls, 28)
            first_generation = next(i for i, r in enumerate(self.calls) if r.url.path == '/v1/chat/completions')
            self.assertEqual(sum(r.url.path == '/tokenize' for r in self.calls[:first_generation]), 15)
            self.assertEqual(result['input_audit']['status'], 'completed')
            self.assertEqual(sum(r['verdict_matches_expected'] is True for r in result['comparisons']), 6)
            self.assertTrue(all(r['semantic_evidence_review'] == 'pending' for r in result['comparisons']))
            for wire, example in zip(wires, examples()):
                payload = json.loads(wire.content)
                self.assertEqual(payload['messages'], [asdict(m) for m in evidence_request(example.item, backend.config).messages])
                self.assertEqual(set(json.loads(payload['messages'][1]['content'])), {'answer', 'context'})
                self.assertEqual(payload['max_completion_tokens'], 512)
                self.assertEqual(payload['response_format']['json_schema']['name'], 'faithfulness_evidence_verdict')
                self.assertEqual(payload['response_format']['json_schema']['schema'], json.loads(EVIDENCE_SCHEMA_JSON))
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
            self.server['code_revision'] = 'evidence-code'
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 14)

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
            self.assertEqual(result['report']['examples_pending'], 14)

    async def test_overlength_blocks_generation_and_no_new_calls_on_repeat(self):
        async with self.backend() as backend:
            self.count = 4097
            result = await self.execute(backend)
            self.assertEqual(result['error_code'], 'diagnostic_input_limit')
            self.assertEqual(result['report']['examples_terminal_failure'], 0)
            self.assertEqual(result['report']['examples_pending'], 14)
            self.assertEqual(result['input_audit']['status'], 'failed')
            self.assertFalse(any(r.url.path == '/v1/chat/completions' for r in self.calls))
            before = len(self.calls)
            self.count = 900
            await self.execute(backend)
            self.assertEqual(len(self.calls), before)

    async def test_output_violation_preserves_known_usage_and_halts(self):
        async with self.backend() as backend:
            self.output_tokens = 513
            result = await self.execute(backend)
            self.assertEqual(result['error_code'], 'output_token_limit_exceeded')
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 513)
            self.assertEqual(result['report']['examples_pending'], 13)

    async def test_failed_attempts_never_become_scores_or_retries(self):
        async with self.backend() as backend:
            self.http_status = 503
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_terminal_failure'], 14)
            self.assertTrue(all(r['verdict'] is None for r in result['report']['predictions']))
            before = len(self.calls)
            self.http_status = 200
            result = await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'], 0)
            self.assertEqual(len(self.calls), before)

    async def test_deadline_recovers_interrupted_attempt_without_new_allowance(self):
        original = asyncio.wait_for
        async def bounded(operation, timeout):
            if timeout != 300:
                return await original(operation, timeout=timeout)
            task = asyncio.create_task(operation)
            try:
                # Trigger deadline cancellation after generation begins, without
                # relying on platform-dependent file-system execution speed.
                await original(self.generation_started.wait(), timeout=10)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise asyncio.TimeoutError
        async with self.backend() as backend:
            self.slow = True
            with patch.object(diagnose.asyncio, 'wait_for', bounded):
                result = await self.execute(backend)
            self.assertEqual(result['outcome'], 'deadline_reached')
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            self.assertEqual(result['report']['examples_pending'], 13)
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

    async def test_probability_output_is_not_coerced_to_evidence(self):
        async with self.backend() as backend:
            self.output_text = '{"unsupported_probability":1.0}'
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_terminal_failure'], 14)
            self.assertEqual(result['report']['examples_valid'], 0)
            self.assertTrue(all(r['verdict'] is None for r in result['report']['predictions']))
            self.assertNotIn('unsupported_probability', json.dumps(result))
            before = len(self.calls)
            self.output_text = json.dumps(SUPPORTED)
            await self.execute(backend)
            self.assertEqual(before, len(self.calls))

    async def test_truncated_evidence_json_is_not_accepted(self):
        async with self.backend() as backend:
            self.finish_reason = 'length'
            result = await self.execute(backend)
            self.assertTrue(all(r['status'] == 'incomplete' and r['verdict'] is None for r in result['report']['predictions']))
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 126)

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
                result['evidence']['verdict'] = 'unsupported'
                db.execute('UPDATE attempts SET result=?,result_hash=? WHERE id=?',
                           (canonical_json(result), content_hash(result), row_id))
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before, len(self.calls))

    async def test_last_input_overlength_prevents_all_generation(self):
        async with self.backend() as backend:
            self.token_overrides[14] = 4097
            result = await self.execute(backend)
            self.assertEqual(self.tokenization_calls, 14)
            self.assertEqual(result['new_attempts_this_invocation'], 0)
            self.assertEqual(result['input_audit']['status'], 'failed')
            self.assertFalse(any(r.url.path == '/v1/chat/completions' for r in self.calls))

    async def test_retokenization_mismatch_blocks_generation_and_replay(self):
        async with self.backend() as backend:
            self.token_overrides[15] = 901
            result = await self.execute(backend)
            self.assertEqual(result['outcome'], 'halted_on_alignment_error')
            self.assertEqual(result['error_code'], 'audited_input_token_mismatch')
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            self.assertFalse(any(r.url.path == '/v1/chat/completions' for r in self.calls))
            before = len(self.calls)
            await self.execute(backend)
            self.assertEqual(len(self.calls), before)

    async def test_invalid_evidence_does_not_salvage_verdict_and_preserves_usage(self):
        async with self.backend() as backend:
            self.output_text = json.dumps(dict(answer_quote='THIS QUOTE IS NOT IN THE ANSWER',
                context_quote=None, issue_type='insufficient_support', explanation='Fixture', verdict='unsupported'))
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 0)
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 126)
            self.assertTrue(all(p['evidence'] is None and p['verdict'] is None and
                                p['error_code'] == 'answer_quote_not_in_input' for p in result['report']['predictions']))
            self.assertTrue(all(p['verdict_matches_expected'] is None for p in result['comparisons']))
            before = len(self.calls)
            self.output_text = json.dumps(SUPPORTED)
            await self.execute(backend)
            self.assertEqual(len(self.calls), before)

    async def test_refused_valid_json_remains_missing(self):
        async with self.backend() as backend:
            self.finish_reason = 'content_filter'
            result = await self.execute(backend)
            self.assertTrue(all(p['status'] == 'refused' and p['evidence'] is None and p['verdict'] is None
                                for p in result['report']['predictions']))

    async def test_default_backend_and_default_server_profile_rejected(self):
        async with VLLMBackend(transport=httpx.MockTransport(lambda r: None)) as backend:
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertFalse(self.directory.exists())
        async with self.backend() as backend:
            self.server['profile'] = load_profile()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])

    async def test_invalid_token_count_fails_audit_without_generation(self):
        async with self.backend() as backend:
            self.token_overrides[1] = True
            result = await self.execute(backend)
            self.assertEqual(result['error_code'], 'invalid_token_count')
            self.assertEqual(result['input_audit']['status'], 'failed')
            self.assertEqual(result['new_attempts_this_invocation'], 0)

    async def test_corrupt_finished_duration_rejected_even_with_recomputed_checksum(self):
        async with self.backend() as backend:
            await self.execute(backend)
            path, state = self.state()
            state['resources']['wall_seconds'] = -1
            diagnose._save(path, state)
            before = len(self.calls)
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before, len(self.calls))

    async def test_valid_unsupported_evidence_roundtrips_through_journal(self):
        by_answer = {row['input']['answer'] + row['input']['context']: row for row in case_records()}
        def fixture(payload):
            item = json.loads(payload['messages'][1]['content'])
            row = by_answer[item['answer'] + item['context']]
            if row['expected_verdict'] == 'supported':
                return json.dumps(SUPPORTED)
            if row['family'] == 'numeric':
                quote = 'The selected group comprises the highest 60% of recorded scores.'
                context_quote = 'The selected group comprises the lowest 60% of recorded scores.'
            else:
                quote = ('The venue does not offer outdoor seating.' if row['family'] == 'absence'
                         else 'The venue offers outdoor seating.')
                context_quote = item['context'].split('. ', 1)[0]
            return json.dumps(dict(answer_quote=quote, context_quote=context_quote,
                issue_type=row['expected_issue_type'], explanation='Offline evidence fixture.', verdict='unsupported'))
        self.output_factory = fixture
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 14)
            self.assertTrue(all(r['verdict_matches_expected'] and r['issue_matches_expected']
                                for r in result['comparisons']))
            before = len(self.calls)
            cached = await self.execute(backend)
            self.assertEqual(before, len(self.calls))
            self.assertEqual(result['report']['predictions'], cached['report']['predictions'])


if __name__ == '__main__':
    unittest.main()
