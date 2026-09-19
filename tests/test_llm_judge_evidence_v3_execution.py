"""Live-request contract exercised through offline HTTP; no model calls."""

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

from post_thesis.llm_judge import evidence_diagnose as diagnostic
from post_thesis.llm_judge.evidence_cases import examples
from post_thesis.llm_judge.evidence_contract import evidence_request
from post_thesis.llm_judge.evidence_schema_v3 import evidence_request_v3, EVIDENCE_SCHEMA_V3_JSON
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

PROFILE = load_profile('evidence-v1')
SUPPORTED = dict(verdict='supported', issue_type='none', answer_quote=None,
                 context_quote=None, explanation=None)


@unittest.skipIf(httpx is None, 'Requires isolated HTTP client')
class SchemaV3ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = run_directory(diagnostic.load_plan('v3')['run_id'], self.root)
        self.calls = []
        self.tokens = 0
        self.overlength_at = None
        self.output = json.dumps(SUPPORTED)
        self.http_status = 200
        self.started = asyncio.Event()
        self.slow = False
        self.server = {'study_stage': 'post_thesis', 'kind': 'serving_session', 'status': 'started',
                       'code_revision': 'v3-test', 'profile': PROFILE, 'gpu': {'name': 'NVIDIA H200'},
                       'command': server_command(profile_name='evidence-v1'),
                       'package_versions': diagnostic.native_observation('v3')['versions']}

    def backend(self):
        async def handle(request):
            self.calls.append(request)
            path = request.url.path
            if path == '/version':
                return httpx.Response(200, json={'version': PROFILE['vllm_version']})
            if path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': PROFILE['served_model_name'],
                                                        'max_model_len': PROFILE['max_model_len']}]})
            if path == '/tokenize':
                self.tokens += 1
                return httpx.Response(200, json={'count': 4097 if self.tokens == self.overlength_at else 900,
                                                'max_model_len': PROFILE['max_model_len']})
            self.assertEqual(path, '/v1/chat/completions')
            self.started.set()
            if self.slow:
                await asyncio.sleep(30)
            return httpx.Response(self.http_status, json={'id': 'fixture', 'model': PROFILE['served_model_name'],
                'usage': {'prompt_tokens': 900, 'completion_tokens': 40},
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': self.output}}]})
        return VLLMBackend(profile_name='evidence-v1', transport=httpx.MockTransport(handle))

    async def execute(self, backend, **kwargs):
        return await diagnostic.execute(backend=backend, revision='v3-test', server_record=self.server,
                                        artifact_root=self.root, schema_version='v3', **kwargs)

    async def test_v3_wire_same_messages_new_schema_separate_cache_and_bounded_replay(self):
        async with self.backend() as backend:
            result = await self.execute(backend, clock=iter([0, 10]).__next__)
            self.assertEqual(result['report']['examples_valid'], 14)
            self.assertEqual(result['report']['response_order_summary']['matches'], 14)
            self.assertEqual(result['new_attempts_this_invocation'], 14)
            self.assertEqual(result['charged_client_seconds'], 10)
            self.assertEqual(self.tokens, 28)
            wires = [r for r in self.calls if r.url.path == '/v1/chat/completions']
            self.assertEqual(len(wires), 14)
            for wire, example in zip(wires, examples()):
                payload = json.loads(wire.content)
                old = evidence_request(example.item, backend.config)
                new = evidence_request_v3(example.item, backend.config)
                self.assertNotEqual(old.key, new.key)
                self.assertEqual(payload['messages'], [asdict(m) for m in old.messages])
                self.assertEqual(set(json.loads(payload['messages'][1]['content'])), {'answer', 'context'})
                self.assertEqual(payload['response_format']['json_schema']['schema'], json.loads(EVIDENCE_SCHEMA_V3_JSON))
                for branch in payload['response_format']['json_schema']['schema']['anyOf']:
                    self.assertEqual(list(branch['properties']), ['verdict', 'issue_type', 'answer_quote', 'context_quote', 'explanation'])
                self.assertEqual(payload['max_completion_tokens'], 512)
            manifest = json.loads((self.directory / 'manifest.json').read_text())
            self.assertTrue(all(r['contract_version'] == 'evidence-diagnostic-request-v3' and
                                r['response_schema_json'] == EVIDENCE_SCHEMA_V3_JSON
                                for r in manifest['requests'].values()))
            # Validity is not verdict accuracy: our all-supported mock misses 8.
            self.assertEqual(sum(r['verdict_matches_expected'] for r in result['comparisons']), 6)
            count = len(self.calls)
            self.server['status'] = 'stopped'
            cached = await self.execute(backend)
            self.assertEqual(cached['new_attempts_this_invocation'], 0)
            self.assertEqual(cached['charged_client_seconds'], 10)
            self.assertEqual(len(self.calls), count)
            self.assertFalse(run_directory(diagnostic.load_plan()['run_id'], self.root).exists())

    async def test_mismatched_environment_or_command_blocks_before_network(self):
        async with self.backend() as backend:
            for change in ('environment', 'command'):
                with self.subTest(change=change):
                    old = dict(self.server)
                    if change == 'environment':
                        self.server['package_versions'] = {**self.server['package_versions'], 'xgrammar': 'other'}
                    else:
                        self.server['command'] = self.server['command'] + ['--structured-outputs-config.backend', 'xgrammar']
                    with self.assertRaises(RunConflict):
                        await self.execute(backend)
                    self.server = old
            self.assertEqual(self.calls, [])

    async def test_changed_compatibility_or_budget_cannot_enable_calls(self):
        original = diagnostic.load_plan
        async with self.backend() as backend:
            for key, value in [('native_observation_sha256', 'changed'), ('client_budget_seconds', 600),
                               ('max_attempts_per_input', 2), ('structured_output_backend_policy', 'xgrammar'),
                               ('serialized_schema_sha256', 'changed'), ('expected_response_field_order', ['verdict'])]:
                altered = original('v3'); altered[key] = value
                with self.subTest(key=key), patch.object(diagnostic, 'load_plan', return_value=altered):
                    with self.assertRaises(RunConflict):
                        await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.directory.exists())

    async def test_last_input_overlength_blocks_all_generation_and_replay(self):
        self.overlength_at = 14
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'], 0)
            self.assertEqual(result['error_code'], 'diagnostic_input_limit')
            count = len(self.calls)
            self.overlength_at = None
            await self.execute(backend)
            self.assertEqual(count, len(self.calls))

    async def test_schema_http_rejection_halts_without_fallback_or_new_allowance(self):
        self.http_status = 400
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'], 1)
            self.assertEqual(result['outcome'], 'halted_on_serving_error')
            self.assertEqual(result['error_code'], 'http_400')
            self.assertEqual(result['report']['examples_pending'], 13)
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            count = len(self.calls)
            self.http_status = 200
            await self.execute(backend)
            self.assertEqual(count, len(self.calls))

    async def test_old_supported_violation_stays_missing_even_with_correct_verdict(self):
        self.output = json.dumps({**SUPPORTED, 'explanation': 'This is supported.'})
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 0)
            self.assertTrue(all(p['verdict'] is None and p['error_code'] == 'inconsistent_supported_output'
                                for p in result['report']['predictions']))
            self.assertEqual(result['report']['known_token_totals']['output_tokens'], 560)

    async def test_interrupted_invocation_is_charged_and_cannot_restart(self):
        original = asyncio.wait_for
        async def deadline(operation, timeout):
            if timeout != 300:
                return await original(operation, timeout=timeout)
            task = asyncio.create_task(operation)
            try:
                await original(self.started.wait(), timeout=10)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise asyncio.TimeoutError
        self.slow = True
        async with self.backend() as backend:
            with patch.object(diagnostic.asyncio, 'wait_for', deadline):
                result = await self.execute(backend)
            self.assertEqual(result['outcome'], 'deadline_reached')
            self.assertEqual(result['report']['examples_terminal_failure'], 1)
            self.assertEqual(result['report']['examples_pending'], 13)
            count = len(self.calls)
            await self.execute(backend)
            self.assertEqual(count, len(self.calls))

    async def test_prior_v1_journal_cannot_be_reused_as_v3(self):
        import shutil
        async with self.backend() as backend:
            await diagnostic.execute(backend=backend, revision='v3-test', artifact_root=self.root,
                                     inspection_only=True)
            old = run_directory(diagnostic.load_plan()['run_id'], self.root)
            shutil.copytree(old, self.directory)
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls, [])


    async def test_legacy_order_is_valid_but_order_mismatch_not_silently_repaired(self):
        self.output = json.dumps(SUPPORTED, sort_keys=True)
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 14)
            self.assertEqual(result['report']['response_order_summary']['matches'], 0)
            self.assertEqual(result['report']['response_order_summary']['mismatches'], 14)
            self.assertEqual(result['report']['predictions'][0]['response_field_order'][0], 'answer_quote')
            count = len(self.calls)
            self.output = json.dumps(SUPPORTED)
            cached = await self.execute(backend)
            self.assertEqual(cached['report']['response_order_summary']['mismatches'], 14)
            self.assertEqual(count, len(self.calls))

    async def test_incomplete_json_cannot_be_counted_as_order_match(self):
        self.output = '{"verdict":"supported",'
        async with self.backend() as backend:
            result = await self.execute(backend)
            self.assertEqual(result['report']['examples_valid'], 0)
            self.assertEqual(result['report']['response_order_summary']['unavailable'], 14)
            self.assertEqual(result['report']['response_order_summary']['matches'], 0)
            self.assertTrue(all(p['verdict'] is None for p in result['report']['predictions']))

    async def test_sorted_schema_with_identical_canonical_hash_rejected_before_calls(self):
        from post_thesis.llm_judge.prompts import canonical_json, content_hash
        original = diagnostic.EVIDENCE_SCHEMA_V3_JSON
        sorted_schema = canonical_json(json.loads(original))
        self.assertEqual(content_hash(json.loads(original)), content_hash(json.loads(sorted_schema)))
        self.assertNotEqual(original, sorted_schema)
        async with self.backend() as backend:
            with patch.object(diagnostic, 'EVIDENCE_SCHEMA_V3_JSON', sorted_schema):
                with self.assertRaises(RunConflict):
                    await self.execute(backend)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.directory.exists())


if __name__ == '__main__':
    unittest.main()
