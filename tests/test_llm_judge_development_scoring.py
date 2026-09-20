"""Reserved-arm scoring bounds, provenance, input isolation and recovery."""
import asyncio
from copy import deepcopy
import json
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import development_scoring as c
from post_thesis.llm_judge import run_development_scores as run
from post_thesis.llm_judge.label_live import completion_payload
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import run_directory, Example
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import Journal, RunConflict, RunLocked, exclusive_run
from post_thesis.llm_judge.vllm_backend import load_profile
from post_thesis.llm_judge.label_score_contract import prepare_tokenized_input
from test_llm_judge_label_live import response
from test_llm_judge_label_pilot_audit import PinnedLabelToy
try:
    import httpx
except ImportError:
    httpx = None


def requests(arm):
    result = []
    for i in range(6):
        rid = str(i + (100 if arm == 'operating_threshold' else 0))
        slot = arm + ':' + rid
        payload = completion_payload([11, 12, 13], load_profile(c.PROFILE))
        mapping = {'supported': 32, 'unsupported': 33}
        identity = {'slot': slot, 'arm': arm, 'payload': payload, 'class_mapping': mapping}
        result.append({'slot': slot, 'sample_id': rid, 'orientation': 'primary',
                       'class_mapping': mapping, 'payload': payload, 'identity': identity,
                       'key': content_hash(identity)})
    return result


@unittest.skipIf(httpx is None, 'isolated HTTP client not installed')
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rows = {arm: requests(arm) for arm in c.ARMS}
        self.plans = {arm: c.plan_descriptor(arm, c.request_references(rows)) for arm, rows in self.rows.items()}
        for plan in self.plans.values():
            plan.update(request_count=6, audited_input_tokens=18, audited_min_input_tokens=3, audited_max_input_tokens=3)
        patcher = patch.object(run, 'load_plan', side_effect=lambda arm: self.plans[arm])
        patcher.start(); self.addCleanup(patcher.stop)
        self.preparation = {'audit_sha256': c.AUDIT_SHA256,
                            'source': {'version': '0.29.0', 'git_blob_sha1': c.SOURCE_BLOBS}}
        self.server = {'study_stage': 'post_thesis', 'kind': 'serving_session', 'status': 'started',
                       'code_revision': 'test-revision', 'profile': load_profile(c.PROFILE),
                       'gpu': {'name': 'NVIDIA H200'}, 'command': server_command(profile_name=c.PROFILE),
                       'package_versions': {'vllm': '0.29.0', 'torch': '2.13.0+cu130'},
                       'label_score_source': self.preparation['source']}
        self.calls = []

    def handler(self, req):
        self.calls.append(req.url.path)
        if req.url.path == '/version': return httpx.Response(200, json={'version': '0.29.0'})
        if req.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': self.server['profile']['served_model_name'], 'max_model_len': 32768}]})
        if req.url.path == '/openapi.json':
            return httpx.Response(200, json={'components': {'schemas': {'CompletionRequest': {'properties': {
                k: {} for k in ('logprobs', 'logprob_token_ids', 'return_token_ids', 'return_tokens_as_token_ids')}}}}})
        self.assertEqual(req.url.path, '/v1/completions')
        payload = json.loads(req.content)
        for key in ('label', 'arm', 'sample_id', 'task', 'generator', 'allowed_token_ids', 'logit_bias'):
            self.assertNotIn(key, payload)
        return httpx.Response(200, json=response({'payload': payload}))

    async def run_arm(self, arm='calibration', *, cap=600, handler=None, server=None, prep=None):
        async with run.LabelBackend(transport=httpx.MockTransport(handler or self.handler)) as backend:
            return await run.execute(arm=arm, backend=backend, revision='test-revision', requests=self.rows[arm],
                preparation=prep or self.preparation, server_record=server or self.server,
                artifact_root=self.temp.name, max_new_attempts=cap)

    async def test_two_arms_are_separate_and_replay_and_inspect_make_no_http(self):
        for arm in c.ARMS:
            first = await self.run_arm(arm)
            self.assertTrue(first['execution_complete'])
            self.assertEqual(first['valid_scores'], 6)
            self.assertEqual(first['known_token_totals'], {'input_tokens': 18, 'output_tokens': 6})
            self.assertFalse(first['threshold_fitting']); self.assertFalse(first['calibration_fitting'])
            before = len(self.calls)
            second = await self.run_arm(arm)
            inspected = run.inspect_cached('test-revision', arm, self.temp.name)
            self.assertEqual(second['new_attempts'], 0)
            self.assertEqual(inspected['valid_scores'], 6)
            self.assertEqual(before, len(self.calls))
        self.assertEqual(self.calls.count('/v1/completions'), 12)
        self.assertNotEqual(self.plans[c.ARMS[0]]['run_id'], self.plans[c.ARMS[1]]['run_id'])

    async def test_partial_resume_uses_only_remaining_slots(self):
        first = await self.run_arm(cap=2)
        self.assertEqual(first['pending'], 4)
        second = await self.run_arm()
        self.assertEqual(second['new_attempts'], 4)
        self.assertEqual(self.calls.count('/v1/completions'), 6)
        self.assertEqual(len(second['execution_windows']), 2)

    async def test_shared_execution_lock_blocks_a_second_arm_before_http(self):
        with exclusive_run(run_directory(run.GLOBAL_LOCK_ID, self.temp.name)):
            with self.assertRaises(RunLocked): await self.run_arm('operating_threshold')
        self.assertFalse(self.calls)

    async def test_malformed_response_halts_without_retry_and_keeps_usage(self):
        def bad(req):
            result = self.handler(req)
            if req.url.path == '/v1/completions':
                value = result.json(); del value['choices'][0]['logprobs']['top_logprobs'][0]['token_id:33']
                return httpx.Response(200, json=value)
            return result
        result = await self.run_arm(handler=bad)
        self.assertEqual(result['terminal_failures'], 1); self.assertEqual(result['pending'], 5)
        self.assertEqual(result['known_token_totals']['output_tokens'], 1)
        self.assertIsNone(result['predictions'][0]['score'])
        await self.run_arm()
        self.assertEqual(self.calls.count('/v1/completions'), 1)

    async def test_http_error_retains_unknown_usage_and_does_not_repeat(self):
        def bad(req):
            if req.url.path == '/v1/completions':
                self.calls.append(req.url.path); return httpx.Response(500, text='failure')
            return self.handler(req)
        result = await self.run_arm(handler=bad)
        self.assertEqual(result['unknown_usage_attempts'], {'input_tokens': 1, 'output_tokens': 1})
        self.assertEqual(result['halt_reason'], 'http_500')
        await self.run_arm()
        self.assertEqual(self.calls.count('/v1/completions'), 1)

    async def test_wrong_server_or_changed_preparation_fails_before_http(self):
        bad = deepcopy(self.server); bad['code_revision'] = 'older-commit'
        with self.assertRaises(RunConflict): await self.run_arm(server=bad)
        self.assertFalse(self.calls)
        await self.run_arm(cap=1)
        before = len(self.calls)
        with self.assertRaises(RunConflict): await self.run_arm(prep={**self.preparation, 'changed': True})
        self.assertEqual(len(self.calls), before)

    async def test_failed_preflight_sends_no_scoring_request(self):
        def bad(req):
            self.calls.append(req.url.path)
            return httpx.Response(200, json={'version': 'different'})
        result = await self.run_arm(handler=bad)
        self.assertEqual(result['new_attempts'], 0)
        self.assertEqual(result['halt_reason'], 'server_version_mismatch')
        before = len(self.calls); await self.run_arm()
        self.assertEqual(len(self.calls), before)
        self.assertNotIn('/v1/completions', self.calls)

    async def test_unknown_window_exhausts_one_arm_without_spending_other_budget(self):
        await self.run_arm(cap=1)
        directory = run_directory(self.plans['calibration']['run_id'], self.temp.name)
        ledger = run._load(directory / 'budget.json')
        ledger['windows'][0].update(status='started', elapsed_seconds=None)
        run._save(directory / 'budget.json', ledger)
        before = len(self.calls); report = await self.run_arm()
        self.assertEqual(report['remaining_client_seconds'], 0)
        self.assertEqual(report['charged_client_seconds'], 1800)
        self.assertEqual(len(self.calls), before)
        other = await self.run_arm('operating_threshold')
        self.assertEqual(other['valid_scores'], 6)
        self.assertGreater(other['remaining_client_seconds'], 0)

    async def test_interrupted_attempt_remains_missing_no_repeat(self):
        await self.run_arm(cap=1)
        directory = run_directory(self.plans['calibration']['run_id'], self.temp.name)
        ledger = run._load(directory / 'budget.json')
        journal = Journal(directory / 'journal.sqlite3', ledger['identity'])
        try: journal.start(self.rows['calibration'][1]['key'], 1)
        finally: journal.close()
        report = await self.run_arm()
        self.assertEqual(report['halt_reason'], 'interrupted_request')
        self.assertEqual(report['predictions'][1]['status'], 'interrupted')
        self.assertIsNone(report['predictions'][1]['score'])
        self.assertEqual(self.calls.count('/v1/completions'), 1)

    async def test_cancelled_request_is_not_retried(self):
        async def cancelled(req):
            if req.url.path == '/v1/completions':
                self.calls.append(req.url.path); raise asyncio.CancelledError()
            return self.handler(req)
        result = await self.run_arm(handler=cancelled)
        self.assertEqual(result['terminal_failures'], 1)
        self.assertEqual(result['unknown_usage_attempts']['output_tokens'], 1)
        await self.run_arm()
        self.assertEqual(self.calls.count('/v1/completions'), 1)

    async def test_cross_arm_payload_drift_cap_and_missing_journal_rejected(self):
        with self.assertRaises(ValueError): await self.run_arm(cap=601)
        self.rows['calibration'][0]['payload']['prompt'] = [91, 92, 93]
        with self.assertRaises(RunConflict): await self.run_arm()
        self.assertFalse(self.calls)
        self.rows['calibration'] = requests('operating_threshold')
        with self.assertRaises(RunConflict): await self.run_arm()
        self.rows['calibration'] = requests('calibration')
        await self.run_arm(cap=1)
        directory = run_directory(self.plans['calibration']['run_id'], self.temp.name)
        (directory / 'journal.sqlite3').unlink()
        with self.assertRaises(RunConflict): await self.run_arm()


class PreparationTests(unittest.TestCase):
    def test_frozen_arm_plans_bind_audit_and_separate_budgets(self):
        plans = [c.load_plan(arm) for arm in c.ARMS]
        self.assertEqual(sum(p['audited_input_tokens'] for p in plans), 1451599)
        self.assertEqual(sum(p['max_output_tokens_total'] for p in plans), 1200)
        for plan in plans:
            self.assertEqual(plan['audit_sha256'], c.AUDIT_SHA256)
            self.assertEqual(plan['client_budget_seconds'], 1800)
            self.assertEqual(plan['attempts_per_slot'], 1)
        with patch.object(c, 'PLAN_SHA256', 'changed'), self.assertRaises(RunConflict):
            c.load_plan('calibration')

    def test_rendering_must_match_audit_and_payload_excludes_metadata(self):
        tokenizer = PinnedLabelToy()
        examples = [Example(str(i), JudgeInput('A venue.', 'A venue exists.')) for i in range(2)]
        state = {'rows': [{'partition': 'calibration', 'sample_id': ex.sample_id,
                          'prepared': {k: v for k, v in prepare_tokenized_input(ex.item, tokenizer).items()
                                       if k != 'rendered_prompt'}} for ex in examples]}
        rows = c.prepare_requests(examples, state, tokenizer, 'calibration')
        for row in rows:
            self.assertEqual(row['payload']['max_tokens'], 1)
            for key in ('label', 'arm', 'sample_id', 'generator', 'task', 'allowed_token_ids', 'logit_bias'):
                self.assertNotIn(key, row['payload'])
        bad = deepcopy(state); bad['rows'][0]['prepared']['input_tokens'] += 1
        with self.assertRaises(RunConflict): c.prepare_requests(examples, bad, tokenizer, 'calibration')
        with self.assertRaises(RunConflict): c.prepare_requests(examples, state, tokenizer, 'operating_threshold')

    def test_audit_checksum_and_source_provenance_guard(self):
        with patch.object(c, 'validate_reservation', return_value={}):
            with self.assertRaisesRegex(RunConflict, 'checksum'):
                c.validate_inputs({}, {'audit': {}, 'audit_sha256': 'wrong'})
        preparation = {'audit_sha256': c.AUDIT_SHA256, 'source': {'version': 'different'}}
        with self.assertRaises(RunConflict): c.validate_preparation(preparation, requests('calibration'), c.load_plan('calibration'))


if __name__ == '__main__': unittest.main()
