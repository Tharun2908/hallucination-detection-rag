"""Paired execution over mocked HTTP, including budget and crash recovery."""

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

from post_thesis.llm_judge import evidence_pair as pair
from post_thesis.llm_judge.evidence_diagnose import native_observation
from post_thesis.llm_judge.evidence_prompt_v2_cases import examples
from post_thesis.llm_judge.evidence_schema_v3 import EVIDENCE_SCHEMA_V3_JSON, EVIDENCE_V3_FIELD_ORDER
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge.vllm_backend import VLLMBackend, load_profile

PROFILE = load_profile('evidence-v1')
SUPPORTED = dict(verdict='supported', issue_type='none', answer_quote=None, context_quote=None, explanation=None)


class PlanTests(unittest.TestCase):
    def test_plan_fixes_both_arms_and_balances_order(self):
        p = pair.load_plan()
        self.assertEqual(p, pair.expected_plan())
        self.assertEqual(p['max_generation_attempts'], 52)
        self.assertEqual(p['client_budget_seconds'], 600)
        self.assertEqual(p['max_output_tokens_total'], 52 * 512)
        self.assertEqual(p['max_tokenization_requests'], 104)
        order = pair.schedule()
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(order[:4], [(0,'v1'), (0,'v2'), (1,'v2'), (1,'v1')])
        self.assertEqual(sum(order[i][1]=='v1' for i in range(0,52,2)), 13)


@unittest.skipIf(httpx is None, 'Requires isolated HTTP client')
class PairedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = run_directory(pair.load_plan()['run_id'], self.root)
        self.calls = []
        self.tokens = 0
        self.overlength_at = None
        self.count_mismatch_at = None
        self.output = json.dumps(SUPPORTED)
        self.http_status = 200
        self.output_tokens = 40
        self.slow = False
        self.started = asyncio.Event()
        self.server = {'study_stage':'post_thesis', 'kind':'serving_session', 'status':'started',
            'code_revision':'pair-test', 'profile':PROFILE, 'gpu':{'name':'NVIDIA H200'},
            'command':server_command(profile_name='evidence-v1'),
            'package_versions':native_observation('v3')['versions']}

    def backend(self):
        async def handle(request):
            self.calls.append(request)
            path = request.url.path
            if path == '/version':
                return httpx.Response(200,json={'version':PROFILE['vllm_version']})
            if path == '/v1/models':
                return httpx.Response(200,json={'data':[{'id':PROFILE['served_model_name'], 'max_model_len':PROFILE['max_model_len']}]})
            if path == '/tokenize':
                self.tokens += 1
                count = 4097 if self.tokens == self.overlength_at else 901 if self.tokens == self.count_mismatch_at else 900
                return httpx.Response(200,json={'count':count, 'max_model_len':PROFILE['max_model_len']})
            self.assertEqual(path, '/v1/chat/completions')
            self.started.set()
            if self.slow:
                await asyncio.sleep(30)
            return httpx.Response(self.http_status,json={'id':'mock', 'model':PROFILE['served_model_name'],
                'usage':{'prompt_tokens':900,'completion_tokens':self.output_tokens},
                'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':self.output}}]})
        return VLLMBackend(profile_name='evidence-v1',transport=httpx.MockTransport(handle))

    async def execute(self,backend,**kwargs):
        return await pair.execute(backend=backend,revision='pair-test',server_record=self.server,artifact_root=self.root,**kwargs)

    def arm_dir(self,arm):
        return run_directory(pair.load_plan()['run_id'] + '-prompt-' + arm,self.root)

    def state(self):
        path = self.directory / 'execution' / 'budget.json'
        return path, json.loads(path.read_text())['execution']

    async def test_all_52_audited_before_generation_and_exact_balanced_requests(self):
        async with self.backend() as backend:
            result = await self.execute(backend,clock=iter([0,10]).__next__)
            self.assertEqual(result['new_attempts_this_invocation'],52)
            self.assertEqual(result['charged_client_seconds'],10)
            self.assertEqual(self.tokens,104)
            first_gen = next(i for i,r in enumerate(self.calls) if r.url.path=='/v1/chat/completions')
            self.assertEqual(sum(r.url.path=='/tokenize' for r in self.calls[:first_gen]),53)
            wires = [r for r in self.calls if r.url.path=='/v1/chat/completions']
            self.assertEqual(len(wires),52)
            cases = examples()
            for (index,arm), wire in zip(pair.schedule(),wires,strict=True):
                payload = json.loads(wire.content)
                req = pair.FACTORIES[arm](cases[index].item,backend.config)
                self.assertEqual(payload['messages'],[asdict(m) for m in req.messages])
                self.assertEqual(set(json.loads(payload['messages'][1]['content'])),{'answer','context'})
                self.assertEqual(payload['max_completion_tokens'],512)
                self.assertEqual(payload['response_format']['json_schema']['schema'],json.loads(EVIDENCE_SCHEMA_V3_JSON))
                for branch in payload['response_format']['json_schema']['schema']['anyOf']:
                    self.assertEqual(tuple(branch['properties']),EVIDENCE_V3_FIELD_ORDER)
            for arm in pair.ARMS:
                self.assertEqual(result['reports'][arm]['examples_valid'],26)
                self.assertEqual(result['reports'][arm]['response_order_summary']['matches'],26)
                self.assertLess(result['arm_summary'][arm]['verdict_matches'],26) # Mock validity is not accuracy.
            before=len(self.calls)
            self.server['status']='stopped'
            cached=await self.execute(backend)
            self.assertEqual(cached['new_attempts_this_invocation'],0)
            self.assertEqual(cached['charged_client_seconds'],10)
            self.assertEqual(before,len(self.calls))

    async def test_final_audit_failure_blocks_both_arms_and_replay(self):
        self.overlength_at=52
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertEqual(result['error_code'],'diagnostic_input_limit')
            self.assertEqual(result['new_attempts_this_invocation'],0)
            self.assertFalse(any(r.url.path=='/v1/chat/completions' for r in self.calls))
            before=len(self.calls)
            await self.execute(backend)
            self.assertEqual(before,len(self.calls))

    async def test_critical_error_in_first_arm_blocks_second_arm(self):
        self.http_status=400
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertEqual(result['error_code'],'http_400')
            self.assertEqual(result['new_attempts_this_invocation'],1)
            self.assertEqual(result['reports']['v2']['attempts_total'],0)
            before=len(self.calls)
            self.http_status=200
            await self.execute(backend)
            self.assertEqual(before,len(self.calls))

    async def test_retokenization_mismatch_blocks_generation(self):
        self.count_mismatch_at=53
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertEqual(result['error_code'],'audited_input_token_mismatch')
            self.assertEqual(result['new_attempts_this_invocation'],1)
            self.assertFalse(any(r.url.path=='/v1/chat/completions' for r in self.calls))

    async def test_output_cap_violation_retains_usage_and_halts_both_arms(self):
        self.output_tokens=513
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertEqual(result['error_code'],'output_token_limit_exceeded')
            self.assertEqual(result['reports']['v1']['known_token_totals']['output_tokens'],513)
            self.assertEqual(result['reports']['v2']['attempts_total'],0)

    async def test_invalid_evidence_stays_missing_without_salvage(self):
        self.output=json.dumps({**SUPPORTED,'explanation':'This is supported.'})
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertEqual(result['new_attempts_this_invocation'],52)
            for arm in pair.ARMS:
                self.assertEqual(result['reports'][arm]['examples_valid'],0)
                self.assertEqual(result['reports'][arm]['response_order_summary']['matches'],26)
                self.assertTrue(all(p['verdict'] is None for p in result['reports'][arm]['predictions']))
                self.assertTrue(all(p[arm]['verdict_matches_expected'] is None for p in result['comparisons']))

    async def test_order_is_separate_from_validity(self):
        self.output=json.dumps(SUPPORTED,sort_keys=True)
        async with self.backend() as backend:
            result=await self.execute(backend)
            self.assertTrue(all(r['examples_valid']==26 and r['response_order_summary']['mismatches']==26
                                for r in result['reports'].values()))

    async def test_inspection_before_first_run_makes_no_http_calls(self):
        async with self.backend() as backend:
            result=await self.execute(backend,inspection_only=True)
            self.assertEqual(self.calls,[])
            self.assertEqual(result['execution_status'],'ready')
            self.assertEqual(result['charged_client_seconds'],0)

    async def test_unknown_window_charges_full_budget_without_calls(self):
        async with self.backend() as backend:
            await self.execute(backend,inspection_only=True)
            path,state=self.state()
            state.update(status='started',reserved_seconds=600)
            pair._save(path,state)
            result=await self.execute(backend)
            self.assertEqual(result['charged_client_seconds'],600)
            self.assertFalse(result['generation_invocation_available'])
            self.assertEqual(self.calls,[])

    async def test_timeout_recovers_attempt_and_never_restarts(self):
        original=asyncio.wait_for
        async def deadline(operation,timeout):
            if timeout!=600:
                return await original(operation,timeout=timeout)
            task=asyncio.create_task(operation)
            try:
                await original(self.started.wait(),timeout=10)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise asyncio.TimeoutError
        self.slow=True
        async with self.backend() as backend:
            with patch.object(pair.asyncio,'wait_for',deadline):
                result=await self.execute(backend)
            self.assertEqual(result['outcome'],'deadline_reached')
            self.assertEqual(result['reports']['v1']['examples_terminal_failure'],1)
            self.assertEqual(result['reports']['v1']['attempts_with_unknown_tokens']['output_tokens'],1)
            self.assertEqual(result['reports']['v2']['attempts_total'],0)
            before=len(self.calls)
            await self.execute(backend)
            self.assertEqual(before,len(self.calls))

    async def test_missing_budget_or_either_arm_journal_cannot_reset(self):
        self.http_status=400
        async with self.backend() as backend:
            await self.execute(backend)
            before=len(self.calls)
            path,_=self.state(); saved=path.read_bytes(); path.unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            path.write_bytes(saved)
            (self.arm_dir('v2')/'journal.sqlite3').unlink()
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(before,len(self.calls))

    async def test_changed_plan_or_environment_rejected_before_network(self):
        async with self.backend() as backend:
            for field,value in [('max_generation_attempts',53),('client_budget_seconds',1200),
                                ('serialized_schema_sha256','other'),('arm_order','other')]:
                plan=pair.load_plan();plan[field]=value
                with self.subTest(field=field),patch.object(pair,'load_plan',return_value=plan):
                    with self.assertRaises(RunConflict):
                        await self.execute(backend)
            self.server['package_versions']['xgrammar']='other'
            with self.assertRaises(RunConflict):
                await self.execute(backend)
            self.assertEqual(self.calls,[])

    async def test_changed_revision_refuses_cached_identity(self):
        async with self.backend() as backend:
            await self.execute(backend,inspection_only=True)
            with self.assertRaises(RunConflict):
                await pair.execute(backend=backend,revision='different',artifact_root=self.root,inspection_only=True)
            self.assertEqual(self.calls,[])


if __name__=='__main__':
    unittest.main()
