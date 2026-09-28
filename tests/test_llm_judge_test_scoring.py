"""Artificial TEST-shaped requests, mock HTTP and durable replay; no model calls."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import audit_test_tokens as audit
from post_thesis.llm_judge import test_scoring as c
from post_thesis.llm_judge import run_test_scores as run
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.label_live import SOURCE_BLOBS
from post_thesis.llm_judge.label_diagnose import _load, _save
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example, run_directory
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import RunConflict, RunLocked, exclusive_run
from test_llm_judge_label_live import response
from test_llm_judge_label_pilot_audit import PinnedLabelToy
try:
    import httpx
except ImportError:
    httpx=None


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.examples=[Example(str(i),JudgeInput('A venue offers seats.','The venue offers seats.')) for i in range(3)]
        self.audited,_=audit.audit(self.examples,PinnedLabelToy(),directory=Path(self.temp.name)/'audit',
            revision='synthetic',files={'fake':'hash'},versions={'fake':'version'},reference_hash='fake')
        self.plan=c.plan_descriptor();summary=self.audited['audit']['summary']
        self.plan.update(request_count=3,max_attempts=3,max_output_tokens_total=3,
            audit_sha256=self.audited['audit_sha256'],audited_input_tokens=summary['total_input_tokens'],
            audited_min_input_tokens=summary['min_input_tokens'],audited_max_input_tokens=summary['max_input_tokens'])
        for target,name,value in ((c,'AUDIT_SHA256',self.audited['audit_sha256']),
                                  (c,'load_plan',lambda:self.plan),(run,'load_plan',lambda:self.plan)):
            p=patch.object(target,name,value);p.start();self.addCleanup(p.stop)
        self.requests=c.make_requests(self.examples,self.audited['audit'],PinnedLabelToy())
        self.preparation={'audited':self.audited,'comparison_manifest_sha256':c.COMPARISON_SHA256,
            'source':{'version':'0.29.0','git_blob_sha1':SOURCE_BLOBS},
            'versions':{'fake':'version'},'tokenizer_files':{'fake':'hash'},'tokenizer_reference_sha256':'fake'}

    def test_audit_and_label_free_payload_match(self):
        c.validate_preparation(self.preparation,self.requests,self.plan)
        for request in self.requests:
            for field in ('label','task','generator','sample_id','test_index','allowed_token_ids','logit_bias'):
                self.assertNotIn(field,request['payload'])
        changed=deepcopy(self.audited);changed['audit']['rows'][0]['prepared']['input_tokens']+=1
        with self.assertRaises(RunConflict): c.validate_audit(changed)

    def test_prepare_only_never_constructs_http_backend(self):
        with patch('sys.argv',['run_test_scores','--prepare-only']), \
             patch.object(run,'code_revision',return_value='synthetic'), \
             patch.object(run,'run_directory',return_value=Path(self.temp.name)/'empty'), \
             patch.object(run,'load_inputs',return_value=(self.examples,self.audited,[])), \
             patch.object(run,'prepare',return_value=(self.requests,self.preparation)), \
             patch.object(run,'LabelBackend',side_effect=AssertionError('HTTP backend created')):
            self.assertEqual(run.main(),0)

    def test_prompt_tokens_and_request_settings_cannot_change(self):
        for mutate in (lambda r:r['payload']['prompt'].append(1),
                       lambda r:r['payload'].update(temperature=1),
                       lambda r:r.update(orientation='swapped'),
                       lambda r:r['identity'].update(test_index=2)):
            requests=deepcopy(self.requests);mutate(requests[0])
            with self.assertRaises(RunConflict): c.validate_preparation(self.preparation,requests,self.plan)
        examples=[Example('0',JudgeInput('changed answer','same context')),*self.examples[1:]]
        with self.assertRaises(RunConflict): c.make_requests(examples,self.audited['audit'],PinnedLabelToy())


@unittest.skipIf(httpx is None,'HTTP mock dependency unavailable')
class ExecutionTests(PreparationTests,unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp();self.calls=[]
        self.server={'study_stage':'post_thesis','kind':'serving_session','status':'started',
            'code_revision':'synthetic','profile':self.plan['profile'],'gpu':{'name':'NVIDIA H200'},
            'command':server_command(profile_name='label-score-v1'),
            'package_versions':{'vllm':'0.29.0','torch':'2.13.0+cu130'},
            'label_score_source':self.preparation['source']}

    def handler(self,request):
        self.calls.append(request.url.path)
        if request.url.path=='/version': return httpx.Response(200,json={'version':'0.29.0'})
        if request.url.path=='/v1/models': return httpx.Response(200,json={'data':[
            {'id':self.plan['profile']['served_model_name'],'max_model_len':32768}]})
        if request.url.path=='/openapi.json': return httpx.Response(200,json={'components':{'schemas':{
            'CompletionRequest':{'properties':{k:{} for k in ('logprobs','logprob_token_ids','return_token_ids','return_tokens_as_token_ids')}}}}})
        self.assertEqual(request.url.path,'/v1/completions')
        return httpx.Response(200,json=response({'payload':json.loads(request.content)}))

    async def invoke(self,cap=3,handler=None,server=None):
        async with run.LabelBackend(transport=httpx.MockTransport(handler or self.handler)) as backend:
            return await run.execute(backend=backend,revision='synthetic',requests=self.requests,
                preparation=self.preparation,server_record=self.server if server is None else server,
                artifact_root=self.temp.name,max_new_attempts=cap)

    async def test_partial_resume_and_stable_zero_call_replay(self):
        first=await self.invoke(cap=1)
        self.assertEqual(first['report']['pending'],2)
        result=await self.invoke()
        self.assertEqual(result['new_attempts'],2)
        self.assertEqual(self.calls.count('/v1/completions'),3)
        self.calls.clear();replay=await self.invoke()
        self.assertEqual(self.calls,[])
        self.assertEqual(replay['new_attempts'],0)
        self.assertEqual(result['report_sha256'],replay['report_sha256'])
        inspected=run.inspect_cached('synthetic',self.temp.name)
        self.assertEqual(result['report_sha256'],inspected['report_sha256'])
        self.assertFalse(inspected['report']['TEST_metrics_computed'])

    async def test_malformed_response_halts_without_retry(self):
        def handler(req):
            result=self.handler(req)
            if req.url.path=='/v1/completions': return httpx.Response(200,json={'model':'bad','usage':{'prompt_tokens':12,'completion_tokens':1}})
            return result
        result=await self.invoke(handler=handler)
        self.assertEqual(result['report']['terminal_failures'],1)
        self.assertEqual(result['report']['pending'],2)
        self.calls.clear();await self.invoke()
        self.assertEqual(self.calls,[])

    async def test_interrupted_call_is_preserved_not_retried(self):
        def handler(req):
            if req.url.path=='/v1/completions': raise asyncio.CancelledError()
            return self.handler(req)
        result=await self.invoke(handler=handler)
        self.assertEqual(result['report']['terminal_failures'],1)
        self.assertEqual(result['report']['unknown_usage_attempts'],{'input_tokens':1,'output_tokens':1})
        self.calls.clear();await self.invoke()
        self.assertEqual(self.calls,[])

    async def test_unknown_window_exhausts_remaining_budget(self):
        await self.invoke(cap=1)
        path=run_directory(c.RUN_ID,self.temp.name)/'budget.json'
        ledger=_load(path);ledger['windows'][0]['status']='started';ledger['windows'][0]['elapsed_seconds']=None
        _save(path,ledger);self.calls.clear();result=await self.invoke()
        self.assertEqual(self.calls,[])
        self.assertEqual(result['report']['remaining_client_seconds'],0)
        self.assertEqual(result['report']['pending'],2)

    async def test_wrong_server_and_lock_block_before_http(self):
        with self.assertRaises(RunConflict): await self.invoke(server={'status':'stopped'})
        self.assertEqual(self.calls,[])
        with exclusive_run(run_directory(run.GLOBAL_LOCK_ID,self.temp.name)):
            with self.assertRaises(RunLocked): await self.invoke()
        self.assertEqual(self.calls,[])

    async def test_changed_cache_revision_or_missing_journal_rejected(self):
        await self.invoke()
        with self.assertRaises(RunConflict): run.inspect_cached('other',self.temp.name)
        (run_directory(c.RUN_ID,self.temp.name)/'journal.sqlite3').unlink()
        with self.assertRaises(RunConflict): run.inspect_cached('synthetic',self.temp.name)


class FrozenPlanTests(unittest.TestCase):
    def test_production_plan_stays_bounded_and_unmasked(self):
        plan=c.load_plan()
        self.assertEqual(plan['request_count'],2700)
        self.assertEqual(plan['audited_input_tokens'],3334607)
        self.assertEqual(plan['max_attempts'],2700)
        self.assertEqual(plan['client_budget_seconds'],7200)
        self.assertEqual(plan['max_output_tokens_per_request'],1)
        self.assertEqual(plan['retries'],0)
        self.assertFalse(plan['TEST_metrics'])


if __name__=='__main__': unittest.main()
