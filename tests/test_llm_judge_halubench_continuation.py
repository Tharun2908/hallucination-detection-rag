"""Status-only continuation, read-only parent replay, and mock transport checks."""
import asyncio
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import continue_halubench_scores as c
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import Journal, RunConflict
from post_thesis.llm_judge.runner import run_directory
import test_llm_judge_halubench_scoring as base


class ContractTests(unittest.TestCase):
    def test_registered_amendment_preserves_original_plan(self):
        plan=c.load_plan();parent=c.parent_contract.load_plan()
        self.assertEqual(parent['client_budget_seconds'],14400)
        self.assertEqual(plan['client_budget_seconds'],7200)
        self.assertEqual(plan['max_attempts'],348)
        self.assertEqual(plan['audited_input_tokens'],1589707)
        self.assertEqual(plan['profile'],parent['profile'])
        self.assertEqual(plan['retries'],0)
        for name,digest in c.IMPLEMENTATION_FILES.items():
            self.assertEqual(c.file_sha256(Path(c.__file__).with_name(name)),digest)

    def population(self):
        requests=[];predictions=[]
        for i in range(8000):
            requests.append({'key':str(i),'sample_id':str(i),'identity':{
                'test_index':i,'input_sha256':str(i)},'payload':{'prompt':[1]}})
            predictions.append({'request_key':str(i),'sample_id':str(i),'test_index':i,
                'input_sha256':str(i),'status':'ok' if i<7652 else 'interrupted' if i==7652 else 'pending'})
        return requests,{'predictions':predictions}

    def test_selects_only_missing_payloads_in_original_order(self):
        requests,report=self.population();before=deepcopy(requests)
        with patch.object(c,'INPUT_TOKENS',348):
            selected,reissued=c.select_remaining(requests,report)
        self.assertEqual(selected,requests[7652:])
        self.assertEqual(reissued,['7652'])
        self.assertEqual(requests,before)
        self.assertTrue(all(a is b for a,b in zip(selected,requests[7652:])))

    def test_unexpected_status_or_alignment_cannot_expand_selection(self):
        for field,value in [('status','pending'),('sample_id','different'),('test_index',8000)]:
            requests,report=self.population();report['predictions'][0][field]=value
            with patch.object(c,'INPUT_TOKENS',348),self.assertRaises(RunConflict):
                c.select_remaining(requests,report)
        requests,report=self.population()
        with self.assertRaises(RunConflict): c.select_remaining(requests,report)

    def test_journal_reader_does_not_write_or_recover(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'journal.sqlite3';identity={'fixture':True}
            journal=Journal(path,identity)
            first=journal.start('a',1);journal.finish(first,{'status':'ok'})
            journal.start('b',1);journal.recover();expected=journal.records();journal.close()
            before=path.read_bytes();path.chmod(0o444)
            self.assertEqual(c.read_records(path,identity),expected)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(RunConflict): c.read_records(path,{'fixture':False})
            path.chmod(0o644)
            with closing(sqlite3.connect(path)) as db:
                with db:
                    db.execute("UPDATE attempts SET state='started' WHERE request_key='b'")
            with self.assertRaises(sqlite3.ProgrammingError):
                db.execute('SELECT 1')  # A transaction context alone does not close the connection.
            before=path.read_bytes()
            with self.assertRaises(RunConflict): c.read_records(path,identity)
            self.assertEqual(path.read_bytes(),before)

    def test_cpu_prepare_creates_no_backend_or_model(self):
        prep={'reissued_interrupted_sample_ids':['fixture'],'selected_requests_sha256':'fixture'}
        with patch('sys.argv',['continuation','--prepare-only']), \
             patch.object(c,'code_revision',return_value='synthetic'), \
             patch.object(c,'verify_population'), \
             patch.object(c,'verify_parent',return_value=([{'payload':{'prompt':[1]}}],prep)), \
             patch.object(c,'LabelBackend',side_effect=AssertionError('HTTP instantiated')), \
             patch.object(c,'installed_source',side_effect=AssertionError('GPU environment inspected')):
            self.assertEqual(c.main(),0)

    def test_changed_parent_summary_blocked_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=run_directory(c.parent_contract.RUN_ID,temp);directory.mkdir(parents=True)
            path=directory/'summary.json';path.write_text(json.dumps({'report_sha256':'changed','report':{}}))
            before=path.read_bytes()
            with self.assertRaises(RunConflict): c.verify_parent(temp)
            self.assertEqual(path.read_bytes(),before)


@unittest.skipIf(base.httpx is None,'HTTP mock dependency unavailable')
class ExecutionTests(base.PreparationTests,unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp();self.calls=[]
        self.plan.update(run_id=c.RUN_ID,client_budget_seconds=7200)
        self.preparation={'source':self.preparation['source'],
            'parent_known_token_totals':{'input_tokens':6204778,'output_tokens':7652},
            'parent_unknown_usage_attempts':{'input_tokens':1,'output_tokens':1},
            'parent_charged_client_seconds':14405.70098266704}
        for name,value in [('load_plan',lambda:self.plan),('verify_parent',lambda:(self.requests,self.preparation))]:
            p=patch.object(c,name,value);p.start();self.addCleanup(p.stop)
        self.server={'study_stage':'post_thesis','kind':'serving_session','status':'started',
            'code_revision':'synthetic','profile':self.plan['profile'],'gpu':{'name':'NVIDIA H200'},
            'command':base.server_command(profile_name='label-score-v1'),
            'package_versions':{'vllm':'0.29.0','torch':'2.13.0+cu130'},
            'label_score_source':self.preparation['source']}

    handler=base.ExecutionTests.handler

    # Parent preparation tests target the full 8k audit format, not this compact child identity.
    def test_audit_and_label_free_payload_match(self):
        c.validate_preparation(self.preparation,self.requests,self.plan)
        self.assertTrue(all('label' not in r['payload'] for r in self.requests))

    def test_prepare_only_never_constructs_http_backend(self):
        changed=deepcopy(self.requests);changed[0]['payload']['temperature']=1
        with self.assertRaises(RunConflict): c.validate_preparation(self.preparation,changed,self.plan)

    def test_prompt_tokens_and_request_settings_cannot_change(self):
        changed=deepcopy(self.preparation);changed['parent_charged_client_seconds']=0
        with self.assertRaises(RunConflict): c.validate_preparation(changed,self.requests,self.plan)

    async def invoke(self,cap=3,handler=None):
        async with c.LabelBackend(transport=base.httpx.MockTransport(handler or self.handler)) as backend:
            return await c.execute(backend=backend,revision='synthetic',requests=self.requests,
                preparation=self.preparation,server_record=self.server,artifact_root=self.temp.name,
                max_new_attempts=cap)

    async def test_resume_preserves_unknown_parent_usage_and_stable_replay(self):
        first=await self.invoke(cap=1);self.assertEqual(first['report']['pending'],2)
        result=await self.invoke();r=result['report']
        self.assertEqual(result['new_attempts'],2)
        self.assertEqual(r['cumulative_unknown_usage_attempts'],{'input_tokens':1,'output_tokens':1})
        self.assertEqual(r['cumulative_attempts'],7656)
        self.assertEqual(r['combined_valid_scores'],7655)  # toy child has three slots
        self.calls.clear();replay=await self.invoke()
        self.assertEqual(self.calls,[])
        self.assertEqual(replay['report_sha256'],result['report_sha256'])
        with patch.object(c,'run_directory',side_effect=lambda name:run_directory(name,self.temp.name)):
            inspected=c.inspect_child('synthetic')
        self.assertEqual(inspected['report_sha256'],result['report_sha256'])

    async def test_interrupted_child_is_not_automatically_retried(self):
        def handler(req):
            if req.url.path=='/v1/completions': raise asyncio.CancelledError()
            return self.handler(req)
        result=await self.invoke(handler=handler)
        self.assertEqual(result['report']['terminal_failures'],1)
        self.assertEqual(result['report']['cumulative_unknown_usage_attempts'],{'input_tokens':2,'output_tokens':2})
        self.calls.clear();await self.invoke();self.assertEqual(self.calls,[])

    async def test_malformed_child_halts_without_retry(self):
        def handler(req):
            result=self.handler(req)
            if req.url.path=='/v1/completions': return base.httpx.Response(200,json={'model':'wrong'})
            return result
        result=await self.invoke(handler=handler)
        self.assertEqual(result['report']['terminal_failures'],1)
        self.calls.clear();await self.invoke();self.assertEqual(self.calls,[])


if __name__=='__main__': unittest.main()
