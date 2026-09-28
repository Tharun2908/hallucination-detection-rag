"""Artificial alignment, read-only replay and fixed-metric application tests."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import audit_test_tokens as audit
from post_thesis.llm_judge import evaluate_test as ev
from post_thesis.llm_judge import test_scoring as scoring
from post_thesis.llm_judge import evaluation_math as metrics
from post_thesis.llm_judge.check_evaluation_math import load_contract, synthetic_fixture
from post_thesis.llm_judge.check_frozen_fit import load_freeze
from post_thesis.llm_judge.label_diagnose import _save
from post_thesis.llm_judge.label_live import parse_response,known_usage,SOURCE_BLOBS
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.serve import server_command
from post_thesis.llm_judge.storage import Journal,RunConflict
from test_llm_judge_comparison_manifest import fixture as baseline_fixture
from test_llm_judge_label_live import response
from test_llm_judge_label_pilot_audit import PinnedLabelToy
try:
    import numpy
    import scipy
    import sklearn
except ImportError:
    NUMERIC=False
else:
    NUMERIC=True


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.targets,self.reports,*_=baseline_fixture()
        self.offline=[{**r,'label':i,'metadata':{'task_type':'QA','model':'synthetic'}}
                      for i,r in enumerate(self.targets)]
        self.judge={'predictions':[{**r,'status':'ok','score':{'unsupported_log_odds':float(i)}}
                                  for i,r in enumerate(self.targets)]}
        self.comparison={'targets':self.targets,'baselines':{name:{'comparison_ready':True,
            **{k:True for k in metrics.PROVENANCE_FIELDS}} for name in metrics.MAIN_SYSTEMS if name!='judge'}}

    def test_join_preserves_label_score_direction_groups_and_all_systems(self):
        joined=ev.join_evaluation_rows(self.offline,self.judge,self.reports,self.comparison)
        self.assertEqual(joined['labels'],[0,1])
        self.assertEqual(joined['scores']['judge'],[0.,1.])
        self.assertEqual(joined['scores']['MiniCheck_7B'],[.3,.3]) # remain support until frozen math converts
        self.assertEqual(joined['groups'],['group','group'])
        self.assertEqual(set(joined['scores']),set(metrics.MAIN_SYSTEMS))
        self.assertEqual(joined['axes'],{'task':['QA','QA'],'generator':['synthetic','synthetic']})

    def test_wrong_input_order_label_or_group_rejected(self):
        for kind in ('order','input','label','group','nonfinite'):
            offline=deepcopy(self.offline);judge=deepcopy(self.judge)
            if kind=='order': judge['predictions'].reverse()
            if kind=='input': judge['predictions'][0]['input_sha256']='different'
            if kind=='label': offline[0]['label']=.0
            if kind=='group': offline[0]['component_sha256']='other'
            if kind=='nonfinite': judge['predictions'][0]['score']['unsupported_log_odds']=float('nan')
            with self.subTest(kind=kind),self.assertRaises(RunConflict):
                ev.join_evaluation_rows(offline,judge,self.reports,self.comparison)

    def test_result_replay_skips_calculation_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.json';calls=[]
            def compute(inputs): calls.append(inputs);return {'synthetic':True}
            first,new=ev.evaluate_once(path,{'pinned':'identity'},{'data':'synthetic'},{},[],compute)
            self.assertTrue(new)
            second,new=ev.evaluate_once(path,{'pinned':'identity'},{'data':'synthetic'},{},[],compute)
            self.assertFalse(new);self.assertEqual(first,second);self.assertEqual(len(calls),1)
            with self.assertRaises(RunConflict): ev.evaluate_once(path,{'pinned':'changed'},{},{},[],compute)
            altered=deepcopy(first);altered['report']['results']['synthetic']=False
            path.write_text(json.dumps(altered))
            with self.assertRaises(RunConflict): ev.evaluate_once(path,{'pinned':'identity'},{},{},[],compute)


class ReadOnlyReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        examples=[Example(str(i),JudgeInput('The venue offers seats.','The venue offers seats.')) for i in range(2)]
        audited,_=audit.audit(examples,PinnedLabelToy(),directory=self.path/'audit',revision='synthetic',
                            files={'fake':'hash'},versions={'fake':'version'},reference_hash='fake')
        plan=scoring.plan_descriptor();summary=audited['audit']['summary']
        plan.update(request_count=2,max_attempts=2,max_output_tokens_total=2,audit_sha256=audited['audit_sha256'],
                    audited_input_tokens=summary['total_input_tokens'])
        for target,name,value in ((scoring,'AUDIT_SHA256',audited['audit_sha256']),
                                  (scoring,'load_plan',lambda:plan),(ev,'load_plan',lambda:plan),
                                  (ev,'SCORING_REVISION','synthetic'),(ev,'PLAN_SHA256',content_hash(plan))):
            p=patch.object(target,name,value);p.start();self.addCleanup(p.stop)
        requests=scoring.make_requests(examples,audited['audit'],PinnedLabelToy())
        prep={'audited':audited,'comparison_manifest_sha256':scoring.COMPARISON_SHA256,
            'source':{'version':'0.29.0','git_blob_sha1':SOURCE_BLOBS},'versions':{'fake':'version'},
            'tokenizer_files':{'fake':'hash'},'tokenizer_reference_sha256':'fake'}
        identity={'plan':plan,'code_revision':'synthetic','preparation':prep,'requests_sha256':content_hash(requests)}
        server={'study_stage':'post_thesis','kind':'serving_session','status':'started','code_revision':'synthetic',
            'profile':plan['profile'],'gpu':{'name':'NVIDIA H200'},'command':server_command(profile_name='label-score-v1'),
            'package_versions':{'vllm':'0.29.0','torch':'2.13.0+cu130'},'label_score_source':prep['source']}
        ledger={'identity':identity,'windows':[{'id':'synthetic-window','status':'finished','reserved_seconds':7200,
                'elapsed_seconds':2.,'server_session_snapshot':server}],'halt_reason':None,'preparation_windows':[]}
        _save(self.path/'budget.json',ledger);_save(self.path/'prepared.json',requests)
        journal=Journal(self.path/'journal.sqlite3',identity)
        try:
            for request in requests:
                raw=response(request)
                journal.finish(journal.start(request['key'],1),{'status':'ok','score':parse_response(raw,request),
                    'response':raw,'usage':known_usage(raw),'error':None,'latency_seconds':1.})
            bundle=ev.replay_summary(requests,journal.records(),ledger)
        finally: journal.close()
        (self.path/'summary.json').write_text(json.dumps(bundle))
        for name,value in (('REQUESTS_SHA256',content_hash(requests)),('JUDGE_REPORT_SHA256',bundle['report_sha256'])):
            p=patch.object(ev,name,value);p.start();self.addCleanup(p.stop)
        self.targets=[{'sample_id':ex.sample_id,'test_index':i,'input_sha256':content_hash(asdict(ex.item))}
                      for i,ex in enumerate(examples)]

    def test_response_replay_does_not_rewrite_sources(self):
        paths=[self.path/name for name in ('budget.json','prepared.json','summary.json','journal.sqlite3')]
        before={p.name:p.read_bytes() for p in paths}
        report,provenance=ev.verify_judge(self.path,self.targets)
        self.assertEqual(report['valid_scores'],2)
        self.assertEqual(provenance['raw_response_replay'],'verified_read_only')
        self.assertEqual(before,{p.name:p.read_bytes() for p in paths})

    def test_report_tamper_and_wrong_input_rejected(self):
        wrong=deepcopy(self.targets);wrong[0]['input_sha256']='wrong'
        with self.assertRaises(RunConflict): ev.verify_judge(self.path,wrong)
        path=self.path/'summary.json';saved=json.loads(path.read_text());saved['report']['valid_scores']=1
        path.write_text(json.dumps(saved))
        with self.assertRaises(RunConflict): ev.verify_judge(self.path,self.targets)


@unittest.skipUnless(NUMERIC,'pinned numerical packages unavailable')
class NumericalApplicationTests(unittest.TestCase):
    def test_frozen_math_applied_with_registered_draws_and_all_slices(self):
        load_contract() # Includes byte checks of both immutable numerical source files.
        fixture=synthetic_fixture();fixture['axes']={'task':['a','b']*6,'generator':['synthetic']*12}
        fixture['provenance']={name:{'comparison_ready':True,**{k:True for k in metrics.PROVENANCE_FIELDS}}
                               for name in metrics.MAIN_SYSTEMS}
        result=ev.compute(fixture);paired=result['paired']
        self.assertEqual(paired['shared_n'],12)
        self.assertEqual(paired['bootstrap']['replicates'],2000)
        self.assertEqual(set(result['descriptive_slices']['judge']['task']),{'a','b'})
        self.assertFalse(paired['bootstrap']['refitting'])
        self.assertEqual(paired['individual']['judge']['rule']['threshold'],-1.25)
        self.assertEqual(set(paired['shared_points']['judge']['reliability']),{'raw','calibrated'})
        raw=metrics.system_arrays('judge',[0.])
        from scipy.special import expit
        self.assertAlmostEqual(raw['probabilities']['calibrated'][0],expit(load_freeze()['calibration']['parameters']['b']))
        for baseline in ('S4','MiniCheck_7B','S2_S4_metadata_free'):
            self.assertEqual(paired['paired_differences'][baseline]['auroc']['valid_draws'],2000)
            self.assertEqual(paired['paired_differences'][baseline]['ece_calibrated']['baseline_metric'],'ece_raw')
        fixture['provenance']['S4']['comparison_ready']=False
        with self.assertRaises(RunConflict): ev.compute(fixture)


if __name__=='__main__': unittest.main()
