"""Artificial-data tests for the frozen fitter and read-only journal replay."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import fitting_inputs as inputs
from post_thesis.llm_judge import fitting_math as fm
from post_thesis.llm_judge import fit_development as fit
from post_thesis.llm_judge.audit_development_tokens import load_fit_protocol
from post_thesis.llm_judge.storage import Journal, RunConflict, atomic_json
from post_thesis.llm_judge.label_diagnose import _save
from post_thesis.llm_judge.label_live import parse_response, known_usage
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import run_directory
from post_thesis.llm_judge import development_scoring as scoring
from post_thesis.llm_judge.run_development_scores import summarize
from post_thesis.llm_judge.serve import server_command
from test_llm_judge_development_scoring import requests
from test_llm_judge_label_live import response
try:
    import numpy as np
    import scipy
except ImportError:
    np = None


class IntegrityTests(unittest.TestCase):
    def test_label_join_is_by_id_and_rejects_missing_duplicate_boolean(self):
        self.assertEqual(inputs.join_labels(['b','a'], [{'sample_id':'a','label':0}, {'sample_id':'b','label':1}]), [1,0])
        for rows in ([{'sample_id':'a','label':0}],
                     [{'sample_id':'a','label':0},{'sample_id':'a','label':1}],
                     [{'sample_id':'a','label':False},{'sample_id':'b','label':1}]):
            with self.assertRaises(RunConflict): inputs.join_labels(['a','b'], rows)

    def test_environment_drift_rejected(self):
        with patch.object(fit, 'version', return_value='wrong'):
            with self.assertRaises(RunConflict): fit.software_versions(load_fit_protocol())

    def test_readonly_replay_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = 'calibration'; reqs = requests(arm)
            expected = {r['sample_id']: content_hash(r['sample_id']) for r in reqs}
            for r in reqs:
                r['identity']['prepared'] = {'input_sha256': expected[r['sample_id']]}
                r['key'] = content_hash(r['identity'])
            plan = scoring.plan_descriptor(arm, scoring.request_references(reqs))
            plan.update(request_count=6, audited_input_tokens=18)
            prep = {'audit_sha256': scoring.AUDIT_SHA256, 'source': {'version':'0.29.0','git_blob_sha1':scoring.SOURCE_BLOBS}}
            identity = {'plan':plan, 'code_revision':inputs.SCORING_REVISION, 'preparation':prep, 'requests_sha256':content_hash(reqs)}
            server = {'study_stage':'post_thesis','kind':'serving_session','status':'started',
                      'code_revision':inputs.SCORING_REVISION,'profile':plan['profile'],'gpu':{'name':'NVIDIA H200'},
                      'command':server_command(profile_name=scoring.PROFILE),
                      'package_versions':{'vllm':'0.29.0','torch':'2.13.0+cu130'}, 'label_score_source':prep['source']}
            ledger = {'identity':identity, 'halt_reason':None, 'windows':[{'id':'synthetic-window','status':'finished','elapsed_seconds':1,
                      'reserved_seconds':1800,'server_session_snapshot':server}]}
            directory = run_directory(plan['run_id'], temp); directory.mkdir(parents=True)
            journal = Journal(directory / 'journal.sqlite3', identity)
            for r in reqs:
                raw = response(r)
                journal.finish(journal.start(r['key'],1), {'status':'ok','score':parse_response(raw,r),
                               'response':raw,'usage':known_usage(raw),'error':None})
            report = summarize(reqs, journal.records(), ledger, new_attempts=6); journal.close()
            _save(directory/'prepared.json', reqs); _save(directory/'budget.json', ledger)
            atomic_json(directory/'summary.json',report)
            before = {p.name:p.read_bytes() for p in directory.iterdir()}
            with patch.object(inputs,'load_plan',return_value=plan), patch.dict(inputs.REPORT_HASHES,{arm:(report['report_sha256'],)}):
                margins, provenance = inputs.verify_arm(arm,expected,artifact_root=temp)
                self.assertEqual(len(margins),6); self.assertEqual(provenance['rows'],6)
                self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir()})
                with self.assertRaises(RunConflict): inputs.verify_arm(arm,dict(reversed(list(expected.items()))),artifact_root=temp)
                with closing(sqlite3.connect(directory/'journal.sqlite3')) as db:
                    db.execute("UPDATE attempts SET result_hash='corrupt' WHERE id=1")
                    db.commit()
                with self.assertRaises(RunConflict): inputs.verify_arm(arm,expected,artifact_root=temp)

    def test_unfinished_journal_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'journal.sqlite3'; journal=Journal(path,{'test':True})
            journal.start('key',1); journal.close(); before=path.read_bytes()
            with self.assertRaises(RunConflict): inputs.read_journal(path,{'test':True})
            self.assertEqual(path.read_bytes(),before)


@unittest.skipIf(np is None, 'isolated fitting dependencies not installed')
class MathTests(unittest.TestCase):
    def setUp(self): self.protocol=load_fit_protocol()

    def test_analytic_gradient_and_accepted_fit(self):
        x=np.array([-2,-2,-1,-1,0,0,1,1,2,2],dtype=float)
        y=np.array([0,0,0,1,0,1,0,1,1,1],dtype=float)
        p=np.array([.7,.2]); value,gradient=fm.objective(p,x,y,.0001)
        for i in range(2):
            delta=np.zeros(2); delta[i]=1e-6
            numerical=(fm.objective(p+delta,x,y,.0001)[0]-fm.objective(p-delta,x,y,.0001)[0])/2e-6
            self.assertAlmostEqual(gradient[i],numerical,places=8)
        result=fm.fit_calibration(x,y,self.protocol['calibration'])
        self.assertEqual(result['status'],'accepted',result)
        self.assertGreater(result['parameters']['a'],0)
        self.assertLessEqual(result['diagnostics']['final_objective'], result['diagnostics']['initial_objective'])

    def test_exact_f1_tie_largest_threshold_and_duplicates(self):
        # t=0: TP2 FP2 F1=2/3; t=3: TP1 FP0 F1=2/3.
        result=fm.select_threshold([0,1,2,3],[1,0,0,1])
        self.assertEqual(result['selected']['margin'],3)
        self.assertEqual(result['selected']['margin_hex'],float(3).hex())
        self.assertEqual(result['candidate_table'][-1]['kind'],'above_max')
        self.assertEqual(fm.select_threshold([0,0,1],[0,1,1])['candidate_table'][0]['tp'],2)
        self.assertEqual(len(fm.select_threshold([0,0,1],[0,1,1])['candidate_table']),3)

    def test_bin_edges_empty_bins_and_extreme_probabilities(self):
        r=fm.reliability([0,.1,.9,1],[0,1,0,1],self.protocol['calibration_diagnostics']['ece_edges'])
        self.assertEqual([b['count'] for b in r['bins']],[1,1,0,0,0,0,0,0,0,2])
        self.assertIsNone(r['bins'][2]['mean_probability'])
        self.assertAlmostEqual(r['brier'],.405)
        self.assertAlmostEqual(r['ece'],.45)
        with self.assertRaises(RunConflict): fm.reliability([1.1],[1],self.protocol['calibration_diagnostics']['ece_edges'])

    def test_failure_no_restart_and_independent_threshold(self):
        arms={'calibration':{'margins':[-1,1],'labels':[0,1]}, 'operating_threshold':{'margins':[0,1,2,3],'labels':[1,0,0,1]}}
        bad=SimpleNamespace(x=np.array([.001,0]),success=True,status=0,message='boundary',nit=1,nfev=1)
        with patch('scipy.optimize.minimize',return_value=bad) as optimize:
            r=fm.fit_arms(arms,self.protocol)
        optimize.assert_called_once()
        self.assertEqual(r['calibration']['status'],'failed'); self.assertIsNone(r['calibration']['parameters'])
        self.assertIn('boundary_reached',r['calibration']['failure_reasons'])
        self.assertEqual(r['operating_threshold']['selected']['margin'],3)
        self.assertIsNone(r['reliability']['operating_threshold']['calibrated'])

    def test_arm_isolation_and_cache_does_not_refit(self):
        arms={'calibration':{'margins':[-2,-2,0,0,2,2],'labels':[0,1,0,1,1,1]},
              'operating_threshold':{'margins':[-3,0,3],'labels':[0,1,1]}}
        other=deepcopy(arms); other['operating_threshold']['labels']=[1,0,0]
        self.assertEqual(fm.fit_arms(arms,self.protocol)['calibration'],fm.fit_arms(other,self.protocol)['calibration'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp); report,new=fit.save_fit(arms,self.protocol,{'test':True},path)
            before=(path/'fit.json').read_bytes()
            with patch.object(fit,'fit_arms',side_effect=AssertionError('must not refit')):
                cached,new=fit.save_fit(arms,self.protocol,{'test':True},path)
            self.assertFalse(new); self.assertEqual(cached,report); self.assertEqual((path/'fit.json').read_bytes(),before)
            with self.assertRaises(RunConflict): fit.save_fit(arms,self.protocol,{'test':False},path)

    def test_invalid_scores_not_dropped(self):
        for x,y in (([0,float('nan')],[0,1]),([0],[0,1]),([0,1],[1,1])):
            with self.assertRaises(RunConflict): fm.select_threshold(x,y)


if __name__ == '__main__': unittest.main()
