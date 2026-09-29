"""Artificial canonical-size joins and CPU evaluation; no benchmark/model calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import evaluate_halubench as ev
from post_thesis.llm_judge import evaluation_math as metrics
from post_thesis.llm_judge.check_evaluation_math import synthetic_fixture
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict
import test_llm_judge_halubench_comparison as baseline
import test_llm_judge_test_evaluation as existing


def population():
    targets,reports,legacy,recovery=baseline.fixture()
    sources=sorted(ev.SOURCE_LEVELS)
    offline=[{**r,'label':i%2,'metadata':{'source':sources[i%len(sources)]}} for i,r in enumerate(targets)]
    rows=[{**t,'status':'ok' if i<7652 else 'interrupted' if i==7652 else 'pending',
           'score':{'unsupported_log_odds':float(i%3-1)} if i<7652 else None,'latency_seconds':.1}
          for i,t in enumerate(targets)]
    parent={'predictions':rows}
    child={'predictions':[{**t,'status':'ok','score':{'unsupported_log_odds':float(i%3-1)},'latency_seconds':.2}
                          for i,t in enumerate(targets[7652:])]}
    return targets,offline,reports,legacy,recovery,parent,child


class AlignmentTests(unittest.TestCase):
    def test_exact_once_join_preserves_parent_and_source_axes(self):
        t,offline,reports,_,_,parent,child=population();before=deepcopy((parent,child))
        merged=ev.merge_predictions(parent,child,t)
        self.assertEqual(len(merged),8000);self.assertEqual((parent,child),before)
        self.assertIs(merged[0],parent['predictions'][0]);self.assertIs(merged[7652],child['predictions'][0])
        joined=ev.join_evaluation_rows(offline,{'predictions':merged},reports,ev.comparison.descriptor())
        self.assertEqual(len(set(joined['groups'])),7198)
        self.assertEqual(set(joined['axes']['source']),ev.SOURCE_LEVELS)
        self.assertEqual(joined['scores']['MiniCheck_7B'],[.3]*8000)
        self.assertEqual(set(joined['scores']),set(metrics.MAIN_SYSTEMS))
        self.assertEqual(metrics.comparison_gate(joined['provenance'])['status'],'ready')
        self.assertFalse(joined['provenance']['S4']['historical_training_provenance_independently_verified'])

    def test_missing_duplicate_swapped_failed_or_wrong_input_block_merge(self):
        for kind in ('missing','duplicate','swap','failure','input','parent_status','nonfinite'):
            t,_,_,_,_,parent,child=population()
            if kind=='missing':child['predictions'].pop()
            if kind=='duplicate':child['predictions'][1]=child['predictions'][0]
            if kind=='swap':child['predictions'].reverse()
            if kind=='failure':child['predictions'][0]['status']='failed'
            if kind=='input':child['predictions'][0]['input_sha256']='wrong'
            if kind=='parent_status':parent['predictions'][0]['status']='pending'
            if kind=='nonfinite':child['predictions'][0]['score']['unsupported_log_odds']=float('nan')
            with self.subTest(kind=kind),self.assertRaises(RunConflict):ev.merge_predictions(parent,child,t)

    def test_labels_sources_baseline_alignment_and_readiness_checked(self):
        for kind in ('label','source','baseline','unready'):
            t,offline,reports,_,_,parent,child=population();desc=ev.comparison.descriptor()
            merged=ev.merge_predictions(parent,child,t)
            if kind=='label':offline[0]['label']=.0
            if kind=='source':offline[0]['metadata']['source']='unknown'
            if kind=='baseline':reports['S4']['predictions'][0]['input_sha256']='wrong'
            if kind=='unready':desc['baselines']['S4']['comparison_ready']=False
            with self.subTest(kind=kind),self.assertRaises(RunConflict):
                ev.join_evaluation_rows(offline,{'predictions':merged},reports,desc)

    def test_baseline_read_only_validation_rejects_threshold_change(self):
        t,_,reports,legacy,recovery,_,_=population()
        def read(spec):
            if spec==ev.LEGACY_PROVENANCE:return legacy
            if spec==ev.FUSION_RECOVERY:return recovery
            return next(reports[k] for k,v in ev.comparison.SOURCES.items() if v==spec)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json';desc=ev.comparison.descriptor()
            path.write_text(json.dumps({'manifest_sha256':content_hash(desc),'manifest':desc}));before=path.read_bytes()
            with patch.object(ev,'read_report',side_effect=read),patch.object(ev,'run_directory',return_value=Path(temp)):
                self.assertEqual(ev.verify_baselines(t)[0],reports)
                legacy['threshold_reproduction']['S4']['reproduced']['threshold']=.5
                with self.assertRaises(RunConflict):ev.verify_baselines(t)
            self.assertEqual(path.read_bytes(),before)

    def test_cached_evaluation_does_not_recompute_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.json';calls=[]
            def compute(inputs):calls.append(inputs);return {'synthetic':True}
            first,new=ev.evaluate_once(path,{'fixed':'identity'},{'fake':True},{},[],compute)
            self.assertTrue(new);before=path.read_bytes()
            second,new=ev.evaluate_once(path,{'fixed':'identity'},{'fake':True},{},[],compute)
            self.assertFalse(new);self.assertEqual(first,second);self.assertEqual(len(calls),1)
            self.assertTrue(first['report']['HaluBench_read']);self.assertFalse(first['report']['adaptation_training'])
            with self.assertRaises(RunConflict):ev.evaluate_once(path,{'fixed':'changed'},{},{},[],compute)
            self.assertEqual(path.read_bytes(),before)
            changed=deepcopy(first);changed['report']['results']['synthetic']=False;path.write_text(json.dumps(changed))
            with self.assertRaises(RunConflict):ev.evaluate_once(path,{'fixed':'identity'},{},{},[],compute)

    def test_byte_pinned_helpers_and_frozen_numeric_contract(self):
        ev.verify_implementation()
        self.assertEqual(ev.load_contract()['operating_rules']['judge']['threshold'],-1.25)


class ReplayTests(existing.ReadOnlyReplayTests):
    # Reuse real artificial raw-response journals, not trusted synthetic score arrays.
    def test_completed_child_raw_replay_and_source_bytes_preserved(self):
        from post_thesis.llm_judge import continue_halubench_scores as child
        from post_thesis.llm_judge.label_diagnose import _load, _save
        ledger=_load(self.path/'budget.json');requests=_load(self.path/'prepared.json');identity=ledger['identity']
        identity['preparation'].update(parent_known_token_totals={'input_tokens':6204778,'output_tokens':7652},
            parent_unknown_usage_attempts={'input_tokens':1,'output_tokens':1},parent_charged_client_seconds=14405.70098266704)
        # Replace only this artificial journal's manifest so it represents a child fixture.
        import sqlite3
        with sqlite3.connect(self.path/'journal.sqlite3') as db:
            db.execute('UPDATE run SET manifest=? WHERE singleton=1',(json.dumps(identity),))
        _save(self.path/'budget.json',ledger)
        records=child.read_records(self.path/'journal.sqlite3',identity)
        bundle=child.summarize(requests,records,ledger)
        (self.path/'summary.json').write_text(json.dumps(bundle))
        paths=[self.path/name for name in ('budget.json','prepared.json','summary.json','journal.sqlite3')]
        before={p.name:p.read_bytes() for p in paths}
        with patch.object(child,'run_directory',return_value=self.path),patch.object(child,'load_plan',return_value=identity['plan']), \
             patch.object(child,'verify_parent',return_value=(requests,identity['preparation'])):
            replay=child.inspect_child('synthetic')
            self.assertEqual(replay['report_sha256'],bundle['report_sha256'])
            self.assertEqual(replay['report']['cumulative_unknown_usage_attempts'],{'input_tokens':1,'output_tokens':1})
            with self.assertRaises(RunConflict):child.inspect_child('wrong-revision')
        self.assertEqual(before,{p.name:p.read_bytes() for p in paths})


@unittest.skipUnless(existing.NUMERIC,'pinned numerical packages unavailable')
class NumericalApplicationTests(unittest.TestCase):
    def test_all_source_slices_and_preexisting_bootstrap_contract(self):
        fixture=synthetic_fixture()
        fixture['axes']={'source':[sorted(ev.SOURCE_LEVELS)[i%5] for i in range(12)]}
        fixture['provenance']={name:{'comparison_ready':True,**{k:True for k in metrics.PROVENANCE_FIELDS}} for name in metrics.MAIN_SYSTEMS}
        result=ev.compute(fixture);paired=result['paired']
        self.assertEqual(paired['shared_n'],12)
        self.assertEqual(paired['bootstrap']['replicates'],2000)
        self.assertEqual(paired['bootstrap']['seed'],20260920)
        self.assertFalse(paired['bootstrap']['refitting'])
        self.assertEqual(set(result['descriptive_slices']['judge']['source']),ev.SOURCE_LEVELS)
        self.assertEqual(paired['individual']['judge']['rule']['threshold'],-1.25)
        for name in ('S4','MiniCheck_7B','S2_S4_metadata_free'):
            self.assertEqual(paired['paired_differences'][name]['auroc']['valid_draws'],2000)
            self.assertEqual(paired['paired_differences'][name]['ece_calibrated']['baseline_metric'],'ece_raw')

    def test_cumulative_time_and_unknown_usage_not_erased(self):
        t,_,reports,_,_,parent,child=population()
        merged=ev.merge_predictions(parent,child,t)
        for name in ('S4','S2_dependency','MiniCheck_7B'): reports[name]['timing']={}
        reports['S4']['truncated_examples']={'context':850,'answer':0}
        reports['S2_dependency']['truncated_pairs']={'context_pairs':35,'answer_pairs':0}
        reports['MiniCheck_7B'].update(known_token_totals={'input_tokens':4905370,'output_tokens':9158},workload={})
        judge={'predictions':merged,'charged_client_seconds':14405.70098266704+573.9732121489942,
            'known_token_totals':{'input_tokens':7794485,'output_tokens':8000},
            'unknown_usage_attempts':{'input_tokens':1,'output_tokens':1}}
        result=ev.observed_efficiency(judge,reports)
        self.assertEqual(result['judge']['total_client_attempts'],8001)
        self.assertEqual(result['judge']['unknown_usage_attempts']['input_tokens'],1)
        self.assertEqual(result['judge']['client_seconds'],judge['charged_client_seconds'])
        self.assertIsNone(result['judge']['monetary_cost'])
        self.assertEqual(result['comparison'],'descriptive_only_no_speedup_claim')


if __name__=='__main__':unittest.main()
