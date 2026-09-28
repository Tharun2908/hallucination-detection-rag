"""Artificial HaluBench alignment, fixed math and immutable CPU application."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import apply_halubench_fusion as fusion
from post_thesis.llm_judge.apply_fresh_fusion import COEF, INTERCEPT
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    targets = [{'sample_id':str(i), 'test_index':i, 'input_sha256':content_hash(i)} for i in range(8000)]
    common = {'study_stage':'post_thesis', 'examples_total':8000, 'valid_scores':8000,
              'pending':0, 'terminal_failures':0, 'HaluBench_read':True, 'adaptation_training':False,
              'identity':{'TEST_manifest_sha256':fusion.TEST_MANIFEST_SHA256,
                          'judge_freeze_sha256':fusion.FREEZE_SHA256}}
    s2 = {**common, 'run_id':fusion.SOURCES['S2'][0], 'empty_pair_examples':1886,
          'truncated_pairs':{'answer_pairs':0,'context_pairs':35},
          'predictions':[{**t, 'status':'ok', 'pairs':int(i>=1886),
                          'raw_min_relevance':0. if i<1886 else .1234567} for i,t in enumerate(targets)]}
    s4 = {**common, 'run_id':fusion.SOURCES['S4'][0],
          'truncated_examples':{'answer':0,'context':850},
          'predictions':[{**t, 'status':'ok', 'unsupported_score':.543276} for t in targets]}
    recovery = {'limitations':['legacy TRAIN provenance remains limited'], 'results':{'full_model':{
        'feature_order':['s2','s4'], 'metadata_features':False, 'classes':[0,1],
        'coef':COEF[:], 'intercept':INTERCEPT,
        's2_normalization':{'min':-11.43,'max':10.641,'clip':[0.,1.]}},
        'threshold':{'comparator':'>=', 'selected':{'threshold':fusion.THRESHOLD,
                                                  'threshold_hex':fusion.THRESHOLD.hex()}}}}
    return {'offline_rows':targets}, s2, s4, recovery


def build(args):
    return fusion.build_report(*args, revision='artificial', versions=fusion.VERSIONS,
                               implementation=fusion.IMPLEMENTATION_FILES)


class ContractTests(unittest.TestCase):
    def test_source_and_implementation_pins(self):
        self.assertEqual(fusion.verify_implementation(),fusion.IMPLEMENTATION_FILES)
        with patch.object(fusion,'file_sha256',return_value='changed'):
            with self.assertRaises(RunConflict): fusion.verify_implementation()
        _,s2,_,_=fixture()
        for field,value in [('valid_scores',7999),('pending',1),('adaptation_training',True),
                            ('run_id','ragtruth'),('HaluBench_read',False)]:
            bad={**s2,field:value}
            with self.assertRaises(RunConflict): fusion.validate_source(bad,'S2')

    def test_empty_pair_fallback_cannot_change(self):
        _,s2,_,_=fixture()
        s2['predictions'][0]['raw_min_relevance']=.5
        with self.assertRaises(RunConflict): fusion.validate_source(s2,'S2')

    def test_artifact_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'summary.json'
            path.write_text(json.dumps({'report':{'changed':True},'report_sha256':fusion.SOURCES['S2'][1]}))
            with self.assertRaises(RunConflict):
                fusion.checked_bundle(path,'report','report_sha256',fusion.SOURCES['S2'][1])


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),
                     'requires CPU fit dependencies')
class ApplicationTests(unittest.TestCase):
    def test_zero_fallback_rounding_and_fixed_math_without_fit(self):
        args=fixture()
        for row in args[0]['offline_rows']: row.update(label='never_used',metadata={'source':'never_used'})
        with patch('post_thesis.llm_judge.recover_fusion.reconstruct',side_effect=AssertionError('no fit')):
            report=build(args)['report']
        self.assertEqual(report['valid_scores'],8000)
        for index,raw in ((0,0.),(1886,.1235)):
            row=report['predictions'][index]
            norm=(raw+11.43)/(10.641+11.43)
            z=COEF[0]*norm+COEF[1]*.5433+INTERCEPT
            self.assertAlmostEqual(row['s2_normalized'],norm,places=14)
            self.assertAlmostEqual(row['unsupported_score'],1/(1+math.exp(-z)),places=14)
            self.assertNotIn('label',row)
            self.assertNotIn('metadata',row)
        self.assertEqual(report['fitting_calls'],0)
        self.assertFalse(report['TEST_label_metrics_computed'])

    def test_input_alignment_and_threshold_drift_fail_closed(self):
        for field,value in [('input_sha256','changed'),('sample_id','duplicate'),('test_index',42),('status','error')]:
            args=fixture(); args[1]['predictions'][0][field]=value
            with self.assertRaises(RunConflict): build(args)
        args=fixture(); args[3]['results']['threshold']['comparator']='>'
        with self.assertRaises(RunConflict): build(args)

    def test_replay_identical_and_conflict_preserves_output(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'; bundle=build(fixture())
            self.assertTrue(fusion.save_once(path,bundle))
            before=path.read_bytes()
            self.assertFalse(fusion.save_once(path,build(fixture())))
            self.assertEqual(path.read_bytes(),before)
            args=fixture(); args[2]['predictions'][0]['unsupported_score']=.1
            with self.assertRaises(RunConflict): fusion.save_once(path,build(args))
            self.assertEqual(path.read_bytes(),before)

    def test_cli_uses_only_pinned_cpu_artifacts(self):
        manifest,s2,s4,recovery=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); input_dir=root/fusion.INPUT_RUN_ID; input_dir.mkdir()
            (input_dir/'manifest.json').write_text(json.dumps({'manifest':manifest}))
            def read(path,*args):
                if path.parent.name==fusion.RECOVERY_RUN_ID: return recovery
                return s2 if path.parent.name==fusion.SOURCES['S2'][0] else s4
            with patch.object(fusion,'run_directory',side_effect=lambda name:root/name), \
                 patch.object(fusion,'code_revision',return_value='artificial'), \
                 patch.object(fusion,'validate_inputs'), patch.object(fusion,'load_freeze'), \
                 patch.object(fusion,'checked_bundle',side_effect=read), \
                 patch.object(fusion,'version',side_effect=lambda p:fusion.VERSIONS[p]), \
                 patch('sys.argv',['apply_halubench_fusion']), patch('builtins.print'):
                self.assertEqual(fusion.main(),0)
                path=root/fusion.RUN_ID/'report.json'; before=path.read_bytes()
                self.assertEqual(fusion.main(),0)
                self.assertEqual(path.read_bytes(),before)


if __name__=='__main__': unittest.main()
