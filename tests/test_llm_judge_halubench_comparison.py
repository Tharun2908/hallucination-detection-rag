"""Artificial full-population alignment; no benchmark scores or metrics."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import halubench_comparison as c
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    targets=[{'sample_id':str(i),'test_index':i,'input_sha256':'h'+str(i),
              'component_sha256':str(i%7198)} for i in range(8000)]
    reports={}
    for name,spec in c.SOURCES.items():
        r={'study_stage':'post_thesis','run_id':spec[0],'valid_scores':8000,'pending':0,'terminal_failures':0,
           'HaluBench_read':True,'adaptation_training':False,
           'identity':{'TEST_manifest_sha256':c.TEST_MANIFEST_SHA256},
           'TEST_manifest_sha256':c.TEST_MANIFEST_SHA256,
           'predictions':[{**t,'status':'ok',spec[3]:.3} for t in targets]}
        reports[name]=r
    contract=c.load_contract()
    legacy={'threshold_reproduction':{old:{'status':'matches_historical_TRAIN_audit',
             'reproduced':{'threshold':contract['operating_rules'][name]['threshold']},
             'comparator':contract['operating_rules'][name]['comparator']}
             for name,old in [('S4','S4'),('MiniCheck_7B','MC_7B')]}}
    model={'metadata_features':False,'artificial':True}
    recovery={'results':{'full_model':model}}
    reports['S2_S4_metadata_free'].update(full_model=model,
        threshold={'value':.45,'comparator':'>='},fusion_recovery_report_sha256=c.FUSION_RECOVERY[2],
        fresh_source_report_sha256={'S2':c.SOURCES['S2_dependency'][2],'S4':c.SOURCES['S4'][2]})
    return targets,reports,legacy,recovery


class ComparisonTests(unittest.TestCase):
    def test_all_8000_rows_and_7198_components_required(self):
        targets,reports,_,_=fixture()
        c.align_sources(reports,targets)
        with self.assertRaises(RunConflict): c.align_sources(reports,targets[:-1])
        targets[0]['component_sha256']='different'
        with self.assertRaises(RunConflict): c.align_sources(reports,targets)

    def test_each_baseline_rejects_bad_identity_scores_and_adaptation(self):
        for name in c.SOURCES:
            for field,value in [('input_sha256','changed'),('test_index',3),('status','error')]:
                t,r,_,_=fixture();r[name]['predictions'][0][field]=value
                with self.assertRaises(RunConflict): c.align_sources(r,t)
            t,r,_,_=fixture();r[name]['predictions'][0][c.SOURCES[name][3]]=float('nan')
            with self.assertRaises(RunConflict): c.align_sources(r,t)
            t,r,_,_=fixture();r[name]['adaptation_training']=True
            with self.assertRaises(RunConflict): c.align_sources(r,t)

    def test_verified_manifest_immutable_replay_and_threshold_guard(self):
        t,reports,legacy,recovery=fixture()
        def read(spec):
            if spec==c.LEGACY_PROVENANCE:return legacy
            if spec==c.FUSION_RECOVERY:return recovery
            return next(reports[k] for k,v in c.SOURCES.items() if v==spec)
        with tempfile.TemporaryDirectory() as d, patch.object(c,'read_report',side_effect=read), \
             patch.object(c,'run_directory',return_value=Path(d)):
            manifest=c.verify_and_save(t);path=Path(d)/'manifest.json';first=path.read_bytes()
            self.assertEqual(c.verify_and_save(t),manifest);self.assertEqual(path.read_bytes(),first)
            self.assertEqual(c.content_hash(manifest),c.COMPARISON_SHA256)
            legacy['threshold_reproduction']['S4']['reproduced']['threshold']=.5
            with self.assertRaises(RunConflict):c.verify_and_save(t)
            self.assertEqual(path.read_bytes(),first)


if __name__=='__main__': unittest.main()
