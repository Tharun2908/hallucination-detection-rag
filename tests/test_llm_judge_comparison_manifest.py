"""Artificial records only; no benchmark data or model imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import freeze_comparison as freeze
from post_thesis.llm_judge.check_evaluation_math import load_contract
from post_thesis.llm_judge.evaluation_math import comparison_gate
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    targets=[{'sample_id':str(i),'test_index':i,'input_sha256':'input-'+str(i),
              'component_sha256':'group'} for i in range(2)]
    reports={}
    for name,spec in freeze.SOURCES.items():
        identity={'TEST_manifest_sha256':freeze.TEST_MANIFEST_SHA256}
        report={'study_stage':'post_thesis','run_id':spec[0],'identity':identity,
                'valid_scores':2,'pending':0,'terminal_failures':0,
                'predictions':[{**r,'status':'ok',spec[3]:.3} for r in targets]}
        if name=='S2_S4_metadata_free': report.update(identity)
        reports[name]=report
    model={'metadata_features':False,'coef':[1.,2.]}
    reports['S2_S4_metadata_free'].update(full_model=model,
        threshold={'value':.45,'comparator':'>='},
        fusion_recovery_report_sha256=freeze.FUSION_RECOVERY[2],
        fresh_source_report_sha256={'S2':freeze.SOURCES['S2_dependency'][2],
                                    'S4':freeze.SOURCES['S4'][2]})
    legacy={'threshold_reproduction':{}}
    for name,t,c in [('S4',.55,'>='),('MC_7B',.20000000000000004,'<')]:
        legacy['threshold_reproduction'][name]={'status':'matches_historical_TRAIN_audit',
            'reproduced':{'threshold':t},'comparator':c}
    recovery={'results':{'threshold':{'selected':{'threshold':.45}},'full_model':model}}
    drift={'fresh_report_sha256':freeze.SOURCES['MiniCheck_7B'][2],
           'agreement':{'fixed_threshold_decision_changes':100,
                        'mean_absolute_difference':.02947428918387699,
                        'maximum_absolute_difference':.3774411483820142}}
    return targets,reports,legacy,recovery,drift


class ComparisonManifestTests(unittest.TestCase):
    def test_aligned_scope_keeps_judge_gate_closed(self):
        args=fixture()
        manifest=freeze.build_manifest('synthetic',load_contract(),*args)
        gate=comparison_gate(manifest['baselines'])
        self.assertEqual(gate['status'],'blocked')
        self.assertEqual(set(gate['missing_requirements']),{'judge'})
        self.assertFalse(manifest['paired_metrics_ready'])
        self.assertEqual(manifest['generation_allowance'],0)
        for source in manifest['baselines'].values():
            self.assertFalse(source['historical_training_provenance_independently_verified'])
            self.assertFalse(source['exact_thesis_cache_reproduction_claimed'])

    def test_changed_content_order_or_coverage_rejected(self):
        targets,reports,*_=fixture()
        changes=[lambda r:r['predictions'][0].update(input_sha256='changed'),
                 lambda r:r['predictions'].reverse(),
                 lambda r:r.update(valid_scores=1),
                 lambda r:r.update(pending=1),
                 lambda r:r['predictions'][0].update(status='invalid_output')]
        for change in changes:
            with self.subTest(change=change):
                altered=copy.deepcopy(reports);change(altered['MiniCheck_7B'])
                with self.assertRaises(RunConflict): freeze.align_sources(altered,targets)

    def test_nonfinite_boolean_and_out_of_range_rejected(self):
        targets,reports,*_=fixture()
        for score in (float('nan'),float('inf'),True,-.1,1.1):
            reports['S4']['predictions'][0]['unsupported_score']=score
            with self.subTest(score=score), self.assertRaises(RunConflict):
                freeze.align_sources(reports,targets)

    def test_labels_and_metadata_not_projected(self):
        targets,reports,*_=fixture()
        expected=freeze.align_sources(reports,targets)
        for report in reports.values():
            for row in report['predictions']:
                row.update(label=object(),task=object(),generator=object())
        self.assertEqual(freeze.align_sources(reports,targets),expected)
        self.assertEqual(set(expected['S4'][0]),{'sample_id','test_index','input_sha256','score'})

    def test_threshold_or_fusion_provenance_change_rejected(self):
        for kind in ('threshold','source','model'):
            args=list(fixture()); reports=args[1];legacy=args[2]
            if kind=='threshold': legacy['threshold_reproduction']['MC_7B']['comparator']='>='
            if kind=='source': reports['S2_S4_metadata_free']['fresh_source_report_sha256']['S2']='wrong'
            if kind=='model': reports['S2_S4_metadata_free']['full_model']={'metadata_features':True}
            with self.subTest(kind=kind), self.assertRaises(RunConflict):
                freeze.build_manifest('synthetic',load_contract(),*args)

    def test_report_pin_checks_bytes_content_and_declared_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'report.json';report={'synthetic':True};digest=content_hash(report)
            with patch.object(freeze,'run_directory',return_value=Path(temp)):
                p.write_text(json.dumps({'report':report,'report_sha256':digest}))
                self.assertEqual(freeze.read_report(('synthetic','report.json',digest)),report)
                p.write_text(json.dumps({'report':{'synthetic':False},'report_sha256':digest}))
                with self.assertRaises(RunConflict):
                    freeze.read_report(('synthetic','report.json',digest))


if __name__=='__main__': unittest.main()
