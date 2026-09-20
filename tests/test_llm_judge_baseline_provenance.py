"""Artificial baseline audit data; no benchmark performance claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from post_thesis.llm_judge import audit_baseline_provenance as audit
from post_thesis.llm_judge.storage import RunConflict
try:
    import numpy as np
    import sklearn
except ImportError:
    np=None


def fixture(n=20):
    expected=[{'label':i%2,'metadata':{'model':'synthetic','task_type':'QA'}} for i in range(n)]
    rows=[{'idx':i,'ground_truth_hallucination':i%2,'model':'synthetic','task_type':'QA','score':.2 if i%2==0 else .8} for i in range(n)]
    return expected,rows


class IntegrityTests(unittest.TestCase):
    def test_alignment_canonicalizes_order_without_claiming_text_identity(self):
        expected,rows=fixture()
        aligned,report=audit.align(list(reversed(rows)),expected,'score',True)
        self.assertEqual(aligned,rows);self.assertFalse(report['answer_context_alignment_verified'])
        self.assertEqual(report['valid_scores'],20)

    def test_invalid_rows_rejected_and_null_preserved(self):
        expected,rows=fixture()
        for kind in ('duplicate','label','metadata','score'):
            broken=deepcopy(rows)
            if kind=='duplicate':broken[1]['idx']=0
            if kind=='label':broken[0]['ground_truth_hallucination']=1
            if kind=='metadata':broken[0]['model']='other'
            if kind=='score':broken[0]['score']=True
            with self.subTest(kind=kind),self.assertRaises(RunConflict):audit.align(broken,expected,'score',True)
        rows[0]['score']=None
        aligned,report=audit.align(rows,expected,'score',True)
        self.assertEqual(report['missing_scores'],1);self.assertIsNone(aligned[0]['score'])

    def test_checkpoint_files_hashed_not_deserialized(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'config.json').write_text('{}');(path/'model.safetensors').write_bytes(b'not loadable')
            (path/'training_args.bin').write_bytes(b'never deserialize')
            before={p.name:p.read_bytes() for p in path.iterdir()}
            report=audit.checkpoint_inventory(path)
            self.assertTrue(report['weight_files_present']);self.assertFalse(report['model_to_cache_link_verified'])
            self.assertNotIn('training_args.bin',report['files'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})
            self.assertEqual(report,audit.checkpoint_inventory(path))


@unittest.skipIf(np is None,'isolated NumPy/sklearn dependencies not installed')
class NumericalTests(unittest.TestCase):
    def test_support_float_boundary_and_no_hallucination_space_substitution(self):
        rows=[{'score':.2,'ground_truth_hallucination':1},{'score':.5,'ground_truth_hallucination':0}]
        threshold=float(np.arange(.10,.901,.05)[2])
        self.assertGreater(threshold,.2)
        report=audit.reproduce_threshold(rows,'score',support=True,reference={'threshold':threshold,'train_f1':1.0})
        self.assertEqual(report['status'],'matches_historical_TRAIN_audit')
        self.assertEqual(report['comparator'],'<');self.assertEqual(report['reproduced']['tp'],1)
        json.dumps(report,allow_nan=False)

    def test_first_maximum_tie_and_mismatch_does_not_change_policy(self):
        rows=[{'score':.8,'ground_truth_hallucination':1},{'score':.1,'ground_truth_hallucination':0}]
        first=float(np.arange(.05,.951,.05)[2])
        report=audit.reproduce_threshold(rows,'score',support=False,reference={'threshold':first,'train_f1':1.0})
        self.assertEqual(report['reproduced']['threshold'],first)
        bad=audit.reproduce_threshold(rows,'score',support=False,reference={'threshold':.9,'train_f1':0.0})
        self.assertEqual(bad['status'],'mismatch_do_not_replace_historical_policy');self.assertFalse(bad['policy_changed'])
        rows[0]['score']=None
        with self.assertRaises(RunConflict):audit.reproduce_threshold(rows,'score',support=False,reference={'threshold':first,'train_f1':1.0})

    def test_oof_membership_is_reconstructed_not_just_counted(self):
        from sklearn.model_selection import StratifiedKFold
        _,rows=fixture()
        y=[r['ground_truth_hallucination'] for r in rows]
        for fold,(_,indices) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(rows)),y),1):
            for i in indices:rows[i].update(fold=fold,score_type='out_of_fold')
        self.assertTrue(audit.oof_check(rows)['fold_membership_matches'])
        rows[0]['fold']=rows[0]['fold']%5+1
        with self.assertRaises(RunConflict):audit.oof_check(rows)


if __name__=='__main__':unittest.main()
