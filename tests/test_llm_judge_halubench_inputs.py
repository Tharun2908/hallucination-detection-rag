"""Synthetic checks for canonical split preservation and honest legacy linkage."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from post_thesis.llm_judge import prepare_halubench as prep
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    metadata = [dict(id='row-'+str(i), upstream_index=i, source_ds='source',
                     question='question '+str(i), passage='passage '+str(i)) for i in range(5)]
    # Exact passage merge across different questions crosses the outer split;
    # preserve membership and disclose it rather than secretly resplitting.
    metadata[3]['passage'] = metadata[0]['passage']
    selected = {i: {'answer':'answer '+str(i), 'label': 'FAIL' if i==2 else 'PASS'} for i in (4,2,3)}
    split = {'train_filtered_indices':[0,1], 'test_filtered_indices':[4,2,3],
             'actual_test_counts':{'source__0':2,'source__1':1}}
    rows = [{'idx':i, 'source':'source', 'label':int(i==2), 'raw_min_relevance':-2.0,
             's2_support':.3, 's4_score':.5, 'mc_hall':.4} for i in range(5)]
    return metadata, selected, split, {k:deepcopy(rows) for k in prep.CACHE_FILES}


def build(args):
    return prep.build_manifest(*args, revision='synthetic', total=5, test_size=3)


class HaluBenchInputTests(unittest.TestCase):
    def test_preserves_test_order_projection_groups_and_readiness(self):
        b=build(fixture());m=b['manifest']
        self.assertEqual([r['sample_id'] for r in m['model_inputs']],['row-4','row-2','row-3'])
        self.assertTrue(all(set(r)=={'sample_id','answer','context'} for r in m['model_inputs']))
        self.assertEqual(m['summary']['test_rows_in_components_touching_adaptation'],1)
        self.assertFalse(m['summary']['adaptation_answers_or_labels_used'])
        self.assertFalse(m['summary']['scoring_authorized'])
        self.assertFalse(m['summary']['baseline_comparison_ready'])
        self.assertTrue(all(not x['answer_context_alignment_verified'] for x in m['baseline_inventory'].values()))
        for row,offline in zip(m['model_inputs'],m['offline_rows']):
            self.assertEqual(content_hash({k:row[k] for k in ('answer','context')}),offline['input_sha256'])
        self.assertEqual(b['manifest_sha256'],content_hash(m))

    def test_bad_split_membership_and_indices_fail(self):
        for kind in ('duplicate','overlap','bool','out_of_range','selected_missing'):
            args=fixture()
            if kind=='duplicate':args[2]['test_filtered_indices']=[2,2,4]
            if kind=='overlap':args[2]['test_filtered_indices']=[0,2,4]
            if kind=='bool':args[2]['train_filtered_indices'][0]=False
            if kind=='out_of_range':args[2]['test_filtered_indices'][0]=9
            if kind=='selected_missing':args[1].pop(4)
            with self.subTest(kind=kind), self.assertRaises(RunConflict):build(args)

    def test_canonical_normalized_group_overlap_blocks(self):
        args=fixture();args[0][3]['question']='  question   0\n'
        with self.assertRaises(RunConflict):build(args)
        self.assertEqual(prep.norm(' A  B\n'),'A B')
        self.assertNotEqual(prep.norm('A'),prep.norm('a'))

    def test_group_closure_is_transitive_and_empty_passages_do_not_merge(self):
        rows=[dict(id=str(i),source_ds='s',question=q,passage=c) for i,(q,c) in enumerate([
            ('q1',' A  B '),('q1','A B'),('q2','A B'),('q3',''),('q4','')])]
        canonical,components=prep.group_components(rows)
        self.assertEqual(canonical[0],canonical[1])
        self.assertEqual(components[0],components[2])
        self.assertNotEqual(components[3],components[4])

    def test_invalid_identity_label_and_canonical_counts_fail(self):
        for kind in ('id','filter','label','counts','upstream_order'):
            args=fixture()
            if kind=='id':args[0][2]['id']=args[0][0]['id']
            if kind=='filter':args[0][2]['source_ds']='RAGTruth'
            if kind=='label':args[1][2]['label']='UNKNOWN'
            if kind=='counts':args[2]['actual_test_counts']['source__1']=2
            if kind=='upstream_order':args[0][2]['upstream_index']=0
            with self.subTest(kind=kind), self.assertRaises(RunConflict):build(args)

    def test_legacy_corruption_rejected_and_no_false_content_attestation(self):
        for kind in ('label','source','score','boolean_score','duplicate','missing'):
            args=fixture();rows=args[3]['S2_S4']
            if kind=='label':rows[2]['label']=0
            if kind=='source':rows[2]['source']='other'
            if kind=='score':rows[2]['s4_score']=float('nan')
            if kind=='boolean_score':rows[2]['s4_score']=True
            if kind=='duplicate':rows.append(deepcopy(rows[0]))
            if kind=='missing':rows.pop()
            with self.subTest(kind=kind), self.assertRaises(RunConflict):build(args)

    def test_replay_unchanged_and_changed_text_not_overwritten(self):
        args=fixture();b=build(args)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json'
            self.assertTrue(prep.save_once(path,b));before=path.read_bytes()
            self.assertFalse(prep.save_once(path,build(args)))
            self.assertEqual(path.read_bytes(),before)
            args[1][2]['answer']+=' changed'
            with self.assertRaises(RunConflict):prep.save_once(path,build(args))
            self.assertEqual(path.read_bytes(),before)

    def test_wrong_parquet_rejected_without_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.parquet';path.write_bytes(b'wrong')
            with self.assertRaises(RunConflict):prep.project_parquet(path,{})


if __name__=='__main__':unittest.main()
