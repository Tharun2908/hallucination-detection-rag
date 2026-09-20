"""Artificial-data tests for exact TEST alignment and honest baseline inventory."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import prepare_test_manifest as prep
from post_thesis.llm_judge import baseline_inventory as inv
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.source_audit import context_from_native
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    sources={sid:{'source_id':sid,'task_type':'Summary','source':'synthetic','source_info':'Article '+sid} for sid in ('train','s1','s2')}
    native=[{'id':rid,'source_id':sid,'split':split,'response':'Answer '+rid,'model':'synthetic','quality':'good',
             'labels':object()} for rid,sid,split in [('0','train','train'),('1','s1','test'),('2','s2','test')]]
    rows=[{'id':r['id'],'output':r['response'],'context':context_from_native(sources[r['source_id']]),
           'model':'synthetic','quality':'good','task_type':'Summary',
           'hallucination_labels_processed':{'evident_conflict':int(r['id']=='2'),'baseless_info':0}} for r in native[1:]]
    components=prep.component_map(sources)
    cross={'native_split_metadata_sha256':content_hash([{k:r[k] for k in ('id','source_id','split')} for r in native]),
           'train_mapping':[{'sample_id':'0','source_id':'train','component_sha256':components['train']}]}
    development={'train_reservation':[{'partition':'calibration','component_sha256':components['train'],'linked_to_pilot':False}]}
    return rows,native,sources,cross,development


def build(args): return prep.build_manifest(*args,revision='artificial-test',fit_hash='artificial-fit')


class AlignmentTests(unittest.TestCase):
    def test_exact_inputs_offline_labels_no_native_annotations_and_replay(self):
        args=fixture();bundle=build(args)
        self.assertEqual(bundle,build(args))
        for row in bundle['manifest']['model_inputs']:
            self.assertEqual(set(row),{'sample_id','answer','context'})
        self.assertEqual(bundle['manifest']['summary']['exact_formatted_context_matches'],2)
        self.assertFalse(bundle['manifest']['summary']['metrics_computed'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json';self.assertTrue(prep.save_once(path,bundle));before=path.read_bytes()
            self.assertFalse(prep.save_once(path,bundle));self.assertEqual(before,path.read_bytes())
            changed=deepcopy(bundle);changed['manifest']['version']='changed'
            with self.assertRaises(RunConflict):prep.save_once(path,changed)

    def test_membership_answer_context_metadata_and_labels_are_not_silently_repaired(self):
        for kind in ('missing','duplicate','answer','context','metadata','label'):
            args=fixture();rows=args[0]
            if kind=='missing':rows.pop()
            if kind=='duplicate':rows.append(deepcopy(rows[0]))
            if kind=='answer':rows[0]['output']='Different answer'
            if kind=='context':rows[0]['context']+=' '
            if kind=='metadata':rows[0]['model']='other'
            if kind=='label':rows[0]['hallucination_labels_processed']['baseless_info']=True
            with self.subTest(kind=kind),self.assertRaises(RunConflict):build(args)

    def test_pilot_fit_overlap_and_prior_group_drift_block_manifest(self):
        for kind in ('pilot','fit','group'):
            args=fixture();component=prep.component_map(args[2])['s1']
            if kind=='group':args[3]['train_mapping'][0]['component_sha256']='wrong'
            else:args[4]['train_reservation'].append({'component_sha256':component,'partition':'calibration' if kind=='fit' else 'excluded','linked_to_pilot':kind=='pilot'})
            with self.subTest(kind=kind),self.assertRaises(RunConflict):build(args)

    def test_exact_reviewed_whitespace_exception_only(self):
        original='First.\n\nNext.';native='First.\n\n Next.'
        known={'source_id':'s','sample_ids':['r'],'processed_context_sha256':content_hash(original),
               'native_formatted_context_sha256':content_hash(native),'native_extra_ASCII_space_offset':8}
        with patch.object(prep,'KNOWN_CONTEXT_DIFFERENCE',known):
            self.assertEqual(prep.context_relation('r','s',original,native),'reviewed_single_ASCII_space_difference')
            for rid,sid,a,b in [('other','s',original,native),('r','other',original,native),('r','s',original,native+' ')]:
                with self.assertRaises(RunConflict):prep.context_relation(rid,sid,a,b)
        self.assertEqual(original,'First.\n\nNext.')

    def test_bad_parquet_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wrong.parquet';path.write_bytes(b'not the pinned dataset')
            with self.assertRaises(RunConflict):prep.read_test(path)

    def test_component_closure_transitive_across_sources(self):
        sources={'a':{'task_type':'QA','source_info':{'passages':'passage 1: A passage 2: B'}},
                 'b':{'task_type':'QA','source_info':{'passages':'passage 1: B passage 2: C'}},
                 'c':{'task_type':'QA','source_info':{'passages':'passage 1: C passage 2: D'}}}
        self.assertEqual(len(set(prep.component_map(sources).values())),1)


class InventoryTests(unittest.TestCase):
    def test_index_matches_do_not_certify_text_identity(self):
        bundle=build(fixture());expected=bundle['manifest']['offline_rows']
        records=[{'idx':r['test_index'],'ground_truth_hallucination':bool(r['label']),'signal4_score':.4} for r in expected]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'signal4_results_test.json'
            self.assertEqual(inv.inspect_cache(path,expected,'signal4_score',True)['status'],'missing')
            path.write_text(json.dumps(records));before=path.read_bytes()
            report=inv.inspect_cache(path,expected,'signal4_score',True)
            self.assertEqual(report['status'],'index_consistent_content_provenance_unproven')
            self.assertFalse(report['comparison_ready']);self.assertEqual(path.read_bytes(),before)
            for record,row in zip(records,expected):record['input_sha256']=row['input_sha256']
            path.write_text(json.dumps(records));report=inv.inspect_cache(path,expected,'signal4_score',True)
            self.assertEqual(report['rows_with_matching_input_hash'],2);self.assertFalse(report['comparison_ready'])

    def test_duplicate_label_score_and_hash_errors_and_missingness(self):
        expected=build(fixture())['manifest']['offline_rows']
        for kind in ('duplicate','wrong_label','boolean_score','bad_hash','missing_score','missing_row'):
            records=[{'idx':r['test_index'],'ground_truth_hallucination':r['label'],'signal4_score':.4} for r in expected]
            if kind=='duplicate':records.append(deepcopy(records[0]))
            if kind=='wrong_label':records[0]['ground_truth_hallucination']=1
            if kind=='boolean_score':records[0]['signal4_score']=True
            if kind=='bad_hash':records[0]['input_sha256']='wrong'
            if kind=='missing_score':records[0]['signal4_score']=None
            if kind=='missing_row':records.pop()
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'cache.json';path.write_text(json.dumps(records))
                report=inv.inspect_cache(path,expected,'signal4_score',True)
            with self.subTest(kind=kind):
                self.assertEqual(report['status'],'incomplete' if kind.startswith('missing') else 'invalid_cache')
                self.assertFalse(report['comparison_ready'])


if __name__=='__main__':unittest.main()
