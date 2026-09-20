"""Label-blind native split metadata and exact-evidence exclusion regressions."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import cross_split_audit as c
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.prepare_pilot import group_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    sources={str(i):{'source_id':str(i),'task_type':'QA','source':'MARCO',
                    'source_info':{'passages':f'passage 1:Unique passage {i}.'}} for i in range(4)}
    metadata=[{'id':str(i),'source_id':str(i),'split':'train' if i<3 else 'test'} for i in range(4)]
    prior={'proposed_exclusion_ids':['0'],'train_mapping':[
        {'sample_id':str(i),'source_id':str(i),'original_partition':'pilot' if i==0 else 'threshold_candidate',
         'context_group_sha256':group_hash(c.context_from_native(sources[str(i)]))} for i in range(3)]}
    return prior,metadata,sources


def refresh(prior,sources):
    for row in prior['train_mapping']:
        row['context_group_sha256']=group_hash(c.context_from_native(sources[row['source_id']]))


class CrossSplitTests(unittest.TestCase):
    def test_projection_never_reads_answers_or_labels(self):
        _,metadata,sources=fixture()
        for row in metadata:row.update(response=object(),labels=object(),quality=object(),model=object())
        projected,selected=c.project_metadata(metadata,sources.values())
        self.assertTrue(all(set(r)=={'id','source_id','split'} for r in projected))
        self.assertTrue(all('prompt' not in r for r in selected.values()))
    def test_duplicate_ids_unknown_split_and_missing_sources_fail(self):
        _,metadata,sources=fixture()
        with self.assertRaisesRegex(RunConflict,'duplicate'):c.project_metadata(metadata*2,sources.values())
        bad=deepcopy(metadata);bad[0]['split']='validation'
        with self.assertRaises(RunConflict):c.project_metadata(bad,sources.values())
        with self.assertRaisesRegex(RunConflict,'missing'):c.project_metadata(metadata,[])
    def test_no_overlap_preserves_original_exclusions(self):
        prior,metadata,sources=fixture();before=deepcopy(prior)
        result=c.audit(prior,metadata,sources)
        self.assertEqual(result['proposed_exclusion_ids'],['0'])
        self.assertEqual(result['summary']['additional_test_linked_candidate_rows'],0)
        self.assertEqual(result['summary']['remaining_candidate_rows'],2)
        self.assertEqual(prior,before)
    def test_shared_passage_without_full_context_match_excludes_candidate(self):
        prior,metadata,sources=fixture()
        sources['1']['source_info']['passages']='passage 1:Shared evidence.\n\npassage 2:Train-only addition.'
        sources['3']['source_info']['passages']='passage 1:Shared evidence.\n\npassage 2:Test-only addition.'
        refresh(prior,sources);result=c.audit(prior,metadata,sources)
        self.assertEqual(result['summary']['native_sources_shared_between_splits'],0)
        self.assertEqual(result['summary']['cross_split_full_context_groups'],0)
        self.assertEqual(result['summary']['cross_split_exact_evidence_indicators'],1)
        self.assertEqual(result['additional_test_linked_candidate_ids'],['1'])
        self.assertEqual(result['proposed_exclusion_ids'],['0','1'])
    def test_transitive_links_through_test_expand_pilot_and_candidate_components(self):
        prior,metadata,sources=fixture()
        sources['0']['source_info']['passages']='passage 1:First bridge.'
        sources['1']['source_info']['passages']='passage 1:Second bridge.'
        sources['3']['source_info']['passages']='passage 1:First bridge.\n\npassage 2:Second bridge.'
        refresh(prior,sources);result=c.audit(prior,metadata,sources)
        self.assertEqual(result['summary']['pilot_train_rows_linked_to_native_test'],1)
        self.assertEqual(result['additional_test_linked_candidate_ids'],['1'])
        self.assertEqual(result['additional_pilot_linked_candidate_ids'],['1'])
        self.assertEqual(result['summary']['proposed_total_excluded_rows'],2) # Not double counted.
    def test_shared_native_id_is_not_assumed_disjoint(self):
        prior,metadata,sources=fixture();metadata[-1]['source_id']='1';sources.pop('3')
        metadata,sources=c.project_metadata(metadata,sources.values())
        result=c.audit(prior,metadata,sources)
        self.assertEqual(result['summary']['native_sources_shared_between_splits'],1)
        self.assertEqual(result['additional_test_linked_candidate_ids'],['1'])
    def test_train_membership_context_and_source_drift_fail(self):
        prior,metadata,sources=fixture()
        with self.assertRaisesRegex(RunConflict,'membership'):c.audit(prior,metadata[1:],sources)
        changed=deepcopy(sources);changed['1']['source_info']['passages']='passage 1:Changed.'
        with self.assertRaisesRegex(RunConflict,'source/context'):c.audit(prior,metadata,changed)
        changed=deepcopy(metadata);changed[1]['source_id']='2'
        with self.assertRaisesRegex(RunConflict,'source/context'):c.audit(prior,changed,sources)
    def test_pinned_prior_checksum_and_revision_checked(self):
        report={'version':c.TRAIN_VERSION,'code_revision':c.TRAIN_REPORT_REVISION,
                'native_files':c.NATIVE_FILES,'native_revision':c.NATIVE_REVISION,'policy':c.POLICY}
        digest=content_hash(report);bundle={'report_sha256':digest,'report':report}
        with patch.object(c,'TRAIN_REPORT_SHA256',digest):
            self.assertEqual(c.validate_train_report(bundle),report)
            changed=deepcopy(bundle);changed['report']['code_revision']='other'
            with self.assertRaises(RunConflict):c.validate_train_report(changed)


if __name__=='__main__':unittest.main()
