"""Native TRAIN mapping and conservative overlap closure; no downloads or models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import source_audit as audit
from post_thesis.llm_judge.prepare_pilot import build_bundle
from post_thesis.llm_judge.storage import RunConflict


def fixtures():
    sources={str(i):{'source_id':str(i),'task_type':'QA','source':'MARCO',
                    'source_info':{'question':'Question','passages':f'passage 1:Unique source {i}.\n\n'}} for i in range(6)}
    responses={};rows=[]
    for sid,s in sources.items():
        for n in range(2):
            rid=sid+'-'+str(n)
            responses[rid]={'id':rid,'source_id':sid,'response':'Answer '+rid,'model':'fixture','quality':'good','split':'train'}
            rows.append({'id':rid,'output':responses[rid]['response'],'context':audit.context_from_native(s),
                         'model':'fixture','quality':'good','task_type':'QA',
                         'hallucination_labels_processed':{'evident_conflict':n,'baseless_info':0}})
    return rows,responses,sources


class SourceAuditTests(unittest.TestCase):
    def run_core(self,rows,responses,sources):
        bundle=build_bundle(rows,revision=audit.PREPARATION_REVISION,pilot_size=1)
        original=deepcopy(bundle)
        build=audit.build_bundle
        with patch.object(audit,'MANIFEST_SHA256',bundle['manifest_sha256']), \
             patch.object(audit,'build_bundle',side_effect=lambda r,revision:build(r,revision=revision,pilot_size=1)):
            result=audit.audit(rows,bundle,responses,sources)
        self.assertEqual(bundle,original)
        return result
    def test_native_projection_ignores_nontrain_payload_and_annotations(self):
        rows,responses,sources=fixtures()
        native=list(responses.values())+[{'id':'test-not-selected'}]
        for r in native[:-1]:r['labels']=object()  # Must never be read or serialized.
        result,selected,ignored=audit.project_native(native,list(sources.values())+[{'source_id':'test-source'}],{r['id'] for r in rows})
        self.assertEqual(len(result),12)
        self.assertNotIn('labels',result['0-0'])
        self.assertEqual(ignored,{'native_response_records_ignored':1,'native_source_records_ignored':1})
    def test_nontrain_mapping_missing_and_duplicate_ids_fail(self):
        rows,responses,sources=fixtures();native=list(responses.values());ids={r['id'] for r in rows}
        bad=deepcopy(native);bad[0]['split']='test'
        with self.assertRaisesRegex(RunConflict,'non-TRAIN'):audit.project_native(bad,sources.values(),ids)
        with self.assertRaisesRegex(RunConflict,'missing'):audit.project_native(native[1:],sources.values(),ids)
        with self.assertRaisesRegex(RunConflict,'duplicate'):audit.project_native(native+[native[0]],sources.values(),ids)
        with self.assertRaisesRegex(RunConflict,'duplicate'):audit.project_native(native,list(sources.values())*2,ids)
        with self.assertRaisesRegex(RunConflict,'missing'):audit.project_native(native,[],ids)
    def test_context_reconstruction_preserves_export_suffixes(self):
        self.assertEqual(audit.context_from_native({'task_type':'Summary','source_info':'News\n'}),'News\n\noutput:')
        self.assertEqual(audit.context_from_native({'task_type':'Summary','source_info':'News'}),'News\noutput:')
        self.assertEqual(audit.context_from_native({'task_type':'Data2txt','source_info':{'available':None}}),"\n{'available': None}\nOverview:")
        with self.assertRaises(RunConflict):audit.indicators({'task_type':'QA','source_info':{'passages':'No numbered prefix'}})
    def test_changed_answer_context_and_task_fail_provenance(self):
        for field in ('output','context','task_type','model','quality'):
            rows,responses,sources=fixtures();rows[0][field]='changed'
            with self.subTest(field=field),self.assertRaisesRegex(RunConflict,'provenance'):
                self.run_core(rows,responses,sources)
    def test_transitive_shared_passages_expand_pilot_exclusions(self):
        rows,responses,sources=fixtures()
        # Connect every source through shared passages without making contexts equal.
        for i,s in enumerate(sources.values()):
            s['source_info']['passages']=f'passage 1:Shared {i}.\n\npassage 2:Shared {i+1}.\n\n'
        for r in rows:r['context']=audit.context_from_native(sources[responses[r['id']]['source_id']])
        result=self.run_core(rows,responses,sources)
        self.assertEqual(result['summary']['connected_components'],1)
        self.assertEqual(result['summary']['additional_pilot_linked_candidate_rows'],10)
        self.assertEqual(result['summary']['remaining_candidate_rows'],0)
        self.assertFalse(result['summary']['threshold_subset_selected'])
    def test_grouping_does_not_depend_on_labels(self):
        rows,responses,sources=fixtures();first=self.run_core(rows,responses,sources)
        for r in rows:r['hallucination_labels_processed']={'evident_conflict':7,'baseless_info':4}
        second=self.run_core(rows,responses,sources)
        self.assertEqual(first,second)
    def test_missing_business_identity_is_flagged_not_invented(self):
        source={'task_type':'Data2txt','source_info':{'name':'Cafe','city':'Berlin','state':'BE','address':''}}
        self.assertEqual(audit.indicators(source),set())
        source['source_info']['address']='Main Street 1'
        first=audit.indicators(source)
        source['source_info']['name']=' CAFE '
        self.assertEqual(first,audit.indicators(source))
    def test_native_checksums_fail_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.jsonl';data=b'{"id":"x"}\n';path.write_bytes(data)
            pins={'sha256':hashlib.sha256(data).hexdigest(),
                  'git_blob_sha1':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()}
            self.assertEqual(list(audit.verified_lines(path,pins)),[{'id':'x'}])
            path.write_bytes(b'not JSON')
            with self.assertRaisesRegex(RunConflict,'checksum'):list(audit.verified_lines(path,pins))


if __name__=='__main__':unittest.main()
