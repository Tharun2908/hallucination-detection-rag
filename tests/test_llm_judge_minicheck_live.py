import copy
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge.minicheck_inputs import document_chunks, prepare
from post_thesis.llm_judge.check_minicheck_live import execute_once, score_outputs
from post_thesis.llm_judge import check_minicheck_environment as environment
from post_thesis.llm_judge import check_minicheck_live as live
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


class Tokenizer:
    def __call__(self,text,**kwargs): return {'input_ids':list(text)}
    def encode(self,text): return [1]+[ord(c) for c in text]
    def apply_chat_template(self,messages,**kwargs): return '<s>'+repr(messages)+'assistant'


def split(text): return [s for s in re.split(r'(?<=[.!?])\s+',text) if s]


def responses(prepared):
    import math
    result=[]
    for ids,p in zip(prepared['prompt_token_ids'],[.8,.1,.8,.7,.8,.1]):
        probs={0:NS(decoded_token='Yes',logprob=math.log(p))}
        for i in range(1,5): probs[i]=NS(decoded_token='No' if i==1 else 'other',logprob=-10.)
        result.append(NS(prompt_token_ids=ids,outputs=[NS(token_ids=[0],logprobs=[probs],text='Yes',finish_reason='length')]))
    return result


class MiniCheckLiveTests(unittest.TestCase):
    def test_runtime_disables_sampling_separately_from_attention(self):
        with patch.dict(os.environ, {'VLLM_USE_FLASHINFER_SAMPLER':'1'}):
            recorded = live.configure_runtime()
            self.assertEqual(os.environ['VLLM_USE_FLASHINFER_SAMPLER'],'0')
            self.assertEqual(recorded['VLLM_USE_FLASHINFER_SAMPLER'],'0')
            self.assertEqual(live.ENGINE['attention_backend'],'FLASH_ATTN')
            self.assertEqual(live.ENGINE['logprobs_mode'],'raw_logprobs')

    def test_failed_predecessor_read_only_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'
            report={'result':{'status':'error','error':'artificial startup failure'}}
            digest=content_hash(report)
            path.write_text(json.dumps({'report':report,'report_sha256':digest}),encoding='utf-8')
            original=path.read_bytes()
            with patch.object(live,'PREVIOUS_REPORT_SHA256',digest):
                self.assertEqual(live.failed_predecessor(path)['report_sha256'],digest)
                self.assertEqual(path.read_bytes(),original)
                report['result']['status']='ok'
                path.write_text(json.dumps({'report':report,'report_sha256':digest}),encoding='utf-8')
                with self.assertRaises(RunConflict): live.failed_predecessor(path)
            self.assertNotEqual(live.RUN_ID,live.PREVIOUS_RUN_ID)

    def test_preparation_and_chunk_boundaries(self):
        t=Tokenizer(); p=prepare(t,split)
        self.assertEqual(len(p['texts']),6)
        self.assertEqual([(r['start'],r['end']) for r in p['rows']],[(0,1),(1,2),(2,4),(4,6)])
        self.assertEqual(document_chunks('First.\nSecond.',t,split,size=100),['First.\nSecond.'])
        self.assertEqual(document_chunks('First. Second.',t,split,size=7),['First.','Second.'])
        self.assertEqual(document_chunks('',t,split),[''])
        # A single over-budget sentence is not silently truncated by upstream chunking.
        self.assertEqual(document_chunks('Long sentence.',t,split,size=2),['Long sentence.'])

    def test_response_aggregation_preserves_min_of_max(self):
        p=prepare(Tokenizer(),split); r=score_outputs(responses(p),p)
        self.assertEqual(r['requested_generations'],6)
        self.assertAlmostEqual(r['rows'][2]['support_score'],.7)
        self.assertAlmostEqual(r['rows'][3]['support_score'],.1)
        self.assertTrue(all(row['upstream_label_matches_expectation'] for row in r['rows']))

    def test_prompt_mismatch_missing_logprobs_and_missing_outputs_rejected(self):
        p=prepare(Tokenizer(),split)
        a=responses(p); a[0].prompt_token_ids=[999]
        b=responses(p); b[0].outputs[0].logprobs=None
        c=responses(p); c[0].outputs[0].logprobs=[{}]
        d=responses(p); d[0].outputs[0].token_ids=[]
        for result in (a,b,c,d,responses(p)[:-1]):
            with self.assertRaises(RunConflict): score_outputs(result,p)

    def test_complete_replay_never_calls_model_again(self):
        with tempfile.TemporaryDirectory() as d:
            compute=Mock(return_value={'status':'ok','requested_generations':6})
            first,new=execute_once(Path(d),{'artificial':1},compute)
            self.assertTrue(new)
            replay,new=execute_once(Path(d),{'artificial':1},compute)
            self.assertFalse(new);self.assertEqual(first,replay);self.assertEqual(compute.call_count,1)
            with self.assertRaises(RunConflict): execute_once(Path(d),{'changed':1},compute)
            self.assertEqual(compute.call_count,1)

    def test_failure_and_interruption_do_not_repeat_generation(self):
        for exception in (RuntimeError('test failure'),KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as d:
                compute=Mock(side_effect=exception)
                if isinstance(exception,KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt): execute_once(Path(d),{'artificial':1},compute)
                else:
                    first,_=execute_once(Path(d),{'artificial':1},compute)
                    self.assertEqual(first['report']['result']['status'],'error')
                replay,new=execute_once(Path(d),{'artificial':1},compute)
                self.assertFalse(new);self.assertEqual(compute.call_count,1)
                self.assertIn(replay['report']['result']['status'],('error','interrupted'))

    def test_environment_report_creates_its_own_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(environment,'collect',return_value={'synthetic':True}), \
                 patch.object(environment,'code_revision',return_value='artificial'), \
                 patch.object(environment,'run_directory',side_effect=lambda name:Path(d)/name), \
                 patch('sys.argv',['check_minicheck_environment']):
                self.assertEqual(environment.main(),0)
            self.assertEqual(len(list(Path(d).glob('minicheck-environment-*/report.json'))),1)


if __name__=='__main__': unittest.main()
