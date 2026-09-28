import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.minicheck_inputs import prepare_item, prepare
from post_thesis.llm_judge.minicheck_inference import reference, decode_outputs, aggregate_responses, validate_result
from post_thesis.llm_judge.run_minicheck_halubench import execute, workload, PLAN
from post_thesis.llm_judge.storage import RunConflict
from tests.test_llm_judge_minicheck_live import Tokenizer, split


def inputs():
    examples=[Example(str(i),JudgeInput(answer=a,context='First. Second.'))
              for i,a in enumerate(['First.','First. Wrong.'])]
    refs=[reference(ex,i,prepare_item(ex.item.answer,ex.item.context,Tokenizer(),split))
          for i,ex in enumerate(examples)]
    return examples,refs


def good_result(ref):
    raw=[]
    for i,prompt in enumerate(ref['prompts']):
        value=.9 if i==0 else .05
        tokens=[{'token_id':0,'decoded_token':'Yes','logprob':math.log(value)}]
        tokens += [{'token_id':i,'decoded_token':'No' if i==1 else 'other','logprob':-10.}
                   for i in range(1,5)]
        raw.append({'prompt_token_ids_sha256':prompt['token_ids_sha256'],
                    'emitted_token_id':0,'text':'Yes','finish_reason':'length', 'returned_logprobs':tokens})
    return {'status':'ok','raw_responses':raw,**aggregate_responses(raw,ref),
            'example_seconds':.25,'runtime':{'artificial':True}}


class MiniCheckHaluBenchExecution(unittest.TestCase):
    def test_preparation_refactor_preserves_synthetic_texts_and_ids(self):
        t=Tokenizer(); p=prepare(t,split)
        for row in p['rows']:
            item=prepare_item(row['answer'],row['context'],t,split)
            self.assertEqual(item['texts'],p['texts'][row['start']:row['end']])
            self.assertEqual(item['prompt_token_ids'],p['prompt_token_ids'][row['start']:row['end']])
        self.assertEqual(len(p['texts']),6)
        self.assertNotIn('expected_supported',repr(p['texts']))

    def test_full_response_decode_and_multichunk_aggregation(self):
        ex=Example('a',JudgeInput(answer='First. Wrong.',context='First. Second.'))
        item=prepare_item(ex.item.answer,ex.item.context,Tokenizer(),split)
        ref=reference(ex,0,item)
        outputs=[]
        for ids,p in zip(item['prompt_token_ids'],[.9,.05]):
            probs={0:NS(decoded_token='Yes',logprob=math.log(p))}
            probs.update({i:NS(decoded_token='other',logprob=-10.) for i in range(1,5)})
            outputs.append(NS(prompt_token_ids=ids,outputs=[NS(token_ids=[0],logprobs=[probs],text='Yes',finish_reason='length')]))
        raw=decode_outputs(outputs,ref)
        result=aggregate_responses(raw,ref)
        self.assertAlmostEqual(result['support_score'],.05)
        self.assertTrue(result['frozen_predicted_unsupported'])
        # Two chunks: maxima [.9,.8], then min .8. Do not take minimum over all pairs.
        ref['context_chunks'].append('Third.');ref['prompts']*=2
        extra=copy.deepcopy(raw)
        extra[0]['returned_logprobs'][0]['logprob']=math.log(.1)
        extra[1]['returned_logprobs'][0]['logprob']=math.log(.8)
        self.assertAlmostEqual(aggregate_responses(raw+extra,ref)['support_score'],.8)

    def test_missing_yes_is_zero_but_missing_or_invalid_output_fails(self):
        _,refs=inputs();ref=refs[0];result=good_result(ref)
        raw=result['raw_responses']
        raw[0]['returned_logprobs'][0]['decoded_token']=' YES'
        self.assertEqual(aggregate_responses(raw,ref)['support_score'],0.)
        for transform in ('empty','hash','nan','duplicate'):
            bad=copy.deepcopy(raw)
            if transform=='empty': bad=[]
            elif transform=='hash': bad[0]['prompt_token_ids_sha256']='changed'
            elif transform=='nan': bad[0]['returned_logprobs'][0]['logprob']=float('nan')
            else: bad[0]['returned_logprobs'][1]['token_id']=0
            with self.assertRaises(RunConflict): aggregate_responses(bad,ref)
        wrong=good_result(ref);wrong['support_score']=.123
        with self.assertRaises(RunConflict): validate_result(wrong,ref)

    def test_length_and_count_limits_before_model_loading(self):
        examples,refs=inputs()
        bad=copy.deepcopy(refs);bad[0]['prompts'][0]['input_tokens']=32768
        with tempfile.TemporaryDirectory() as d:
            factory=Mock()
            with self.assertRaises(RunConflict):
                execute(examples,bad,directory=Path(d),identity={},backend_factory=factory)
            factory.assert_not_called()
        with patch.dict(PLAN,{'maximum_total_input_tokens':1}):
            with self.assertRaises(RunConflict): workload(refs)
        with patch.dict(PLAN,{'maximum_sentence_requests':1}):
            with self.assertRaises(RunConflict): workload(refs)

    def test_partial_resume_and_final_replay_without_model_loading(self):
        examples,refs=inputs()
        with tempfile.TemporaryDirectory() as d:
            backend=Mock();backend.score.side_effect=lambda ex,ref:good_result(ref)
            factory=Mock(return_value=backend)
            first,new=execute(examples,refs,directory=Path(d),identity={'artificial':1},backend_factory=factory,max_new_examples=1)
            self.assertEqual((new,first['report']['valid_scores'],first['report']['pending']),(1,1,1))
            second,new=execute(examples,refs,directory=Path(d),identity={'artificial':1},backend_factory=factory)
            self.assertEqual((new,second['report']['valid_scores'],second['report']['pending']),(1,2,0))
            self.assertEqual(second['report']['known_token_totals']['output_tokens'],3)
            factory.reset_mock()
            replay,new=execute(examples,refs,directory=Path(d),identity={'artificial':1},backend_factory=factory,max_new_examples=0)
            self.assertEqual(new,0);self.assertEqual(replay,second);factory.assert_not_called()
            with self.assertRaises(RunConflict):
                execute(examples,refs,directory=Path(d),identity={'changed':1},backend_factory=factory)
            factory.assert_not_called()

    def test_error_and_interruption_are_terminal_and_usage_unknown(self):
        examples,refs=inputs()
        for failure in (ValueError('bad transport'),KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as d:
                backend=Mock();backend.score.side_effect=failure;factory=Mock(return_value=backend)
                if isinstance(failure,KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                else: execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                factory.reset_mock()
                replay,new=execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                self.assertEqual(new,0);factory.assert_not_called()
                self.assertEqual(replay['report']['terminal_failures'],1)
                self.assertEqual(replay['report']['examples_with_unknown_usage'],1)
                self.assertEqual(replay['report']['pending'],1)

    def test_initialization_failure_or_interruption_is_preserved(self):
        examples,refs=inputs()
        for failure in (RuntimeError('startup failed'),KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as d:
                factory=Mock(side_effect=failure)
                if isinstance(failure,KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                else: execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                factory.reset_mock()
                result,new=execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
                self.assertTrue(result['report']['initialization_failure'])
                self.assertEqual(result['report']['model_initializations'],1)
                self.assertEqual(new,0);factory.assert_not_called()

    def test_zero_allowance_does_not_initialize(self):
        examples,refs=inputs()
        with tempfile.TemporaryDirectory() as d:
            factory=Mock()
            with patch.dict(PLAN,{'maximum_client_seconds':0}):
                result,new=execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
            self.assertEqual(new,0);factory.assert_not_called()
            self.assertEqual(result['report']['pending'],2)


class HaluBenchContractTests(unittest.TestCase):
    def test_byte_pins_and_fixed_transfer_contract(self):
        from post_thesis.llm_judge import run_minicheck_halubench as run
        self.assertEqual(run.verify_implementation(),run.IMPLEMENTATION_FILES)
        self.assertEqual(run.PLAN['examples'],8000)
        self.assertFalse(run.PLAN['adaptation_training'])
        self.assertEqual(run.PLAN['TRAIN_threshold']['threshold_hex'],'0x1.999999999999bp-3')
        with patch.object(run,'file_sha256',return_value='changed'):
            with self.assertRaises(RunConflict): run.verify_implementation()

    def test_report_scope_and_noop_replay_bytes(self):
        from post_thesis.llm_judge import run_minicheck_halubench as run
        examples,refs=inputs()
        with tempfile.TemporaryDirectory() as d:
            factory=Mock();factory.return_value.score.side_effect=lambda ex,ref:good_result(ref)
            result,_=run.execute(examples,refs,directory=Path(d),identity={},backend_factory=factory)
            self.assertEqual(result['report']['run_id'],run.RUN_ID)
            self.assertTrue(result['report']['HaluBench_read'])
            self.assertFalse(result['report']['adaptation_training'])
            self.assertFalse(result['report']['TEST_label_metrics_computed'])
            path=Path(d)/'summary.json'; before=path.read_bytes();mtime=path.stat().st_mtime_ns
            factory.reset_mock()
            again,new=run.execute(examples,refs,directory=Path(d),identity={},backend_factory=factory,max_new_examples=0)
            self.assertEqual((new,again),(0,result));factory.assert_not_called()
            self.assertEqual(path.read_bytes(),before);self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_complete_fusion_alignment_and_tampering(self):
        from dataclasses import asdict
        from post_thesis.llm_judge import run_minicheck_halubench as run
        from post_thesis.llm_judge.prompts import content_hash
        examples=[Example(str(i),JudgeInput(answer='A',context='C')) for i in range(8000)]
        report={'run_id':run.FUSION_RUN_ID,'TEST_manifest_sha256':run.TEST_MANIFEST_SHA256,
                'judge_freeze_sha256':run.FREEZE_SHA256,'valid_scores':8000,'adaptation_training':False,
                'predictions':[{'sample_id':ex.sample_id,'test_index':i,'status':'ok',
                                'input_sha256':content_hash(asdict(ex.item))} for i,ex in enumerate(examples)]}
        with patch.object(run,'checked_bundle',return_value=report):
            self.assertEqual(run.verify_fusion(examples),run.FUSION_SHA256)
            report['predictions'][7999]['input_sha256']='changed'
            with self.assertRaises(RunConflict): run.verify_fusion(examples)


if __name__=='__main__': unittest.main()
