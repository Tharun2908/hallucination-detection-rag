"""S2 aggregation, exact sentence projection and durable replay; no inference."""
import ast
from dataclasses import asdict
from post_thesis.llm_judge.prompts import content_hash
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge import run_s2_halubench as run
from post_thesis.llm_judge.s2_inference import aggregate, reference, sentences, split_sentences


def examples(n=3):
    return [Example(str(i), JudgeInput('This is answer number '+str(i)+'.',
                                      'This is one context sentence. This is another context sentence.')) for i in range(n)]


def fake_result(example):
    answers, contexts = sentences(example)
    batches = [{'answer_index': a, 'context_start': start,
                'logits': [-2.] * len(contexts[start:start+32]), 'forward_tensors_sha256': 'a'*64,
                'coverage': [{'answer': [6,6], 'context': [8,8]} for _ in contexts[start:start+32]]}
               for a in range(len(answers)) for start in range(0,len(contexts),32)]
    return {'status': 'ok', 'pair_batches': batches, **aggregate(batches,len(answers),len(contexts)),
            'example_seconds': .2, 'cuda_forward_seconds': .1, 'runtime': {'fake': True}}


class AggregationTests(unittest.TestCase):
    def test_min_of_context_maxima_preserves_negative_raw_logits(self):
        batches = [{'answer_index':0,'context_start':0,'logits':[-8.]*32},
                   {'answer_index':0,'context_start':32,'logits':[4.]},
                   {'answer_index':1,'context_start':0,'logits':[-3.]*32},
                   {'answer_index':1,'context_start':32,'logits':[-5.]}]
        result = aggregate(batches,2,33)
        self.assertEqual(result, {'raw_min_relevance':-3., 'raw_mean_relevance':.5,
                                  'per_answer_best_logits':[4.,-3.]})
        for bad in (batches[:-1],list(reversed(batches)),batches+[batches[-1]]):
            with self.assertRaises(RunConflict): aggregate(bad,2,33)

    def test_empty_original_policy_and_nonfinite_pairs(self):
        for a,c in ((0,1),(1,0),(0,0)):
            self.assertEqual(aggregate([],a,c), {'raw_min_relevance':0.,'raw_mean_relevance':0.,'per_answer_best_logits':[]})
        for value in (float('nan'),float('inf'),True):
            with self.assertRaises(RunConflict):
                aggregate([{'answer_index':0,'context_start':0,'logits':[value]}],1,1)

    def test_splitter_matches_original_without_importing_training_script(self):
        source = Path(__file__).resolve().parents[1] / 'signals/relevance_verifier_full_v2.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n,ast.FunctionDef) and n.name=='split_into_sentences')
        namespace={'re':re}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<original splitter only>','exec'),namespace)
        for text in ('',None,'Short. A sufficiently long sentence!',
                     'First complete sentence.\nSecond complete sentence?',
                     'Price is 2.50 euros. Dr. Smith arrived today.',
                     "{'OutdoorSeating': None, 'WiFi': 'no'}",'  Unicode café sentence.\r\nAnother café sentence. '):
            self.assertEqual(split_sentences(text),namespace['split_into_sentences'](text))

    def test_forged_aggregate_and_missing_visibility_are_rejected(self):
        ex=examples(1)[0]; ref=reference(ex,0); good=fake_result(ex)
        run.validate_result(good,ref)
        bad=copy.deepcopy(good); bad['raw_min_relevance']=.2
        with self.assertRaises(RunConflict): run.validate_result(bad,ref)
        bad=copy.deepcopy(good); bad['pair_batches'][0]['coverage']=[]
        with self.assertRaises(RunConflict): run.validate_result(bad,ref)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.backend=Mock(); self.backend.score.side_effect=fake_result
        self.factory=Mock(return_value=self.backend)
    def execute(self, items=None, **kwargs):
        return run.execute(examples() if items is None else items,directory=self.directory,
                           identity={'synthetic':True},backend_factory=self.factory,**kwargs)

    def test_replay_avoids_model_loading_and_keeps_identical_hash(self):
        first,new=self.execute()
        self.assertEqual((new,first['report']['valid_scores'],first['report']['completed_pairs']),(3,3,6))
        self.assertEqual(first['report']['run_id'],'s2-halubench-test-fresh-v1')
        self.assertTrue(first['report']['HaluBench_read'])
        self.assertFalse(first['report']['adaptation_training'])
        self.assertFalse(first['report']['TEST_label_metrics_computed'])
        self.factory.reset_mock(); self.backend.score.reset_mock()
        replay,new=self.execute()
        self.assertEqual((replay,new),(first,0))
        self.factory.assert_not_called(); self.backend.score.assert_not_called()

    def test_empty_short_answer_uses_original_zero_feature_without_pairs(self):
        ex=Example('short',JudgeInput('Yes.','This is a complete context sentence.'))
        final,new=self.execute([ex])
        self.assertEqual((new,final['report']['valid_scores'],final['report']['completed_pairs']),(1,1,0))
        self.assertEqual(final['report']['empty_pair_examples'],1)
        self.assertEqual(final['report']['predictions'][0]['raw_min_relevance'],0.)
        self.assertEqual(final['report']['predictions'][0]['per_answer_best_logits'],[])

    def test_fixed_runtime_and_s4_alignment(self):
        run.verify_implementation()
        self.assertEqual((run.PLAN['examples'],run.PLAN['sentence_pairs'],run.PLAN['maximum_forward_batches']),(8000,81107,8327))
        items=examples()
        report={'run_id':run.S4_RUN_ID,'identity':{'code_revision':run.S4_REVISION,
                'TEST_manifest_sha256':run.TEST_MANIFEST_SHA256,'judge_freeze_sha256':run.FREEZE_SHA256},
                'examples_total':3,'valid_scores':3,'terminal_failures':0,'pending':0,
                'HaluBench_read':True,'adaptation_training':False,
                'predictions':[{'sample_id':ex.sample_id,'test_index':i,'status':'ok',
                    'input_sha256':content_hash(asdict(ex.item))} for i,ex in enumerate(items)]}
        def verify(value):
            digest=content_hash(value)
            with patch.object(run,'S4_REPORT_SHA256',digest):
                return run.validate_s4({'report':value,'report_sha256':digest},items)
        self.assertEqual(verify(report),content_hash(report))
        for kind in ('hash','order','pending','revision','training'):
            bad=copy.deepcopy(report)
            if kind=='hash':bad['predictions'][0]['input_sha256']='wrong'
            if kind=='order':bad['predictions'].reverse()
            if kind=='pending':bad['pending']=1
            if kind=='revision':bad['identity']['code_revision']='wrong'
            if kind=='training':bad['adaptation_training']=True
            with self.subTest(kind=kind),self.assertRaises(RunConflict):verify(bad)

    def test_partial_resume_preserves_previously_committed_examples(self):
        first,new=self.execute(max_new_examples=1)
        self.assertEqual((new,first['report']['pending']),(1,2))
        self.backend.score.reset_mock()
        final,new=self.execute()
        self.assertEqual((new,final['report']['valid_scores']),(2,3))
        self.assertEqual([c.args[0].sample_id for c in self.backend.score.call_args_list],['1','2'])

    def test_interruption_is_missing_and_not_retried(self):
        self.backend.score.side_effect=KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt): self.execute()
        self.backend.score.side_effect=fake_result
        final,new=self.execute()
        self.assertEqual((new,final['report']['terminal_failures'],final['report']['valid_scores']),(2,1,2))
        self.assertIsNone(final['report']['predictions'][0]['raw_min_relevance'])

    def test_model_failure_stops_invocation_and_does_not_fabricate_zero(self):
        self.backend.score.side_effect=RuntimeError('OOM')
        final,new=self.execute()
        self.assertEqual((new,final['report']['terminal_failures'],final['report']['pending']),(1,1,2))
        self.assertIsNone(final['report']['predictions'][0]['raw_min_relevance'])

    def test_changed_text_or_order_blocks_before_loading(self):
        self.execute(max_new_examples=1); self.factory.reset_mock()
        changed=[Example('0',JudgeInput('Different answer text.','Different context text.'))]+examples()[1:]
        for items in (changed,list(reversed(examples()))):
            with self.assertRaises(RunConflict): self.execute(items)
        self.factory.assert_not_called()

    def test_zero_and_invalid_budgets(self):
        result,new=self.execute(max_new_examples=0)
        self.assertEqual((new,result['report']['pending']),(0,3)); self.factory.assert_not_called()
        for limit in (-1,8001,True):
            with self.assertRaises(ValueError): self.execute(max_new_examples=limit)

    def test_derived_report_rebuilt_from_journal(self):
        first,_=self.execute()
        (self.directory/'summary.json').write_text('{}',encoding='utf-8')
        self.factory.reset_mock()
        replay,new=self.execute()
        self.assertEqual((first,0),(replay,new)); self.factory.assert_not_called()


if __name__=='__main__': unittest.main()
