"""HaluBench S4 journal, exact-input and visibility checks; no model/GPU needed."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.storage import RunConflict
from post_thesis.llm_judge import run_s4_halubench as run
from post_thesis.llm_judge.s4_inference import prepare_pair


def examples(n=17):
    return [Example(str(i), JudgeInput(answer='answer ' + str(i), context='context')) for i in range(n)]


def result_for(items):
    return {'status': 'ok', 'batch_seconds': .2, 'cuda_forward_seconds': .1, 'runtime': {'fake': True},
            'rows': [{'sample_id': ex.sample_id, 'input_sha256': content_hash(asdict(ex.item)),
                      'forward_tensors_sha256': 'a' * 64, 'nonpadding_tokens': 7,
                      'coverage': {'answer': {'full_tokens': 2, 'kept_tokens': 2},
                                   'context': {'full_tokens': 2, 'kept_tokens': 2}},
                      'logits': [0., 0.], 'unsupported_score': .5} for ex in items]}


class FakeBackend:
    def __init__(self): self.calls = []
    def score(self, items):
        self.calls.append([ex.sample_id for ex in items])
        return result_for(items)


class HaluBenchS4RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.backend = FakeBackend()
        self.factory = Mock(return_value=self.backend)

    def execute(self, items=None, **kwargs):
        return run.execute(examples() if items is None else items, directory=self.directory,
                           identity={'artificial': True}, backend_factory=self.factory, **kwargs)

    def test_complete_replay_skips_loading_and_preserves_identical_summary(self):
        first, new = self.execute()
        self.assertEqual(new, 2)
        self.assertEqual(first['report']['valid_scores'], 17)
        self.assertEqual(first['report']['run_id'], 's4-halubench-test-fresh-v1')
        self.assertTrue(first['report']['HaluBench_read'])
        self.assertFalse(first['report']['RAGTruth_inputs_read'])
        self.assertFalse(first['report']['adaptation_training'])
        self.assertFalse(first['report']['benchmark_metrics_computed'])
        self.assertEqual([len(c) for c in self.backend.calls], [16, 1])
        raw = (self.directory / 'summary.json').read_bytes()
        self.factory.reset_mock()
        replay, new = self.execute()
        self.assertEqual(new, 0)
        self.factory.assert_not_called()
        self.assertEqual(first, replay)
        self.assertEqual(raw, (self.directory / 'summary.json').read_bytes())

    def test_8000_examples_have_500_fixed_batches_and_frozen_policy(self):
        run.verify_implementation()
        batches=run.batch_plan(examples(8000))
        self.assertEqual(len(batches),500)
        self.assertTrue(all(len(b['examples'])==16 for b in batches))
        self.assertEqual(batches[-1]['references'][-1]['test_index'],7999)
        self.assertEqual(run.PLAN['frozen_historical_threshold'],{'comparator':'>=','value':.55})
        self.assertEqual(run.PLAN['max_length'],512)
        self.assertEqual(run.PLAN['dtype'],'float32')
        self.assertFalse(run.PLAN['tf32'])
        with self.assertRaises(RunConflict): self.execute(examples(8001),max_new_batches=0)
        self.factory.assert_not_called()

    def test_completed_token_audit_identity_is_required(self):
        audit={'status':'completed','identity':{'code_revision':run.TOKEN_AUDIT_REVISION,
               'TEST_manifest_sha256':run.TEST_MANIFEST_SHA256,'judge_freeze_sha256':run.FREEZE_SHA256},
               'summary':{'examples_counted':8000,'total_input_tokens':7794485,'all_inputs_fit':True}}
        def verify(value):
            digest=content_hash(value)
            with patch.object(run,'TOKEN_AUDIT_SHA256',digest):
                return run.validate_token_audit({'audit':value,'audit_sha256':digest})
        self.assertEqual(verify(audit),content_hash(audit))
        for kind in ('status','revision','manifest','count','length','fit'):
            bad=copy.deepcopy(audit)
            if kind=='status':bad['status']='interrupted'
            if kind=='revision':bad['identity']['code_revision']='changed'
            if kind=='manifest':bad['identity']['TEST_manifest_sha256']='changed'
            if kind=='count':bad['summary']['examples_counted']=7999
            if kind=='length':bad['summary']['total_input_tokens']=7794484
            if kind=='fit':bad['summary']['all_inputs_fit']=False
            with self.subTest(kind=kind),self.assertRaises(RunConflict):verify(bad)
        with self.assertRaises(RunConflict):run.validate_token_audit({'audit':audit,'audit_sha256':'bad'})

    def test_partial_resume_uses_original_batch_boundaries(self):
        first, new = self.execute(max_new_batches=1)
        self.assertEqual((new, first['report']['valid_scores'], first['report']['pending']), (1, 16, 1))
        last, new = self.execute()
        self.assertEqual((new, last['report']['valid_scores']), (1, 17))
        self.assertEqual(self.backend.calls[-1], ['16'])

    def test_interrupted_batch_is_terminal_and_never_repeated(self):
        self.backend.score = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt): self.execute()
        self.backend.score = Mock(side_effect=result_for)
        summary, new = self.execute()
        self.assertEqual(new, 1)
        self.assertEqual(summary['report']['terminal_failures'], 16)
        self.assertEqual(summary['report']['valid_scores'], 1)
        self.assertEqual(summary['report']['timing']['unknown_timing_batches'], 1)
        self.assertEqual(self.backend.score.call_args.args[0][0].sample_id, '16')

    def test_backend_failure_is_saved_and_stops_invocation(self):
        self.backend.score = Mock(side_effect=RuntimeError('out of memory'))
        summary, new = self.execute()
        self.assertEqual((new, summary['report']['terminal_failures'], summary['report']['pending']), (1, 16, 1))
        self.backend.score = Mock(side_effect=result_for)
        summary, new = self.execute()
        self.assertEqual((new, summary['report']['valid_scores']), (1, 1))

    def test_reordered_or_changed_input_blocks_before_backend_loading(self):
        self.execute(max_new_batches=1)
        self.factory.reset_mock()
        for items in (list(reversed(examples())),
                      [Example('0', JudgeInput('different answer', 'context'))] + examples()[1:]):
            with self.assertRaises(RunConflict): self.execute(items)
        self.factory.assert_not_called()

    def test_wrong_batch_output_is_failure_not_misaligned_prediction(self):
        def wrong(items):
            result = result_for(items)
            result['rows'][0]['sample_id'] = 'another-id'
            return result
        self.backend.score = wrong
        summary, _ = self.execute()
        self.assertEqual(summary['report']['valid_scores'], 0)
        self.assertTrue(all(p['unsupported_score'] is None for p in summary['report']['predictions']))

    def test_bad_scores_logits_and_coverage_are_rejected(self):
        items = examples(1)
        refs = run.batch_plan(items)[0]['references']
        good = result_for(items)
        run.validate_result(good, refs)
        for key, value in [('unsupported_score', float('nan')), ('unsupported_score', .9),
                           ('logits', [float('inf'), 0.]), ('forward_tensors_sha256', 'invalid'),
                           ('nonpadding_tokens', 513),
                           ('coverage', {'answer': {'full_tokens': 1, 'kept_tokens': 2},
                                         'context': {'full_tokens': 2, 'kept_tokens': 2}})]:
            bad = copy.deepcopy(good)
            bad['rows'][0][key] = value
            with self.subTest(key=key), self.assertRaises(RunConflict): run.validate_result(bad, refs)

    def test_zero_budget_does_not_load_model(self):
        summary, new = self.execute(max_new_batches=0)
        self.assertEqual((new, summary['report']['pending']), (0, 17))
        self.factory.assert_not_called()
        for value in (-1, 501, True):
            with self.assertRaises(ValueError): self.execute(max_new_batches=value)

    def test_corrupt_derived_summary_is_rebuilt_from_journal_without_calls(self):
        original, _ = self.execute()
        (self.directory / 'summary.json').write_text('{}', encoding='utf-8')
        self.factory.reset_mock()
        replay, new = self.execute()
        self.assertEqual((replay, new), (original, 0))
        self.factory.assert_not_called()



if __name__ == '__main__': unittest.main()
