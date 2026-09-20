"""Artificial TEST token audits; no benchmark inference or model weights."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import audit_test_tokens as mod
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.storage import RunConflict
from test_llm_judge_label_pilot_audit import PinnedLabelToy


def manifest_fixture():
    frozen = mod.load_freeze()
    inputs = [{'sample_id': str(i), 'answer': 'A venue.', 'context': 'A venue exists.'} for i in range(3)]
    rows = [{'sample_id': row['sample_id'], 'test_index': i,
             'input_sha256': content_hash({k: row[k] for k in ('answer', 'context')}),
             'label': i % 2, 'metadata': {'model': 'DO_NOT_SEND', 'task_type': 'DO_NOT_SEND'}}
            for i, row in enumerate(inputs)]
    manifest = {'code_revision': mod.TEST_REVISION, 'freeze_sha256': content_hash(frozen),
                'fit_report_sha256': frozen['source_fit_report_sha256'],
                'model_inputs': inputs, 'offline_rows': rows}
    return {'manifest': manifest, 'manifest_sha256': content_hash(manifest)}, frozen


class TestTokenAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.examples = [Example(str(i), JudgeInput('A venue.', 'A venue exists.')) for i in range(3)]

    def run_audit(self, *, examples=None, revision='artificial-test', reference_hash='toy'):
        return mod.audit(examples or self.examples, PinnedLabelToy(), directory=self.directory,
                         revision=revision, files={'toy': 'test'}, versions={'toy': 'test'}, reference_hash=reference_hash)

    def validate_fixture(self, bundle, frozen):
        with patch.object(mod, 'TEST_MANIFEST_SHA256', bundle['manifest_sha256']), patch.object(mod, 'TEST_ROWS', 3):
            return mod.validate_inputs(bundle, frozen)

    def test_manifest_projects_only_exact_answer_context(self):
        bundle, frozen = manifest_fixture()
        examples = self.validate_fixture(bundle, frozen)
        self.assertEqual(examples, self.examples)
        self.assertEqual(set(asdict(examples[0].item)), {'answer', 'context'})
        prompts = [mod.label_messages(ex.item) for ex in examples]
        self.assertNotIn('DO_NOT_SEND', json.dumps(prompts))
        self.assertNotIn('sample_id', json.dumps(prompts))

    def test_resigned_structural_drift_rejected(self):
        bundle, frozen = manifest_fixture()
        for kind in ('extra_field', 'duplicate', 'order', 'changed_text', 'missing', 'freeze', 'revision'):
            bad = deepcopy(bundle); manifest = bad['manifest']
            if kind == 'extra_field': manifest['model_inputs'][0]['label'] = 1
            if kind == 'duplicate': manifest['model_inputs'][1]['sample_id'] = '0'
            if kind == 'order': manifest['offline_rows'].reverse()
            if kind == 'changed_text': manifest['model_inputs'][0]['answer'] += ' '
            if kind == 'missing': manifest['model_inputs'].pop()
            if kind == 'freeze': manifest['freeze_sha256'] = 'changed'
            if kind == 'revision': manifest['code_revision'] = 'changed'
            bad['manifest_sha256'] = content_hash(manifest)
            with self.subTest(kind=kind), self.assertRaises(RunConflict):
                self.validate_fixture(bad, frozen)
        with self.assertRaises(RunConflict): mod.validate_inputs(bundle, frozen)

    def test_cached_replay_no_tokenization_no_rewrite(self):
        result, new = self.run_audit()
        self.assertEqual(new, 3)
        original = (self.directory / 'audit.json').read_bytes()
        with patch.object(mod, 'prepare_tokenized_input', side_effect=AssertionError('retokenized')), \
                patch.object(mod, 'atomic_json', side_effect=AssertionError('rewrote')):
            replay, new = self.run_audit()
        self.assertEqual(new, 0); self.assertEqual(result, replay)
        self.assertEqual(original, (self.directory / 'audit.json').read_bytes())
        summary = result['audit']['summary']
        self.assertTrue(summary['all_inputs_fit'])
        self.assertFalse(summary['scoring_authorized_by_this_audit'])
        self.assertEqual(summary['generation_calls'], 0); self.assertEqual(summary['http_requests'], 0)
        for row in result['audit']['rows']: self.assertNotIn('rendered_prompt', row['prepared'])

    def test_interruption_preserves_successful_prefix(self):
        prepare = mod.prepare_tokenized_input
        calls = 0
        def interrupted(item, tokenizer):
            nonlocal calls
            calls += 1
            if calls == 3: raise KeyboardInterrupt()
            return prepare(item, tokenizer)
        with patch.object(mod, 'prepare_tokenized_input', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt): self.run_audit()
        saved = json.loads((self.directory / 'audit.json').read_text())
        self.assertEqual(saved['audit']['status'], 'interrupted')
        self.assertEqual(saved['audit']['summary']['examples_counted'], 2)
        result, new = self.run_audit()
        self.assertEqual(new, 1); self.assertEqual(result['audit']['status'], 'completed')

    def test_corrupt_rows_and_changed_identity_rejected(self):
        self.run_audit()
        for kwargs in ({'revision': 'other'}, {'reference_hash': 'changed'},
                       {'examples': [Example('0', JudgeInput('Different answer.', 'A venue exists.'))] + self.examples[1:]}):
            with self.assertRaises(RunConflict): self.run_audit(**kwargs)
        path = self.directory / 'audit.json'; original = json.loads(path.read_text())
        for kind in ('mapping', 'input', 'position', 'summary', 'state', 'rendered_text'):
            bad = deepcopy(original); state = bad['audit']; prepared = state['rows'][0]['prepared']
            if kind == 'mapping': prepared['labels']['unsupported']['token_id'] = 32
            if kind == 'input': prepared['input_sha256'] = '0' * 64
            if kind == 'position': prepared['score_position'] += 1
            if kind == 'summary': state['summary']['total_input_tokens'] += 1
            if kind == 'state': state['status'] = 'unknown'
            if kind == 'rendered_text': prepared['rendered_prompt'] = 'unexpected'
            bad['audit_sha256'] = content_hash(state); path.write_text(json.dumps(bad))
            with self.subTest(kind=kind), self.assertRaises(RunConflict): self.run_audit()

    def test_overlength_input_preserved_and_replay_unchanged(self):
        examples = self.examples[:2] + [Example('2', JudgeInput('A' * 33000, 'A venue exists.'))]
        result, new = self.run_audit(examples=examples)
        self.assertEqual(new, 3); self.assertEqual(result['audit']['status'], 'overlength')
        self.assertEqual(result['audit']['summary']['overlength_ids'], ['2'])
        self.assertFalse(result['audit']['summary']['all_inputs_fit'])
        self.assertGreater(result['audit']['rows'][2]['prepared']['input_tokens'], 33000)
        replay, new = self.run_audit(examples=examples)
        self.assertEqual(new, 0); self.assertEqual(result, replay)

    def test_fusion_verification_keeps_comparison_unready_and_checks_alignment(self):
        bundle, _ = manifest_fixture(); manifest = bundle['manifest']
        predictions = [{'sample_id': row['sample_id'], 'test_index': row['test_index'],
                        'target_manifest_input_sha256': row['input_sha256'], 'unsupported_score': .6}
                       for row in manifest['offline_rows']]
        report = {'run_id': mod.FUSION_RUN_ID, 'study_stage': 'post_thesis',
                  'identity': {'code_revision': mod.FUSION_REVISION,
                               'TEST_manifest_sha256': mod.TEST_MANIFEST_SHA256,
                               'judge_freeze_sha256': mod.FREEZE_SHA256},
                  'results': {'status': 'reconstructed_legacy_baseline_with_provenance_limits',
                              'rounded_TRAIN_reference_matches': True, 'fusion_fit_calls': 6,
                              'TEST_predictions': predictions},
                  'comparison_ready': False, 'TEST_metrics_computed': False}
        def check(value):
            digest = content_hash(value)
            with patch.object(mod, 'FUSION_REPORT_SHA256', digest), patch.object(mod, 'TEST_ROWS', 3):
                return mod.validate_recovery({'report_sha256': digest, 'report': value}, manifest)
        result = check(report)
        self.assertFalse(result['comparison_ready']); self.assertEqual(result['new_fits'], 0)
        for kind in ('index', 'input', 'missing', 'out_of_range', 'readiness'):
            bad = deepcopy(report)
            if kind == 'index': bad['results']['TEST_predictions'][0]['test_index'] = 1
            if kind == 'input': bad['results']['TEST_predictions'][0]['target_manifest_input_sha256'] = 'wrong'
            if kind == 'missing': bad['results']['TEST_predictions'].pop()
            if kind == 'out_of_range': bad['results']['TEST_predictions'][0]['unsupported_score'] = -1
            if kind == 'readiness': bad['comparison_ready'] = True
            with self.subTest(kind=kind), self.assertRaises(RunConflict): check(bad)


if __name__ == '__main__': unittest.main()
