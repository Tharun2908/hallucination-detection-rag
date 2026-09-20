"""CPU audit recovery, input boundary and historical provenance regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import audit_label_pilot as mod
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.storage import RunConflict
from test_llm_judge_label_score import ToyTokenizer


class PinnedLabelToy(ToyTokenizer):
    # Deliberately fake tokenizer; exercises input boundaries, not Qwen behavior.
    def encode(self, text, *, add_special_tokens):
        return [32 if c == 'A' else 33 if c == 'B' else ord(c) + 100 for c in text]
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return ''.join('A' if i == 32 else 'B' if i == 33 else chr(i - 100) for i in ids)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.examples = [Example(str(i), JudgeInput('A venue.', 'A venue exists.')) for i in range(3)]
    def run_audit(self, tokenizer=None, examples=None, revision='audit-commit'):
        return mod.audit(self.examples if examples is None else examples, tokenizer or PinnedLabelToy(),
                         directory=self.directory, revision=revision, files={'fake':'hash'},
                         versions={'fake':'test'}, reference_hash='fake-reference')
    def test_replay_preserves_hash_and_does_not_retokenize(self):
        first, count = self.run_audit()
        self.assertEqual(count, 3)
        with patch.object(mod, 'prepare_tokenized_input', side_effect=AssertionError('cache missed')):
            second, count = self.run_audit()
        self.assertEqual(first, second)
        self.assertEqual(count, 0)
        self.assertEqual(first['audit']['summary']['generation_calls'], 0)
        self.assertFalse(first['audit']['summary']['scoring_authorized_by_this_audit'])
        for row in first['audit']['rows']:
            rendered = row['prepared']['rendered_prompt']
            self.assertNotIn('sample_id', rendered)
            self.assertNotIn('fake-reference', rendered)
    def test_interruption_keeps_completed_rows_only(self):
        prepare = mod.prepare_tokenized_input
        calls = 0
        def interrupt(item, tokenizer):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return prepare(item, tokenizer)
        with patch.object(mod, 'prepare_tokenized_input', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_audit()
        saved = json.loads((self.directory / 'audit.json').read_text())
        self.assertEqual(len(saved['audit']['rows']), 1)
        report, count = self.run_audit()
        self.assertEqual(count, 2)
        self.assertEqual(report['audit']['status'], 'completed')
    def test_changed_inputs_revision_or_corrupt_rows_fail(self):
        self.run_audit()
        with self.assertRaises(RunConflict): self.run_audit(revision='changed')
        changed = [Example('0', JudgeInput('changed', 'changed')), *self.examples[1:]]
        with self.assertRaises(RunConflict): self.run_audit(examples=changed)
        path = self.directory / 'audit.json'
        saved = json.loads(path.read_text()); saved['audit']['rows'].pop()
        path.write_text(json.dumps(saved))
        with self.assertRaises(RunConflict): self.run_audit()
    def test_overlength_is_counted_without_truncation_or_generation(self):
        examples = [Example('long', JudgeInput('A' * 33000, 'B'))]
        report, _ = self.run_audit(examples=examples)
        self.assertEqual(report['audit']['status'], 'overlength')
        self.assertEqual(report['audit']['summary']['overlength_ids'], ['long'])
        self.assertGreater(report['audit']['rows'][0]['prepared']['input_tokens'], 33000)
        self.assertEqual(report['audit']['summary']['http_requests'], 0)
    def test_wrong_label_mapping_fails_before_saving_row(self):
        with self.assertRaisesRegex(RunConflict, 'mapping'):
            self.run_audit(tokenizer=ToyTokenizer())
        self.assertEqual(json.loads((self.directory / 'audit.json').read_text())['audit']['rows'], [])
    def test_historical_report_pin_is_checked_without_current_revision(self):
        report = {'code_revision': mod.SYNTHETIC_REVISION, 'run_id':'qwen3-label-score-synthetic-v1',
                  'requests_total':30,'valid_scores':30,'terminal_failures':0,'pending':0,
                  'new_attempts':0,'halt_reason':None,'transport_checks_passed':True}
        digest = content_hash(report); report['report_sha256'] = digest
        bundle = {'manifest_sha256':mod.MANIFEST_SHA256}
        with patch.object(mod, 'pilot_examples', return_value=self.examples * 16 + self.examples[:2]), \
             patch.object(mod, 'SYNTHETIC_REPORT_SHA256', digest):
            self.assertEqual(len(mod.validate_inputs(bundle, report)), 50)
            bad = deepcopy(report); bad['valid_scores'] = 29
            with self.assertRaisesRegex(RunConflict, 'checksum'): mod.validate_inputs(bundle, bad)
            with self.assertRaisesRegex(RunConflict, 'manifest'):
                mod.validate_inputs({'manifest_sha256':'changed'}, report)


if __name__ == '__main__':
    unittest.main()
