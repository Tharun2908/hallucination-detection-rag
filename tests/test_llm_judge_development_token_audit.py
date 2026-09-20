"""Reserved-arm identity, prompt isolation and resumable CPU audit regressions."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import audit_development_tokens as mod
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.runner import Example
from post_thesis.llm_judge.storage import RunConflict
from test_llm_judge_label_pilot_audit import PinnedLabelToy
from test_llm_judge_label_score import ToyTokenizer


def fixture():
    protocol = deepcopy(mod.load_fit_protocol())
    protocol['calibration'].update(required_rows=2, required_components=1)
    protocol['threshold'].update(required_rows=2, required_components=1)
    inputs, reserved = {}, []
    for index, arm in enumerate(mod.ARMS):
        ids = [str(index * 2 + j) for j in range(2)]
        component = content_hash([arm])
        inputs[arm] = [{'sample_id': rid, 'answer': 'A venue.', 'context': 'A venue exists.'} for rid in ids]
        for row in inputs[arm]:
            reserved.append({'sample_id': row['sample_id'], 'partition': arm,
                             'component_sha256': component, 'input_sha256': content_hash(
                                 {k: row[k] for k in ('answer', 'context')}),
                             'proposed_excluded': False, 'linked_to_pilot': False,
                             'linked_to_native_test': False})
        protocol['partition_membership_hashes'][arm] = {
            'sample_ids_sha256': content_hash(ids), 'component_ids_sha256': content_hash([component])}
    manifest = {'code_revision': protocol['allocator_code_revision'],
                'design_sha256': protocol['reservation_design_sha256'], 'status': 'reserved',
                'model_inputs': inputs, 'train_reservation': reserved,
                'labels_offline': {'never_send': 'SECRET_LABEL_METADATA'}}
    return {'manifest': manifest, 'manifest_sha256': content_hash(manifest)}, protocol


class DevelopmentTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.examples = {arm: [Example(str(i * 2 + j), JudgeInput('A venue.', 'A venue exists.'))
                               for j in range(2)] for i, arm in enumerate(mod.ARMS)}

    def run_audit(self, examples=None, tokenizer=None, revision='audit-test'):
        return mod.audit(examples or self.examples, tokenizer or PinnedLabelToy(), directory=self.directory,
                         revision=revision, files={'toy': 'not_real'}, versions={'toy': 'test'},
                         reference_hash='toy-reference')

    def test_manifest_projects_only_answer_context_and_local_id(self):
        bundle, protocol = fixture()
        with patch.object(mod, 'MANIFEST_SHA256', bundle['manifest_sha256']):
            examples = mod.validate_inputs(bundle, protocol)
        self.assertEqual(examples, self.examples)
        self.assertEqual(set(asdict(examples['calibration'][0].item)), {'answer', 'context'})

    def test_manifest_hash_and_protocol_drift_stop(self):
        bundle, protocol = fixture()
        with self.assertRaisesRegex(RunConflict, 'reservation'):
            mod.validate_inputs(bundle, protocol)
        with patch.object(mod, 'FIT_SHA256', 'changed'), self.assertRaisesRegex(RunConflict, 'protocol'):
            mod.load_fit_protocol()

    def test_extra_metadata_changed_inputs_missing_and_cross_arm_ids_fail(self):
        bundle, protocol = fixture()
        for mutation in ('metadata', 'answer', 'missing', 'cross_arm', 'exclusion', 'revision'):
            bad = deepcopy(bundle)
            manifest = bad['manifest']
            if mutation == 'metadata': manifest['model_inputs']['calibration'][0]['label'] = 1
            if mutation == 'answer': manifest['model_inputs']['calibration'][0]['answer'] = 'changed'
            if mutation == 'missing': manifest['model_inputs']['calibration'].pop()
            if mutation == 'cross_arm': manifest['model_inputs']['operating_threshold'][0]['sample_id'] = '0'
            if mutation == 'exclusion': manifest['train_reservation'][0]['linked_to_native_test'] = True
            if mutation == 'revision': manifest['code_revision'] = 'changed'
            # Re-sign the toy manifest to exercise structural guards beyond checksum.
            bad['manifest_sha256'] = content_hash(manifest)
            with self.subTest(mutation=mutation), patch.object(mod, 'MANIFEST_SHA256', bad['manifest_sha256']):
                with self.assertRaises(RunConflict): mod.validate_inputs(bad, protocol)

    def test_replay_has_no_tokenization_no_report_rewrite_and_same_hash(self):
        report, count = self.run_audit()
        self.assertEqual(count, 4)
        with patch.object(mod, 'prepare_tokenized_input', side_effect=AssertionError('retokenized')), \
                patch.object(mod, 'atomic_json', side_effect=AssertionError('rewrote report')):
            repeated, count = self.run_audit()
        self.assertEqual(count, 0)
        self.assertEqual(repeated, report)
        for arm in mod.ARMS:
            self.assertEqual(report['audit']['summary']['partitions'][arm]['examples_counted'], 2)
        self.assertFalse(report['audit']['summary']['scoring_authorized_by_this_audit'])
        self.assertEqual(report['audit']['summary']['generation_calls'], 0)
        for row in report['audit']['rows']:
            self.assertNotIn('rendered_prompt', row['prepared'])
            self.assertIn('rendered_prompt_sha256', row['prepared'])

    def test_interruption_resumes_completed_prefix_across_arms(self):
        prepare = mod.prepare_tokenized_input
        calls = 0
        def interrupt(item, tokenizer):
            nonlocal calls
            calls += 1
            if calls == 3: raise KeyboardInterrupt()
            return prepare(item, tokenizer)
        with patch.object(mod, 'prepare_tokenized_input', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_audit()
        cached = json.loads((self.directory / 'audit.json').read_text())
        self.assertEqual(len(cached['audit']['rows']), 2)
        self.assertEqual(cached['audit']['rows'][-1]['partition'], 'calibration')
        result, new = self.run_audit()
        self.assertEqual(new, 2)
        self.assertEqual(result['audit']['status'], 'completed')

    def test_changed_revision_input_partition_or_corrupt_cache_stop(self):
        self.run_audit()
        with self.assertRaises(RunConflict): self.run_audit(revision='changed')
        changed = deepcopy(self.examples)
        changed['calibration'][0] = Example('0', JudgeInput('changed', 'A venue exists.'))
        with self.assertRaises(RunConflict): self.run_audit(examples=changed)
        path = self.directory / 'audit.json'
        saved = json.loads(path.read_text())
        saved['audit']['rows'][0]['partition'] = 'operating_threshold'
        saved['audit_sha256'] = content_hash(saved['audit'])
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RunConflict, 'alignment'): self.run_audit()

    def test_overlength_preserved_per_arm_without_truncation(self):
        examples = deepcopy(self.examples)
        examples['operating_threshold'][0] = Example('2', JudgeInput('A' * 33000, 'A venue exists.'))
        result, count = self.run_audit(examples=examples)
        self.assertEqual(count, 4)
        state = result['audit']
        self.assertEqual(state['status'], 'overlength')
        self.assertTrue(state['summary']['partitions']['calibration']['all_inputs_fit'])
        self.assertEqual(state['summary']['partitions']['operating_threshold']['overlength_ids'], ['2'])
        self.assertEqual(state['summary']['http_requests'], 0)
        self.assertGreater(state['rows'][2]['prepared']['input_tokens'], 33000)

    def test_wrong_label_mapping_or_duplicate_input_stops(self):
        with self.assertRaisesRegex(RunConflict, 'mapping'): self.run_audit(tokenizer=ToyTokenizer())
        cached = json.loads((self.directory / 'audit.json').read_text())
        self.assertEqual(cached['audit']['rows'], [])
        changed = deepcopy(self.examples)
        changed['operating_threshold'][0] = changed['calibration'][0]
        with self.assertRaisesRegex(RunConflict, 'duplicate'): self.run_audit(examples=changed)


if __name__ == '__main__':
    unittest.main()
