"""Checkpoint identity and replay safety, without weights or third-party packages."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from post_thesis.llm_judge import check_s4_checkpoint as check
from post_thesis.llm_judge.storage import RunConflict


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'checkpoint'
        self.checkpoint.mkdir()
        config = {'model_type': 'deberta-v2', 'architectures': ['DebertaV2ForSequenceClassification'],
                  'id2label': {'0': 'FAITHFUL', '1': 'HALLUCINATED'},
                  'vocab_size': 128100, 'max_position_embeddings': 512}
        for name, data in {'config.json': json.dumps(config).encode(),
                           'tokenizer.json': b'{}', 'tokenizer_config.json': b'{}',
                           'model.safetensors': b'artificial-weights-not-loadable'}.items():
            (self.checkpoint / name).write_bytes(data)
        self.pins = {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in self.checkpoint.iterdir()}
        self.patch = patch.object(check, 'FILES', self.pins)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_changed_weight_bytes_block_even_if_size_is_unchanged(self):
        check.verify_checkpoint(self.checkpoint)
        p = self.checkpoint / 'model.safetensors'
        data = p.read_bytes()
        p.write_bytes(b'X' + data[1:])
        with self.assertRaisesRegex(RunConflict, 'fingerprint mismatch'):
            check.verify_checkpoint(self.checkpoint)

    def test_missing_files_and_added_fold_tokenizer_are_rejected(self):
        extra = self.checkpoint / 'spm.model'
        extra.write_bytes(b'borrowed-from-fold')
        with self.assertRaises(RunConflict): check.verify_checkpoint(self.checkpoint)
        extra.unlink()
        (self.checkpoint / 'tokenizer.json').unlink()
        with self.assertRaises(RunConflict): check.verify_checkpoint(self.checkpoint)

    def test_reversed_label_mapping_is_rejected_even_with_matching_bytes(self):
        p = self.checkpoint / 'config.json'
        config = json.loads(p.read_text())
        config['id2label'] = {'0': 'HALLUCINATED', '1': 'FAITHFUL'}
        data = json.dumps(config).encode()
        p.write_bytes(data)
        self.pins[p.name] = (len(data), hashlib.sha256(data).hexdigest())
        with self.assertRaisesRegex(RunConflict, 'class mapping'):
            check.verify_checkpoint(self.checkpoint)

    def test_partial_loading_cannot_be_treated_as_success(self):
        clean = {k: set() for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
        check.validate_loading_info(clean)
        for key in clean:
            bad = {**clean, key: ['classifier.weight']}
            with self.subTest(key=key), self.assertRaises(RunConflict):
                check.validate_loading_info(bad)
        with self.assertRaises(RunConflict): check.validate_loading_info({})

    def test_nonfinite_and_wrong_shape_logits_are_rejected(self):
        check.validate_logits([[0.1, -0.2]])
        for rows in ([], [[1.]], [[1., 2., 3.]], [[0., float('nan')]],
                     [[float('inf'), 0.]], [[True, 0.]], [[0., 1.], [0., 1.]]):
            with self.subTest(rows=rows), self.assertRaises(RunConflict): check.validate_logits(rows)

    def test_replay_checks_identity_and_checkpoint_without_another_forward(self):
        probe = Mock(return_value={'status': 'passed', 'synthetic_forward_calls': 4})
        directory = self.root / 'run'
        bundle, created = check.check_once(directory, self.checkpoint, 'revision', {}, probe)
        self.assertTrue(created)
        before = (directory / 'report.json').read_bytes()
        replay, created = check.check_once(directory, self.checkpoint, 'revision', {}, probe)
        self.assertFalse(created)
        self.assertEqual(bundle, replay)
        self.assertEqual(before, (directory / 'report.json').read_bytes())
        probe.assert_called_once()
        self.assertFalse(bundle['report']['comparison_ready'])
        with self.assertRaises(RunConflict):
            check.check_once(directory, self.checkpoint, 'other-revision', {}, probe)
        (self.checkpoint / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(RunConflict):
            check.check_once(directory, self.checkpoint, 'revision', {}, probe)
        probe.assert_called_once()

    def test_checkpoint_mutation_during_probe_does_not_save_a_success(self):
        def mutate(_):
            (self.checkpoint / 'model.safetensors').write_bytes(b'changed')
            return {'status': 'passed'}
        with self.assertRaises(RunConflict):
            check.check_once(self.root / 'run', self.checkpoint, 'revision', {}, mutate)
        self.assertFalse((self.root / 'run/report.json').exists())

    def test_corrupt_record_is_not_replayed(self):
        directory = self.root / 'run'
        check.check_once(directory, self.checkpoint, 'revision', {}, Mock(return_value={'status': 'passed'}))
        path = directory / 'report.json'
        bundle = json.loads(path.read_text())
        bundle['report']['comparison_ready'] = True
        path.write_text(json.dumps(bundle), encoding='utf-8')
        with self.assertRaises(RunConflict):
            check.check_once(directory, self.checkpoint, 'revision', {}, Mock())


if __name__ == '__main__':
    unittest.main()
