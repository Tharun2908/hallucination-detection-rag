"""Artificial-data checks for post-thesis legacy fusion reconstruction."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from post_thesis.llm_judge import recover_fusion as recovery
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    np = None


class PersistenceTests(unittest.TestCase):
    def test_completed_replay_does_not_fit_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            compute = Mock(return_value={'study_stage': 'post_thesis', 'results': {'status': 'fixture'}})
            original, created = recovery.recover_once(path, {'revision': 'a'}, compute)
            before = path.read_bytes()
            self.assertTrue(created)
            replay, created = recovery.recover_once(path, {'revision': 'a'}, compute)
            self.assertFalse(created); self.assertEqual(original, replay)
            self.assertEqual(compute.call_count, 1); self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(RunConflict):
                recovery.recover_once(path, {'revision': 'b'}, compute)
            changed = json.loads(before); changed['report']['results']['status'] = 'tampered'
            path.write_text(json.dumps(changed))
            with self.assertRaises(RunConflict):
                recovery.recover_once(path, {'revision': 'a'}, compute)
            self.assertEqual(compute.call_count, 1)

    def test_pinned_bundle_requires_content_and_declared_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.json'
            report = {'synthetic': True}; digest = content_hash(report)
            path.write_text(json.dumps({'report': report, 'report_sha256': digest}))
            self.assertEqual(recovery.checked_bundle(path, 'report', 'report_sha256', digest), report)
            report['synthetic'] = False
            path.write_text(json.dumps({'report': report, 'report_sha256': digest}))
            with self.assertRaises(RunConflict):
                recovery.checked_bundle(path, 'report', 'report_sha256', digest)


@unittest.skipIf(np is None, 'isolated NumPy/sklearn dependencies not installed')
class NumericalTests(unittest.TestCase):
    def test_two_features_ignore_metadata_and_preserve_legacy_normalization(self):
        s2 = [{'idx': i, 'raw_min_relevance': v, 'model': 'irrelevant', 'task_type': 'QA'}
              for i, v in enumerate((-20, -11.430, 0, 10.641, 20))]
        s4 = [{'idx': i, 'signal4_score': .4} for i in range(5)]
        x = recovery.features(s2, s4)
        self.assertEqual(x.shape, (5, 2))
        self.assertEqual(x[:, 0].tolist(), [0, 0, 11.430 / (10.641 - (-11.430)), 1, 1])
        changed = deepcopy(s2)
        for row in changed: row.update(model='other', task_type='Summary', ground_truth_hallucination=999)
        np.testing.assert_array_equal(x, recovery.features(changed, s4))
        for value in (None, float('nan'), float('inf'), True):
            broken = deepcopy(s2); broken[0]['raw_min_relevance'] = value
            with self.assertRaises(RunConflict): recovery.features(broken, s4)
        with self.assertRaises(RunConflict): recovery.features(list(reversed(s2)), s4)

    def test_historical_threshold_float_grid_and_tie_rule(self):
        probs = np.array([.10, .80]); labels = np.array([0, 1])
        result = recovery.select_threshold(probs, labels)
        expected = float(np.arange(.05, .96, .05)[2])
        self.assertEqual(result['selected']['threshold_hex'], expected.hex())
        self.assertEqual(result['selected']['f1'], 1.0)

    def test_matches_original_procedure_and_json_coefficients(self):
        # Load only pure original function definitions; never execute its script body.
        source = recovery.audit.REPO_ROOT / 'fusion/fusion_decomposition_review.py'
        tree = ast.parse(source.read_text())
        names = {'norm_s2', 'best_threshold', 'make_features'}
        pure = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {'np': np, 'f1_score': f1_score, 'S2_MIN': -11.430, 'S2_MAX': 10.641,
                 'THRESHOLDS': np.arange(.05, .96, .05)}
        exec(compile(pure, str(source), 'exec'), scope)
        rng = np.random.RandomState(72)
        labels = np.tile([0, 1], 50)
        s2 = [{'idx': i, 'raw_min_relevance': float(v)} for i, v in enumerate(rng.uniform(-15, 15, 100))]
        s4 = [{'idx': i, 'signal4_score': float(v)} for i, v in enumerate(rng.uniform(0, 1, 100))]
        x = recovery.features(s2, s4)
        rows = [{'s2': scope['norm_s2'](a['raw_min_relevance']), 's4': b['signal4_score']}
                for a, b in zip(s2, s4)]
        original_x, _ = scope['make_features'](rows, ['s2', 's4'], False)
        np.testing.assert_array_equal(x, original_x)
        oof = np.zeros(100)
        for train, heldout in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(100), labels):
            model = LogisticRegression(max_iter=1000, random_state=42).fit(original_x[train], labels[train])
            oof[heldout] = model.predict_proba(original_x[heldout])[:, 1]
        threshold, f1 = scope['best_threshold'](oof, labels)
        reference = {'train_threshold': round(threshold, 4), 'oof_train_f1': round(f1, 4)}
        result = recovery.reconstruct(x, labels, x[:12], reference)
        np.testing.assert_array_equal(result['TRAIN_meta_oof_scores'], oof)
        self.assertEqual(result['fusion_fit_calls'], 6)
        self.assertEqual(result['threshold']['selected']['threshold'], threshold)
        self.assertEqual(sorted(i for fold in result['folds'] for i in fold['heldout_indices']), list(range(100)))
        final = LogisticRegression(max_iter=1000, random_state=42).fit(x, labels)
        record = json.loads(json.dumps(result['full_model'], allow_nan=False))
        np.testing.assert_allclose(recovery.predict_record(record, x[:12]), final.predict_proba(x[:12])[:, 1], rtol=0, atol=1e-14)
        # A TRAIN mismatch must never advance to final fitting/TEST predictions.
        bad = recovery.reconstruct(x, labels, x[:12], {'train_threshold': 99, 'oof_train_f1': 99})
        self.assertEqual(bad['fusion_fit_calls'], 5)
        self.assertEqual(bad['TEST_scores'], []); self.assertIsNone(bad['full_model'])
        self.assertEqual(bad['status'], 'historical_TRAIN_reference_mismatch')


if __name__ == '__main__': unittest.main()
