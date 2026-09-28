"""Fresh feature alignment and frozen fusion application on artificial data."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import apply_fresh_fusion as fusion
from post_thesis.llm_judge.prepare_test_manifest import save_once
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def fixture():
    targets = [{'sample_id': str(i), 'test_index': i, 'input_sha256': str(i)*64} for i in range(3)]
    s2 = [{**t, 'status': 'ok', 'raw_min_relevance': v} for t,v in zip(targets, [-20., 0.1234567, 20.])]
    s4 = [{**t, 'status': 'ok', 'unsupported_score': v} for t,v in zip(targets, [.1, .543276, .9])]
    recovery = {'results': {'full_model': {
        'feature_order': ['s2', 's4'], 'metadata_features': False, 'classes': [0,1],
        'coef': fusion.COEF[:], 'intercept': fusion.INTERCEPT,
        's2_normalization': {'min': -11.43, 'max': 10.641, 'clip': [0.,1.]}},
        'threshold': {'comparator': '>=', 'selected': {
            'threshold': fusion.THRESHOLD, 'threshold_hex': fusion.THRESHOLD.hex()}}}}
    return s2, s4, targets, recovery


class AlignmentTests(unittest.TestCase):
    def test_only_declared_features_and_identifiers_are_projected(self):
        s2,s4,targets,_ = fixture()
        for rows in (s2,s4,targets):
            for row in rows: row.update(label='unused', model='unused', task_type='unused')
        a,b,projected = fusion.aligned_features(s2,s4,targets)
        self.assertEqual(a[1]['raw_min_relevance'], .1235)
        self.assertEqual(b[1]['signal4_score'], .5433)
        self.assertEqual(set(projected[0]), {'sample_id','test_index','input_sha256',
                                           's2_raw_min_rounded','s4_probability_rounded'})

    def test_missing_reordered_misaligned_and_duplicate_rows_rejected(self):
        s2,s4,targets,_ = fixture()
        bad = copy.deepcopy(s2); bad[0]['input_sha256'] = 'bad'
        failed = copy.deepcopy(s2); failed[0]['status'] = 'error'
        for rows in (s2[:-1], list(reversed(s2)), bad, failed, [s2[0],s2[0],s2[2]]):
            with self.assertRaises(RunConflict): fusion.aligned_features(rows,s4,targets)

    def test_invalid_values_rejected(self):
        for value in (None, True, float('nan'), float('inf')):
            s2,s4,t,_ = fixture(); s2[0]['raw_min_relevance'] = value
            with self.assertRaises(RunConflict): fusion.aligned_features(s2,s4,t)
        for value in (-.1, 1.1, False, float('nan')):
            s2,s4,t,_ = fixture(); s4[0]['unsupported_score'] = value
            with self.assertRaises(RunConflict): fusion.aligned_features(s2,s4,t)

    def test_frozen_parameters_and_comparator_cannot_drift(self):
        for key, value in (('coef', [0.,0.]), ('metadata_features', True),
                           ('feature_order',['s4','s2']), ('intercept',0.)):
            *_,r = fixture(); r['results']['full_model'][key] = value
            with self.assertRaises(RunConflict): fusion.fixed_model(r)
        *_,r = fixture(); r['results']['threshold']['comparator'] = '>'
        with self.assertRaises(RunConflict): fusion.fixed_model(r)

    def test_agreement_uses_frozen_inclusive_threshold_without_labels(self):
        row = {'sample_id':'a','test_index':0,'input_sha256':'hash',
               'unsupported_score':fusion.THRESHOLD,'predicted_unsupported':True}
        old = {'sample_id':'a','test_index':0,'target_manifest_input_sha256':'hash',
               'unsupported_score':math.nextafter(fusion.THRESHOLD, 0.)}
        result = fusion.legacy_agreement([row],[old])
        self.assertEqual(result['fixed_threshold_decision_changes'],1)
        self.assertFalse(result['labels_used'])
        old['unsupported_score'] = fusion.THRESHOLD
        self.assertEqual(fusion.legacy_agreement([row],[old])['fixed_threshold_decision_changes'],0)


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),
                     'requires the pinned CPU fit dependencies')
class ApplicationTests(unittest.TestCase):
    def test_probabilities_match_explicit_frozen_math_without_fit(self):
        args = fixture()
        with patch('post_thesis.llm_judge.recover_fusion.reconstruct', side_effect=AssertionError('no refit')):
            rows = fusion.apply(*args)
        self.assertEqual([rows[i]['s2_normalized'] for i in (0,2)],[0.,1.])
        self.assertAlmostEqual(rows[1]['s2_normalized'], (.1235+11.43)/(10.641+11.43))
        for row in rows:
            z = sum(c*x for c,x in zip(fusion.COEF,[row['s2_normalized'],row['s4_probability_rounded']]))+fusion.INTERCEPT
            self.assertAlmostEqual(row['unsupported_score'], 1/(1+math.exp(-z)), places=14)
            self.assertEqual(row['predicted_unsupported'], row['unsupported_score']>=fusion.THRESHOLD)

    def test_output_replay_is_identical_and_changed_inputs_do_not_overwrite(self):
        rows = fusion.apply(*fixture())
        value = {'report':rows,'report_sha256':content_hash(rows)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            self.assertTrue(save_once(path,value))
            self.assertFalse(save_once(path,value))
            changed = fixture(); changed[0][0]['raw_min_relevance'] = 0.
            other = fusion.apply(*changed)
            with self.assertRaises(RunConflict):
                save_once(path, {'report':other,'report_sha256':content_hash(other)})
            self.assertEqual(json.loads(path.read_text()),value)


if __name__ == '__main__': unittest.main()
