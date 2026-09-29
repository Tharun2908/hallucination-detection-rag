"""Artificial disagreement selection and frozen-decision checks; no benchmark reads."""
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from post_thesis.llm_judge import disagreement_review as review
from post_thesis.llm_judge.prompts import content_hash
from post_thesis.llm_judge.storage import RunConflict


def rows_fixture():
    rows = []
    for i, pattern in enumerate(itertools.product((0, 1), repeat=4)):
        for label in (0, 1):
            for copy in range(3):
                rows.append({'benchmark': 'ragtruth', 'sample_id': f'{i}-{label}-{copy}',
                    'component': f'{i}-{label}-{copy}', 'slice': 'QA', 'label': label,
                    'input_sha256': 'toy', 'answer': 'One.', 'context': 'One.',
                    'scores': {}, 'predictions': dict(zip(review.metrics.MAIN_SYSTEMS, pattern))})
    return rows


class ReviewTests(unittest.TestCase):
    def test_pair_counts_exhaustive_binary_truth_table(self):
        result = review.count_rows(rows_fixture())
        self.assertEqual(result['n'], 96)
        self.assertEqual(result['all_four_correct'], 6)
        self.assertEqual(result['all_four_wrong'], 6)
        self.assertEqual(result['mixed_decisions'], 84)
        self.assertEqual(len(result['decision_patterns']), 16)
        for pair in result['pair_correctness'].values():
            self.assertEqual(pair, dict.fromkeys(review.STATES, 24))

    def test_selection_deterministic_component_limited_and_blinded(self):
        rows = rows_fixture()
        first = review.build_review(rows)
        self.assertEqual(first, review.build_review(list(reversed(rows))))
        result, packet, key, annotations = first
        for stratum in result['selection_strata']:
            self.assertEqual(len(stratum['selected_ids']), 2)
            selected = [r for r in rows if r['sample_id'] in stratum['selected_ids']]
            self.assertEqual(len({r['component'] for r in selected}), 2)
            for row in selected:
                self.assertEqual(review.correctness_state(row['predictions']['judge'], row['predictions'][stratum['baseline']], row['label']), stratum['state'])
        self.assertEqual(len(packet), len({r['review_id'] for r in packet}))
        self.assertEqual([r['review_id'] for r in packet], [r['review_id'] for r in key])
        self.assertEqual([r['review_id'] for r in packet], [r['review_id'] for r in annotations])
        self.assertTrue(all(set(r) == {'review_id', 'answer', 'context'} for r in packet))
        self.assertTrue(all(r['independent_verdict'] is None for r in annotations))
        self.assertTrue(all(r['selection_memberships'] for r in key))

    def test_shared_components_shortfalls_and_cross_benchmark_ids(self):
        rows = rows_fixture()[:3]
        for row in rows: row['component'] = 'one-component'
        other = deepcopy(rows)
        for row in other: row['benchmark'] = 'halubench'; row['slice'] = 'DROP'
        result, packet, key, _ = review.build_review(rows + other)
        self.assertGreaterEqual(len(packet), 2)
        self.assertLessEqual(len(packet), 4)
        self.assertEqual({r['benchmark'] for r in key}, {'ragtruth', 'halubench'})
        self.assertTrue(all(s['shortfall'] >= 1 for s in result['selection_strata']))
        with self.assertRaises(RunConflict): review.build_review(rows + rows)

    def test_evaluation_hash_and_aligned_input_identity(self):
        inputs = {'synthetic': True}
        report = {'identity': {'aligned_evaluation_inputs_sha256': content_hash(inputs)}}
        digest = content_hash(report); bundle = {'report_sha256': digest, 'report': report}
        self.assertEqual(review.verify_evaluation(bundle, digest, inputs), report)
        with self.assertRaises(RunConflict): review.verify_evaluation(bundle, digest, {'synthetic': False})
        bundle['report']['extra'] = 'tampered'
        with self.assertRaises(RunConflict): review.verify_evaluation(bundle, digest, inputs)

    def test_replay_does_not_write_and_tamper_prevents_partial_write(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); payloads = {'report.json': {'fixed': True}, 'review_packet.json': []}
            self.assertTrue(all(review.write_outputs(directory, payloads).values()))
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            (directory / 'my_annotations.json').write_text('human work')
            self.assertFalse(any(review.write_outputs(directory, payloads).values()))
            for name, state in before.items():
                p = directory / name; self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), state)
            self.assertEqual((directory / 'my_annotations.json').read_text(), 'human work')
            with self.assertRaises(RunConflict):
                review.write_outputs(directory, {'new.json': {}, 'report.json': {'fixed': False}})
            self.assertFalse((directory / 'new.json').exists())


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'), 'pinned CPU numerical dependencies required')
class DecisionTests(unittest.TestCase):
    def fixture(self):
        ids = ['one', 'two']; text = {'answer': 'One.', 'context': 'One.'}
        manifest = {'model_inputs': [{'sample_id': i, **text} for i in ids],
            'offline_rows': [{'sample_id': s, 'test_index': i, 'input_sha256': content_hash(text),
                             'component_sha256': s, 'label': i} for i, s in enumerate(ids)]}
        inputs = {'sample_ids': ids, 'groups': ids, 'labels': [0, 1], 'axes': {'task': ['QA', 'QA']},
            'scores': {'judge': [-1.25, -1.2501], 'S4': [.55, .5499],
                       'MiniCheck_7B': [.20000000000000004, .2], 'S2_S4_metadata_free': [.45, .4499]}}
        points = {s: {'confusion': {'tp': 0, 'fp': 1, 'fn': 1, 'tn': 0}} for s in review.metrics.MAIN_SYSTEMS}
        points['MiniCheck_7B']['confusion'] = {'tp': 1, 'fp': 0, 'fn': 0, 'tn': 1}
        return manifest, inputs, {'results': {'paired': {'shared_points': points}}}

    def test_original_threshold_boundaries_and_confusion_reproduction(self):
        manifest, inputs, evaluation = self.fixture()
        rows = review.decision_rows('ragtruth', manifest, inputs, evaluation)
        self.assertEqual(rows[0]['predictions'], {'judge': 1, 'S4': 1, 'MiniCheck_7B': 0, 'S2_S4_metadata_free': 1})
        self.assertEqual(rows[1]['predictions'], {'judge': 0, 'S4': 0, 'MiniCheck_7B': 1, 'S2_S4_metadata_free': 0})

    def test_misalignment_missing_score_or_confusion_change_rejected(self):
        for kind in ('text', 'id', 'missing', 'label', 'confusion', 'group'):
            manifest, inputs, evaluation = self.fixture()
            if kind == 'text': manifest['model_inputs'][0]['answer'] = 'Two.'
            if kind == 'id': inputs['sample_ids'] = ['one', 'one']
            if kind == 'missing': inputs['scores']['S4'][0] = None
            if kind == 'label': inputs['labels'][0] = 1
            if kind == 'confusion': evaluation['results']['paired']['shared_points']['judge']['confusion']['tp'] = 1
            if kind == 'group': manifest['offline_rows'][0]['component_sha256'] = 'other'
            with self.subTest(kind=kind), self.assertRaises(RunConflict):
                review.decision_rows('ragtruth', manifest, inputs, evaluation)


if __name__ == '__main__': unittest.main()
