"""Artificial evaluation examples; no benchmark files or model calls."""
import json
import unittest
from unittest.mock import patch
from post_thesis.llm_judge import evaluation_math as m
from post_thesis.llm_judge.storage import RunConflict
try:
    import numpy as np
    import sklearn
except ImportError:
    np = None


def ready_fixture():
    return {name: {'comparison_ready': True, **{k: True for k in m.PROVENANCE_FIELDS}}
            for name in m.MAIN_SYSTEMS}


def paired_fixture():
    labels = [0, 1, 0, 1, 0, 1]
    scores = {'judge': [-3., 3., -3., 3., -3., 3.], 'S4': [.1, .9] * 3,
              'MiniCheck_7B': [.9, .1] * 3, 'S2_S4_metadata_free': [.1, .9] * 3}
    return labels, scores, [str(i) for i in range(6)], ['a', 'a', 'b', 'b', 'c', 'c']


class GateTests(unittest.TestCase):
    def test_missing_or_unready_evidence_blocks_comparison(self):
        provenance = ready_fixture()
        self.assertEqual(m.comparison_gate(provenance)['status'], 'ready')
        provenance['S4']['input_content_verified'] = False
        self.assertEqual(m.comparison_gate(provenance)['missing_requirements'], {'S4': ['input_content_verified']})
        self.assertEqual(set(m.comparison_gate({})['missing_requirements']), set(m.MAIN_SYSTEMS))


class ContractTests(unittest.TestCase):
    def test_contract_pins_implementation_and_disallows_silent_changes(self):
        from post_thesis.llm_judge import check_evaluation_math as check
        contract = check.load_contract()
        self.assertEqual(contract['bootstrap']['replicates'], 2000)
        self.assertEqual(contract['generation_allowance'], 0)
        with patch.object(check, 'file_sha256', return_value='changed'):
            with self.assertRaises(RunConflict): check.load_contract()


@unittest.skipIf(np is None, 'pinned numerical dependencies not installed')
class EvaluationTests(unittest.TestCase):
    def test_frozen_threshold_equality_and_original_minicheck_float(self):
        cases = [('judge', [-1.2500001, -1.25, 0], [0, 1, 1]),
                 ('S4', [.549999, .55, .56], [0, 1, 1]),
                 ('S2_S4_metadata_free', [.44999, .45, .46], [0, 1, 1]),
                 ('MiniCheck_7B', [.2, .20000000000000004, .21], [1, 0, 0])]
        for name, values, expected in cases:
            with self.subTest(name=name):
                np.testing.assert_array_equal(m.system_arrays(name, values)['predictions'], expected)
        report = m.single_metrics([0, 1, 1], 'judge', [-2., -1., .5])
        self.assertEqual(report['metrics']['f1'], 1.)
        self.assertEqual(report['fixed_zero_reference_secondary']['metrics']['recall'], .5)

    def test_rank_uses_margin_not_saturated_sigmoid(self):
        result = m.single_metrics([0, 1], 'judge', [100., 200.])
        self.assertEqual(result['metrics']['auroc'], 1.)
        self.assertEqual(result['metrics']['average_precision'], 1.)
        self.assertEqual(m.system_arrays('judge', [100., 200.])['probabilities']['raw'].tolist(), [1., 1.])
        # MC ranking likewise must not gain ties from 1-p subtraction.
        result = m.single_metrics([1, 0], 'MiniCheck_7B', [1e-20, 2e-20])
        self.assertEqual(result['metrics']['auroc'], 1.)

    def test_average_precision_and_ties(self):
        result = m.single_metrics([1, 0, 1], 'S4', [.9, .8, .7])
        self.assertAlmostEqual(result['metrics']['average_precision'], (1 + 2 / 3) / 2)
        self.assertEqual(result['metrics']['auroc'], .5)
        tied = m.single_metrics([1, 0, 1, 0], 'S4', [.5] * 4)
        self.assertEqual(tied['metrics']['auroc'], .5)
        self.assertEqual(tied['metrics']['average_precision'], .5)

    def test_reliability_bins_endpoints_and_empty_bins(self):
        result = m.single_metrics([0, 0, 1, 1], 'S4', [0., .1, .9, 1.])
        raw = result['reliability']['raw']
        self.assertEqual([b['count'] for b in raw['bins']], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertAlmostEqual(raw['brier'], .005)
        self.assertAlmostEqual(raw['ece'], .05)
        self.assertIsNone(raw['bins'][2]['mean_probability'])
        self.assertTrue(raw['bins'][9]['right_inclusive'])

    def test_missing_empty_one_class_and_invalid_outputs(self):
        result = m.single_metrics([0, 1, 1], 'S4', [.1, None, .9])
        self.assertEqual((result['attempted_n'], result['n'], result['missing_n']), (3, 2, 1))
        self.assertAlmostEqual(result['coverage'], 2 / 3)
        self.assertEqual(result['metrics']['f1'], 1.)
        for labels, scores in (([0, 0], [.1, .2]), ([1, 1], [.9, .8])):
            result = m.single_metrics(labels, 'S4', scores)
            self.assertIsNone(result['metrics']['auroc']); self.assertIsNone(result['metrics']['average_precision'])
            self.assertEqual(result['undefined_ranking_reason'], 'one_class_subset')
        empty = m.single_metrics([0, 1], 'judge', [None, None])
        self.assertEqual(empty['metrics']['f1'], 0.)
        self.assertIsNone(empty['metrics']['brier_calibrated'])
        self.assertEqual(empty['undefined_ranking_reason'], 'empty_valid_subset')
        json.dumps(empty, allow_nan=False)
        for values in ([True], [float('nan')], [float('inf')], [1.1]):
            with self.assertRaises(RunConflict): m.single_metrics([1], 'S4', values)
        with self.assertRaises(RunConflict): m.single_metrics([2], 'S4', [.5])
        with self.assertRaises(RunConflict): m.single_metrics([0, 1], 'S4', [.5])

    def test_all_descriptive_slices_include_missing_and_one_class(self):
        report = m.descriptive_slices([0, 1, 0], 'judge', [-3., None, None],
                                     {'task': ['QA', 'Summary', 'Data2txt'], 'generator': ['x', 'x', 'y']})
        self.assertEqual(set(report['task']), {'QA', 'Summary', 'Data2txt'})
        self.assertEqual(report['task']['Summary']['attempted_n'], 1)
        self.assertEqual(report['task']['Summary']['n'], 0)
        self.assertIsNone(report['task']['QA']['metrics']['auroc'])

    def test_whole_group_draws_preserve_unequal_sizes_and_multiplicity(self):
        groups = ['b', 'a', 'b', 'c', 'b', 'a']; eligible = np.arange(6)
        with patch.object(m, 'REPLICATES', 8):
            names, members, draws = m.cluster_draws(groups, eligible)
            _, _, replay = m.cluster_draws(groups, eligible)
        self.assertEqual(names, ['a', 'b', 'c'])
        self.assertEqual([v.tolist() for v in members], [[1, 5], [0, 2, 4], [3]])
        np.testing.assert_array_equal(draws, replay)
        reference = np.random.Generator(np.random.PCG64(20260920)).integers(0, 3, size=(8, 3))
        np.testing.assert_array_equal(draws, reference)
        for draw in draws:
            rows = np.concatenate([members[int(i)] for i in draw])
            for i, member in enumerate(members):
                for row in member:
                    self.assertEqual(int((rows == row).sum()), int((draw == i).sum()))

    def test_minimum_valid_replicates_and_percentile_rule(self):
        insufficient = m.interval([1.] * 1899 + [None] * 101)
        self.assertIsNone(insufficient['lower']); self.assertEqual(insufficient['invalid_draws'], 101)
        values = list(range(1900))
        accepted = m.interval(values + [None] * 100)
        self.assertEqual(accepted['lower'], float(np.percentile(values, 2.5, method='linear')))
        self.assertEqual(accepted['upper'], float(np.percentile(values, 97.5, method='linear')))

    def test_shared_intersection_paired_draws_and_difference_sign(self):
        y, scores, ids, groups = paired_fixture()
        scores['S4'][0] = None
        scores['MiniCheck_7B'][1] = None
        with patch.object(m, 'REPLICATES', 8), patch.object(m, 'MIN_VALID_DRAWS', 1):
            result = m.paired_metrics(y, scores, ids, groups, ready_fixture())
        self.assertEqual(result['shared_sample_ids'], ['2', '3', '4', '5'])
        self.assertEqual(result['individual']['judge']['n'], 6)
        self.assertEqual(result['individual']['S4']['n'], 5)
        for name in m.MAIN_SYSTEMS:
            self.assertEqual(result['shared_points'][name]['n'], 4)
        for name, metrics in result['paired_differences'].items():
            self.assertEqual(metrics['auroc']['lower'], 0.)
            self.assertEqual(metrics['auroc']['upper'], 0.)
            self.assertLess(metrics['brier_raw']['point_difference'], 0.)
            self.assertFalse(metrics['brier_raw']['larger_is_better'])
            self.assertEqual(metrics['brier_calibrated']['baseline_metric'], 'brier_raw')
        json.dumps(result, allow_nan=False)

    def test_one_class_bootstrap_does_not_resample_to_force_validity(self):
        y, scores, ids, _ = paired_fixture()
        groups = ['negative', 'positive'] * 3
        with patch.object(m, 'REPLICATES', 20), patch.object(m, 'MIN_VALID_DRAWS', 19):
            result = m.paired_metrics(y, scores, ids, groups, ready_fixture())
        interval = result['intervals']['judge']['auroc']
        self.assertEqual(interval['valid_draws'] + interval['invalid_draws'], 20)
        self.assertGreater(interval['invalid_draws'], 0)
        self.assertIsNone(interval['lower'])
        self.assertEqual(result['intervals']['judge']['f1']['valid_draws'], 20)

    def test_empty_shared_subset_has_no_intervals_or_imputed_scores(self):
        y, scores, ids, groups = paired_fixture()
        scores['S4'] = [None] * len(y)
        with patch.object(m, 'REPLICATES', 8), patch.object(m, 'MIN_VALID_DRAWS', 1):
            result = m.paired_metrics(y, scores, ids, groups, ready_fixture())
        self.assertEqual(result['shared_n'], 0)
        for metric in result['intervals']['judge'].values():
            self.assertIsNone(metric['lower'])
            self.assertEqual(metric['valid_draws'], 0)
            self.assertEqual(metric['invalid_draws'], 8)
        json.dumps(result, allow_nan=False)

    def test_unready_provenance_stops_before_computing_metrics(self):
        y, scores, ids, groups = paired_fixture()
        with patch.object(m, '_point', side_effect=AssertionError('metrics ran')):
            result = m.paired_metrics(y, scores, ids, groups, {})
        self.assertFalse(result['metrics_computed'])
        scores['S2_S4_metadata'] = scores.pop('S2_S4_metadata_free')
        with self.assertRaises(RunConflict): m.paired_metrics(y, scores, ids, groups, ready_fixture())


if __name__ == '__main__': unittest.main()
