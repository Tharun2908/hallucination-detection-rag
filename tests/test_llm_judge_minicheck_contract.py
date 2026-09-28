import math
import unittest
from post_thesis.llm_judge.minicheck_contract import aggregate_support, support_from_returned_tokens, SUPPORT_THRESHOLD
from post_thesis.llm_judge.storage import RunConflict


class MiniCheckContractTests(unittest.TestCase):
    def test_minimum_of_per_sentence_maxima_not_global_maximum(self):
        result = aggregate_support([[.9,.2],[.4,.7]])
        self.assertEqual(result['best_support_per_answer_sentence'],[.9,.7])
        self.assertEqual(result['support_score'],.7)
        self.assertTrue(result['upstream_supported_label'])

    def test_probability_mass_is_not_yes_no_renormalized(self):
        tokens = [{'decoded_token':'Yes','logprob':math.log(.3)},
                  {'decoded_token':'yes','logprob':math.log(.1)},
                  {'decoded_token':'No','logprob':math.log(.2)},
                  {'decoded_token':' Yes','logprob':math.log(.1)}]
        self.assertAlmostEqual(support_from_returned_tokens(tokens),.4)

    def test_absent_yes_is_zero_but_missing_response_is_error(self):
        self.assertEqual(support_from_returned_tokens([{'decoded_token':'No','logprob':-.1}]),0.)
        with self.assertRaises(RunConflict): support_from_returned_tokens([])

    def test_empty_ragged_and_bad_scores_rejected(self):
        for matrix in ([],[[]],[[.2],[.3,.4]],[[float('nan')]],[[float('inf')]],[[True]],[[-.1]],[[1.1]]):
            with self.assertRaises(RunConflict): aggregate_support(matrix)
        for value in (float('nan'),float('inf'),True,.1):
            with self.assertRaises(RunConflict):
                support_from_returned_tokens([{'decoded_token':'Yes','logprob':value}])

    def test_strict_frozen_threshold_and_upstream_half_boundary(self):
        self.assertFalse(aggregate_support([[SUPPORT_THRESHOLD]])['frozen_predicted_unsupported'])
        self.assertTrue(aggregate_support([[math.nextafter(SUPPORT_THRESHOLD,0.)]])['frozen_predicted_unsupported'])
        self.assertFalse(aggregate_support([[.5]])['upstream_supported_label'])


if __name__ == '__main__': unittest.main()
