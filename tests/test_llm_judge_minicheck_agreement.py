import copy
import math
import unittest

from post_thesis.llm_judge.compare_minicheck_legacy import agreement
from post_thesis.llm_judge.minicheck_contract import SUPPORT_THRESHOLD
from post_thesis.llm_judge.storage import RunConflict


class ScoreOnly(dict):
    def __getitem__(self,key):
        if key not in ('idx','minicheck_score'): raise AssertionError('accessed label/metadata: '+key)
        return super().__getitem__(key)


def fixtures(new=(.800012,.1),old=(.8,.3)):
    targets=[{'sample_id':str(i),'test_index':i,'input_sha256':str(i)*64} for i in range(len(new))]
    fresh=[{**r,'status':'ok','support_score':p,'frozen_predicted_unsupported':p<SUPPORT_THRESHOLD}
           for r,p in zip(targets,new)]
    legacy=[ScoreOnly(idx=i,minicheck_score=p,ground_truth_hallucination='must not read',task_type='must not read')
            for i,p in enumerate(old)]
    return fresh,legacy,targets


class MiniCheckAgreementTests(unittest.TestCase):
    def test_known_differences_and_no_label_access(self):
        fresh,legacy,targets=fixtures()
        result=agreement(fresh,list(reversed(legacy)),targets)
        self.assertEqual(result['matching_after_rounding_fresh_to_4dp'],1)
        self.assertEqual(result['fixed_threshold_decision_changes'],1)
        self.assertEqual(result['differences_above_0_1'],1)
        self.assertAlmostEqual(result['mean_absolute_difference'],(.000012+.2)/2)
        self.assertEqual(result['largest_differences'][0]['test_index'],1)
        self.assertFalse(result['labels_used'])

    def test_strict_original_float_threshold(self):
        t=SUPPORT_THRESHOLD
        fresh,legacy,targets=fixtures((t,math.nextafter(t,-math.inf)),(t,t))
        result=agreement(fresh,legacy,targets)
        self.assertEqual(result['fixed_threshold_decision_changes'],1)
        self.assertEqual(result['fixed_threshold']['hex'],'0x1.999999999999bp-3')
        fresh[0]['frozen_predicted_unsupported']=True
        with self.assertRaises(RunConflict): agreement(fresh,legacy,targets)

    def test_missing_duplicate_reordered_or_misaligned_rows_rejected(self):
        a,b,c=fixtures()
        cases=[(a[:-1],b,c),(list(reversed(a)),b,c),(a,[b[0],b[0]],c)]
        changed=copy.deepcopy(a);changed[0]['input_sha256']='different';cases.append((changed,b,c))
        changed=copy.deepcopy(a);changed[0]['status']='error';cases.append((changed,b,c))
        for args in cases:
            with self.assertRaises(RunConflict): agreement(*args)

    def test_invalid_scores_never_become_zero(self):
        for value in (None,float('nan'),float('inf'),-.1,1.1,True):
            a,b,c=fixtures();a[0]['support_score']=value
            with self.assertRaises(RunConflict): agreement(a,b,c)
            a,b,c=fixtures();b[0]['minicheck_score']=value
            with self.assertRaises(RunConflict): agreement(a,b,c)

    def test_identical_scores_are_not_claimed_historical_proof(self):
        result=agreement(*fixtures((.1,.8),(.1,.8)))
        self.assertEqual(result['maximum_absolute_difference'],0.)
        self.assertEqual(result['fixed_threshold_decision_changes'],0)
        self.assertFalse(result['historical_provenance_proven_by_agreement'])


if __name__=='__main__': unittest.main()
