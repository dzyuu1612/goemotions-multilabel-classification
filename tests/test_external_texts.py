"""Kiểm mẫu ngoài dữ liệu và đếm lỗi theo điểm chưa làm tròn."""
import copy
import unittest

import numpy as np
import pandas as pd

from scripts.analysis.evaluate_external_texts import (
    FIXTURE, boundary_check, check_fixture, read, summarize,
)


class ExternalTextsTest(unittest.TestCase):
    labels = ['admiration','amusement','anger','annoyance','approval','caring','confusion',
              'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
              'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
              'pride','realization','relief','remorse','sadness','surprise','neutral']

    def test_provenance_and_review_status_are_explicit(self):
        fixture=read(FIXTURE)
        rows=check_fixture(fixture,self.labels)
        self.assertEqual(sum(r['kind']=='external_review' for r in rows),20)
        self.assertTrue(all(r.get('source_line',0)>0 for r in rows if r['kind']=='external_review'))
        changed=copy.deepcopy(fixture)
        changed['human_review_status']='VERIFIED'
        with self.assertRaises(ValueError):check_fixture(changed,self.labels)

    def test_normalized_overlap_is_not_silently_removed(self):
        rows=[dict(id='E01',text=' Hello   world! ')]
        result=boundary_check(rows,{'train':pd.DataFrame({'text':['hello world!']})})
        self.assertEqual(result['train']['exact_matches'],[])
        self.assertEqual(result['train']['normalized_matches'],['E01'])

    def test_error_counts_use_full_precision_and_separate_sets(self):
        rows=[dict(id='E01',kind='external_review',labels=['joy'],ambiguous=False),
              dict(id='P1a',kind='controlled_probe',labels=['neutral'],ambiguous=False)]
        scores=np.zeros((2,28));scores[0,17]=.49999;scores[0,4]=.5;scores[1,27]=.8
        metrics,pred=summarize(rows,self.labels,[dict(model='test',seed=None,scores=scores,
                                 thresholds={'fixed':.5,'tuned':np.full(28,.4)})])
        self.assertEqual((metrics[0]['tp'],metrics[0]['fp'],metrics[0]['fn']),(0,1,1))
        self.assertAlmostEqual(metrics[0]['hamming_loss'],2/28)
        self.assertEqual(pred[0]['fn'],'joy')
        self.assertEqual(metrics[1]['exact_match_count'],1)
        self.assertEqual(metrics[2]['fn'],0)


if __name__=='__main__':unittest.main()
