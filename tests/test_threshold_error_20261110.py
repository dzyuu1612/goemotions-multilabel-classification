"""Scientific edge cases for the saved-score error audit."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import tempfile
P=Path(__file__).resolve().parents[1]/'experiments/threshold_error_20261110/analysis.py'
spec=importlib.util.spec_from_file_location('threshold_error_analysis',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class ErrorAuditTests(unittest.TestCase):
 def test_manual_four_sentence_metrics(self):
  y=np.array([[1,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]],dtype=bool)
  p=np.array([[1,0,0,0],[0,0,1,0],[0,0,0,0],[1,1,0,0]],dtype=bool)
  m=a.metrics(y,p)
  self.assertAlmostEqual(m['macro_f1'],.5);self.assertAlmostEqual(m['micro_f1'],2/3);self.assertAlmostEqual(m['hamming'],3/16)
 def test_taxonomy_is_exclusive_without_losing_fp_in_partial(self):
  y=np.array([[1,0,0],[1,0,0],[1,1,0],[1,0,0],[1,0,0],[1,1,0]],dtype=bool)
  p=np.array([[1,0,0],[1,1,0],[1,0,0],[0,1,0],[0,0,0],[1,0,1]],dtype=bool)
  self.assertEqual(a.taxonomy(y,p).tolist(),['dung_hoan_toan','doan_thua','dung_mot_phan','doan_sai_nhan','khong_doan_gi','dung_mot_phan'])
  tp,fp,fn=a.counts(y[3:4],p[3:4]);self.assertEqual(int(fp.sum()),1);self.assertEqual(int(fn.sum()),1)
 def test_tie_prefers_near_half_then_higher(self):
  self.assertGreater(a.tie(.4,.5),a.tie(.4,.2));self.assertGreater(a.tie(.4,.6),a.tie(.4,.4))
 def test_all_28_unsupported_labels_keep_half(self):
  s=np.zeros((3,28));y=np.zeros((3,28),dtype=bool)
  per,glob,curve=a.select(s,y)
  np.testing.assert_array_equal(per,np.full(28,.5));self.assertEqual(glob,.5)
 def test_agreement_requires_original_complete_items_and_real_labels(self):
  spec=importlib.util.spec_from_file_location('annotation_agreement',P.with_name('annotation_agreement.py'))
  agreement=importlib.util.module_from_spec(spec);spec.loader.exec_module(agreement)
  with tempfile.TemporaryDirectory() as folder:
   folder=Path(folder);ref=folder/'registry.csv';first=folder/'first.csv';second=folder/'second.csv'
   data=pd.DataFrame({'annotation_id':['N0001','N0002'],'pair':['relief->admiration']*2,'id':['a','b'],'text':['one','two'],'true_labels':['relief']*2,'predicted_labels':['admiration']*2,'nhan_thu_cong':['','']})
   data.to_csv(ref,index=False);data.to_csv(first,index=False);data.to_csv(second,index=False)
   with self.assertRaises(AssertionError):agreement.compare(first,second,ref)
   data['nhan_thu_cong']=['thieu_ngu_canh','khong_du_can_cu'];data.to_csv(first,index=False);data.to_csv(second,index=False)
   self.assertEqual(agreement.compare(first,second,ref)['agreement_rate'],1.)
   data.iloc[:1].to_csv(first,index=False);data.iloc[:1].to_csv(second,index=False)
   with self.assertRaises(AssertionError):agreement.compare(first,second,ref)
   data.loc[0,'text']='changed';data.to_csv(first,index=False);data.to_csv(second,index=False)
   with self.assertRaises(AssertionError):agreement.compare(first,second,ref)
if __name__=='__main__':unittest.main()
