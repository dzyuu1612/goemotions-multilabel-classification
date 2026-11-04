"""Recompute the current report tables from the nine saved test-score arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,hamming_loss

ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'main_results'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
 lock=read(MAIN/'LOCK_BEFORE_TEST.json')
 assert sha(MAIN/'selection.json')==lock['selection_sha256']
 test=pd.read_parquet(MAIN/'data/raw/test-00000-of-00001.parquet')
 labels_by_id=dict(zip(test.id.astype(str),test.labels))
 saved=pd.read_csv(MAIN/'report_data/all_runs.csv'); summaries=pd.read_csv(MAIN/'report_data/mean_std.csv')
 records=[]
 for run in lock['runs']:
  path=MAIN/Path(run['protocol'].replace('\\','/'));assert sha(path)==run['protocol_sha256'];protocol=read(path)
  receipt=read(path.parent/'test_results.json');scores_file=path.parent/'test_scores.npz';assert sha(scores_file)==receipt['scores_sha256']
  with np.load(scores_file,allow_pickle=False) as array:
   ids=array['ids'].astype(str);scores=array['scores'];names=array['label_names'].astype(str).tolist()
  assert names==protocol['label_names'] and scores.shape==(5427,28)
  truth=np.zeros_like(scores,dtype=np.uint8)
  for i,identifier in enumerate(ids):truth[i,labels_by_id[identifier]]=1
  for config in protocol['configurations']:
   mode=config['threshold_mode']
   if mode not in ['fixed','tuned']:continue
   predicted=scores>=np.asarray(config['thresholds'])
   row={'system':'C_'+run['architecture'],'seed':run['seed'],'mode':mode}
   for average in ['macro','micro']:
    for metric,fn in [('f1',f1_score),('precision',precision_score),('recall',recall_score)]:row[average+'_'+metric]=fn(truth,predicted,average=average,zero_division=0)
   row['hamming_loss']=hamming_loss(truth,predicted)
   expected=saved[(saved.system==row['system'])&(saved.seed==row['seed'])&(saved.split=='test')&(saved.threshold_mode==mode)]
   assert len(expected)==1
   for key in ['macro_f1','micro_f1','macro_precision','macro_recall','micro_precision','micro_recall','hamming_loss']:assert abs(row[key]-expected.iloc[0][key])<1e-12,(run['architecture'],run['seed'],mode,key)
   records.append(row)
 actual=pd.DataFrame(records)
 for (system,mode),group in actual.groupby(['system','mode']):
  assert set(group.seed)=={42,123,2026}
  expected=summaries[(summaries.system==system)&(summaries.split=='test')&(summaries.threshold_mode==mode)].iloc[0]
  for key in ['macro_f1','micro_f1','hamming_loss']:
   assert abs(group[key].mean()-expected[key+'_mean'])<1e-12
   assert abs(group[key].std(ddof=1)-expected[key+'_std'])<1e-12
 historical=pd.read_csv(MAIN/'report_data/original_test_summary.csv')
 assert len(historical)==6 and set(historical.threshold_mode)=={'fixed','tuned'}
 repository=ROOT.parents[1]
 original=pd.read_csv(repository/'results/final_submission/report_data/mean_std.csv')
 for row in historical.to_dict('records'):
  ref=original[(original.system==row['system'])&(original.split=='test')&(original.threshold_mode==row['threshold_mode'])].iloc[0]
  for key in ['macro_f1_mean','macro_f1_std','micro_f1_mean','micro_f1_std','hamming_loss_mean','hamming_loss_std']:assert abs(row[key]-ref[key])<1e-12
 print('Passed: 18 test rows × 7 metrics, mean/sample std, and six historical rows. No training or threshold fitting.')
 return {'passed':True,'test_rows':len(records),'metrics_per_row':7,'historical_rows':len(historical),'training':False,'test_used_to_select':False}

if __name__=='__main__':main()
