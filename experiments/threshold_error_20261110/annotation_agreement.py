"""Compute agreement only from two completed, independently supplied forms."""
from pathlib import Path
import argparse,json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
ALLOWED={'thieu_ngu_canh','giao_thoa_sac_thai','phu_dinh_mia_mai','van_ban_bat_thuong','khong_du_can_cu'}
def compare(first,second,registry=None):
 frames=[pd.read_csv(p,keep_default_na=False).set_index('annotation_id') for p in [first,second]]
 a,b=frames
 assert a.index.is_unique and b.index.is_unique and len(a)>0 and set(a.index)==set(b.index),'Different/duplicate annotation IDs'
 if registry is None:
  here=Path(__file__).resolve().parent
  registry=next((here/p/'annotation_sampling_registry.csv' for p in ['results','so_lieu'] if (here/p/'annotation_sampling_registry.csv').exists()),None)
  assert registry is not None,'Original sampling registry required'
 reference=pd.read_csv(registry,keep_default_na=False).set_index('annotation_id')
 assert reference.index.is_unique and set(a.index)==set(reference.index),'Missing or unexpected items versus original registry'
 reference=reference.loc[a.index]
 b=b.loc[a.index]
 for col in ['pair','id','text','true_labels','predicted_labels']:
  assert a[col].equals(b[col]) and a[col].equals(reference[col]),'Different original context: '+col
 for f in frames:assert set(f.nhan_thu_cong)<=ALLOWED,'Incomplete/invalid labels; do not compute agreement'
 yes=int((a.nhan_thu_cong==b.nhan_thu_cong).sum());k=float(cohen_kappa_score(a.nhan_thu_cong,b.nhan_thu_cong))
 return {'n_items':len(a),'n_agree':yes,'agreement_rate':yes/len(a),'cohen_kappa':k if np.isfinite(k) else None,'note':'Agreement of supplied labels; independence must be ensured by the human raters, and agreement does not establish causality.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second');p.add_argument('out');v=p.parse_args()
 result=compare(v.first,v.second);Path(v.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result)
