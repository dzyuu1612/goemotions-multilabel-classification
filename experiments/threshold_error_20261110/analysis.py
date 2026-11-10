"""Audit saved validation thresholds, freeze them, then describe test errors.

Run from the repository root:
 python experiments/threshold_error_20261110/analysis.py freeze
 git add experiments/threshold_error_20261110/validation_lock.json
 git commit -m "Freeze audited validation thresholds"
 python experiments/threshold_error_20261110/analysis.py evaluate

No model loading or training. Test input is opened only by evaluate, after the
lock is committed. Old test results had already been observed before this work.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/best_epochs_20261103'
GRID=np.round(np.arange(.05,.951,.05),2)
RARE=['grief','pride','relief','nervousness','embarrassment']
PAIRS=[('relief','admiration'),('neutral','approval')]
CATEGORIES=['dung_hoan_toan','doan_thua','dung_mot_phan','doan_sai_nhan','khong_doan_gi']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p):return Path(p).relative_to(ROOT).as_posix()
def records():
 runs=[]
 for arch,name in [('bert','C1 BERT'),('roberta','C2 RoBERTa'),('distilbert','C3 DistilBERT')]:
  for seed in [42,123,2026]:
   folder=OLD/f'main_results/{arch}/seed_{seed}'
   runs.append({'model':name,'architecture':arch,'seed':seed,'val_scores':relative(OLD/f'main_results/training_receipts/{arch}/seed_{seed}/validation_scores.npz'),'test_scores':relative(folder/'test_scores.npz'),'protocol':relative(folder/'final_protocol.json')})
 runs+=[{'model':'A TF-IDF balanced','architecture':'baseline','seed':42,'val_scores':'data/processed/baseline/balanced/full/validation_scores.npz','test_scores':'data/processed/baseline/final/balanced_test_scores.npz','protocol':'data/processed/baseline/balanced/full/thresholds_validation.json'},
        {'model':'B BART-MNLI','architecture':'zero_shot','seed':None,'val_scores':'artifacts/final_submission/data/processed/zero_shot/full/validation_scores.npz','test_scores':'artifacts/final_submission/data/processed/zero_shot/full/test_scores.npz','protocol':'artifacts/final_submission/data/processed/zero_shot/full/final_protocol.json'}]
 return runs
def load_scores(path,frame,names):
 with np.load(path,allow_pickle=False) as z:
  ids=z['ids'].astype(str);labels=z['label_names'].astype(str).tolist();s=z['scores'].astype(np.float64)
 assert labels==names and s.shape==(len(frame),28),str(path)
 assert len(set(ids))==len(ids) and set(ids)==set(frame['id'].astype(str)),str(path)
 assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all(),str(path)
 pos={v:i for i,v in enumerate(ids)}
 return s[[pos[v] for v in frame['id'].astype(str)]]
def targets(frame):
 y=np.zeros((len(frame),28),dtype=bool)
 for i,labs in enumerate(frame['labels']):
  assert len(labs)>0 and len(set(labs))==len(labs)
  y[i,np.asarray(labs,dtype=int)]=True
 return y
def counts(y,p):return (y&p).sum(0),(~y&p).sum(0),(y&~p).sum(0)
def f1(tp,fp,fn):
 d=2*tp+fp+fn
 return np.divide(2*tp,d,out=np.zeros_like(d,dtype=float),where=d!=0)
def metrics(y,p):
 tp,fp,fn=counts(y,p);TP,FP,FN=map(int,[tp.sum(),fp.sum(),fn.sum()])
 d=2*TP+FP+FN
 return {'macro_f1':float(f1(tp,fp,fn).mean()),'micro_f1':2*TP/d if d else 0.,'hamming':float((y!=p).mean()),'micro_p':TP/(TP+FP) if TP+FP else 0.,'micro_r':TP/(TP+FN) if TP+FN else 0.,'n_fp':FP,'n_fn':FN,'n_tp':TP,'exact_match':float((y==p).all(1).mean())}
def tie(score,t):return (round(float(score),12),-round(abs(float(t)-.5),10),float(t))
def select(s,y):
 per=[];curve=[]
 for j in range(28):
  cand=[(float(f1(*counts(y[:,j:j+1],s[:,j:j+1]>=t))[0]),t) for t in GRID]
  per.append(max(cand,key=lambda x:tie(*x))[1])
 for t in GRID:curve.append({'threshold':float(t),**metrics(y,s>=t)})
 glob=max(curve,key=lambda x:tie(x['macro_f1'],x['threshold']))['threshold']
 return np.asarray(per),glob,curve
def frozen_thresholds(proto):
 if 'configurations' in proto:
  cfg={r['threshold_mode']:r['thresholds'] for r in proto['configurations']}
  return np.asarray(cfg['tuned'],dtype=float),float(cfg['global'])
 return np.asarray(proto['thresholds'],dtype=float),float(proto['global_threshold'])
def freeze():
 valpath=ROOT/'data/raw/validation-00000-of-00001.parquet';frame=pd.read_parquet(valpath)
 assert len(frame)==5426
 runs=records();names=read(ROOT/runs[0]['protocol'])['label_names'];y=targets(frame)
 out={'recorded_at':datetime.now(timezone.utc).isoformat(),'previous_test_seen':True,'selection_split':'validation','grid':GRID.tolist(),'tie_break':'maximum F1; nearest .5; higher threshold','validation_data':relative(valpath),'validation_data_sha256':sha(valpath),'label_names':names,'checkpoints':[],'manual_annotation_completed':False}
 curves=[];mismatches=[]
 for r in runs:
  proto=read(ROOT/r['protocol']);assert proto['label_names']==names
  s=load_scores(ROOT/r['val_scores'],frame,names)
  th,tg,curve=select(s,y);saved,sg=frozen_thresholds(proto)
  diff=np.flatnonzero(np.abs(saved-th)>1e-6)
  global_diff=abs(sg-tg)>1e-6
  print(f"{r['model']}/{r['seed']}: ngưỡng riêng lệch {len(diff)}/28; chung {tg:.2f}, đã lưu {sg:.2f}",flush=True)
  if len(diff) or global_diff:mismatches.append({'run':r,'labels':[names[i] for i in diff],'saved':saved.tolist(),'recomputed':th.tolist(),'global_saved':sg,'global_recomputed':tg})
  if proto.get('validation_scores_sha256'):assert sha(ROOT/r['val_scores'])==proto['validation_scores_sha256']
  out['checkpoints'].append({**r,'validation_scores_sha256':sha(ROOT/r['val_scores']),'protocol_sha256':sha(ROOT/r['protocol']),'thresholds':saved.tolist(),'global_threshold':sg,'selected_epoch':proto.get('selected_epoch'),'validation_fixed':metrics(y,s>=.5),'validation_global':metrics(y,s>=sg),'validation_tuned':metrics(y,s>=saved)})
  curves.extend({**{'model':r['model'],'seed':r['seed']},**c} for c in curve)
 pd.DataFrame(curves).to_csv(HERE/'validation_curves.csv',index=False,encoding='utf-8-sig')
 write(HERE/'threshold_audit.json',{'passed':not mismatches,'mismatches':mismatches,'runs':len(runs),'test_opened':False})
 if mismatches:raise RuntimeError('Ngưỡng lệch. Dừng trước test; không tự thay ngưỡng đã lưu.')
 write(HERE/'validation_lock.json',out)
 print('Validation audit passed; lock saved. Commit lock before evaluate.')
def taxonomy(y,p):
 truth=y.sum(1);n=p.sum(1);inter=(y&p).sum(1)
 cat=np.full(len(y),'dung_mot_phan',dtype=object)
 cat[(inter==truth)&(n==truth)]='dung_hoan_toan'
 cat[(inter==truth)&(n>truth)]='doan_thua'
 cat[(inter==0)&(n>0)]='doan_sai_nhan'
 cat[n==0]='khong_doan_gi'
 assert sum(np.count_nonzero(cat==c) for c in CATEGORIES)==len(y)
 return cat
def evaluate():
 lockpath=HERE/'validation_lock.json';lock=read(lockpath)
 committed=subprocess.run(['git','diff','HEAD','--',relative(lockpath)],cwd=ROOT,capture_output=True,text=True,check=True).stdout
 tracked=subprocess.run(['git','ls-files','--error-unmatch',relative(lockpath)],cwd=ROOT,capture_output=True,text=True)
 assert tracked.returncode==0 and not committed,'Commit the validation lock before test evaluation.'
 assert sha(ROOT/lock['validation_data'])==lock['validation_data_sha256']
 names=lock['label_names'];rare=np.array([names.index(n) for n in RARE]);out=HERE/'results';out.mkdir(exist_ok=True)
 testpath=OLD/'main_results/data/raw/test-00000-of-00001.parquet';frame=pd.read_parquet(testpath);assert len(frame)==5427
 y=targets(frame);v=targets(pd.read_parquet(ROOT/lock['validation_data']))
 rows=[];labels=[];thresholds=[];taxrows=[];overlaps=[];near=[];pairrows=[];samples=[];receipts=[];examples=[]
 for r in lock['checkpoints']:
  assert sha(ROOT/r['val_scores'])==r['validation_scores_sha256'] and sha(ROOT/r['protocol'])==r['protocol_sha256']
  s=load_scores(ROOT/r['test_scores'],frame,names)
  th=np.asarray(r['thresholds']);modes={'fixed':np.full(28,.5),'global':np.full(28,r['global_threshold']),'tuned':th}
  base={'model':r['model'],'seed':r['seed']}
  for mode,ts in modes.items():
   p=s>=ts;m=metrics(y,p)
   assert abs(m['macro_f1']-f1_score(y,p,average='macro',zero_division=0))<1e-12
   assert abs(m['micro_f1']-f1_score(y,p,average='micro',zero_division=0))<1e-12
   assert abs(m['hamming']-hamming_loss(y,p))<1e-12
   assert abs(m['micro_p']-precision_score(y,p,average='micro',zero_division=0))<1e-12
   assert abs(m['micro_r']-recall_score(y,p,average='micro',zero_division=0))<1e-12
   rows.append({**base,'mode':mode,'global_t':r['global_threshold'] if mode=='global' else np.nan,**m})
   tp,fp,fn=counts(y,p)
   for j,n in enumerate(names):labels.append({**base,'mode':mode,'label':n,'tp':int(tp[j]),'fp':int(fp[j]),'fn':int(fn[j]),'precision':float(tp[j]/(tp[j]+fp[j])) if tp[j]+fp[j] else 0.,'recall':float(tp[j]/(tp[j]+fn[j])) if tp[j]+fn[j] else 0.,'f1':float(f1(tp,fp,fn)[j])})
   cat=taxonomy(y,p)
   for c in CATEGORIES:taxrows.append({**base,'mode':mode,'category':c,'n':int((cat==c).sum()),'denom':len(y)})
   inter=(y&p).sum(1);t=y.sum(1);FNm=y&~p;FPm=~y&p
   g1=(t>=2)&(inter>=1)&(inter<t);g2=FNm[:,rare].any(1);g3=FNm.any(1)&FPm.any(1)
   ov={**base,'mode':mode,'g1':int(g1.sum()),'g1_denom':int((t>=2).sum()),'g2':int(g2.sum()),'g2_denom':int(y[:,rare].any(1).sum()),'g3':int(g3.sum()),'g3_denom':len(y),'g1_and_g2':int((g1&g2).sum()),'g1_and_g3':int((g1&g3).sum()),'g2_and_g3':int((g2&g3).sum()),'all_three':int((g1&g2&g3).sum()),'union':int((g1|g2|g3).sum())}
   bits=g1.astype(int)+2*g2.astype(int)+4*g3.astype(int)
   for b in range(1,8):ov[f'only_mask_{b}']=int((bits==b).sum())
   assert sum(ov[f'only_mask_{b}'] for b in range(1,8))==ov['union'];overlaps.append(ov)
   for typ,mask in [('FN',FNm),('FP',FPm)]:
    i,j=np.where(mask);sc=s[i,j];limits=ts[j]
    if typ=='FN':a=sc/limits;conditions=[a>=.8,(a>=.5)&(a<.8),a<.5];keys=['ratio_ge_0_8','ratio_0_5_to_0_8','ratio_lt_0_5']
    else:a=sc-limits;conditions=[a<=.1,(a>.1)&(a<=.3),a>.3];keys=['margin_le_0_1','margin_0_1_to_0_3','margin_gt_0_3']
    assert sum(int(c.sum()) for c in conditions)==len(i)
    near.append({**base,'mode':mode,'type':typ,'n':len(i),**{f'n_{k}':int(c.sum()) for k,c in zip(keys,conditions)},**{f'share_{k}':float(c.mean()) if len(i) else np.nan for k,c in zip(keys,conditions)}})
   # A fixed sample is drawn for each predeclared pair, never to pick thresholds.
   if mode=='tuned':
    M=FNm.astype(np.int64).T@FPm.astype(np.int64);np.fill_diagonal(M,0)
    slug=r['model'].replace(' ','_').replace('-','_');seed='single' if r['seed'] is None else str(r['seed'])
    pd.DataFrame(M,index=names,columns=names).to_csv(out/f'pair_counts_{slug}_{seed}.csv',encoding='utf-8-sig')
    for j,k in zip(*np.where(M>0)):pairrows.append({**base,'fn_label':names[j],'fp_label':names[k],'n':int(M[j,k])})
    rng=np.random.default_rng(20261107)
    for a,b in PAIRS:
     j,k=names.index(a),names.index(b);idx=np.flatnonzero(FNm[:,j]&FPm[:,k]);pick=np.sort(rng.choice(idx,min(20,len(idx)),replace=False))
     for i in pick:samples.append({**base,'pair':a+'->'+b,'row':int(i),'id':str(frame.iloc[i]['id']),'text':frame.iloc[i]['text'],'true_labels':'|'.join(np.asarray(names)[y[i]]),'predicted_labels':'|'.join(np.asarray(names)[p[i]]),'score_fn':float(s[i,j]),'threshold_fn':float(th[j]),'score_fp':float(s[i,k]),'threshold_fp':float(th[k])})
   if r['seed']==2026 and mode in ['fixed','tuned']:
    q=np.flatnonzero(frame['id'].astype(str).to_numpy()=='eeq65gb');assert len(q)==1;i=int(q[0]);examples.append({**base,'mode':mode,'id':'eeq65gb','caring':float(s[i,names.index('caring')]),'optimism':float(s[i,names.index('optimism')]),'truth':'|'.join(np.asarray(names)[y[i]]),'prediction':'|'.join(np.asarray(names)[p[i]])})
  for j,n in enumerate(names):thresholds.append({**base,'label':n,'threshold_label':float(th[j]),'global_threshold':r['global_threshold'],'support_val':int(v[:,j].sum()),'support_test':int(y[:,j].sum()),'is_rare':n in RARE})
  receipts.append({**base,'test_scores_sha256':sha(ROOT/r['test_scores']),'validation_scores_sha256':r['validation_scores_sha256']})
 for filename,data in [('modes_per_checkpoint',rows),('thresholds_28',thresholds),('label_tpfpfn',labels),('error_taxonomy',taxrows),('error_overlap',overlaps),('near_miss',near),('pair_counts_long',pairrows),('sample_pairs',samples),('caring_optimism_verified',examples)]:pd.DataFrame(data).to_csv(out/(filename+'.csv'),index=False,encoding='utf-8-sig')
 df=pd.DataFrame(rows);stats=[]
 for (model,mode),g in df.groupby(['model','mode'],sort=False):
  stats.append({'model':model,'mode':mode,'n_runs':len(g),**{f'{k}_{stat}':float(g[k].mean() if stat=='mean' else g[k].std(ddof=1)) for k in ['macro_f1','micro_f1','hamming','micro_p','micro_r','n_fp','n_fn','exact_match'] for stat in ['mean','std']}})
 pd.DataFrame(stats).to_csv(out/'modes_mean_std.csv',index=False,encoding='utf-8-sig')
 pool=pd.DataFrame(samples).drop_duplicates(['id','pair']).sort_values(['pair','row']);pool['annotation_id']=['N%04d'%i for i in range(1,len(pool)+1)]
 blind=pool[['annotation_id','pair','id','text','true_labels','predicted_labels']].copy();blind['nhan_thu_cong']='';blind['ghi_chu']=''
 for person in ['Duy','Huy']:blind.to_csv(out/f'phieu_doc_lap_{person}.csv',index=False,encoding='utf-8-sig')
 pool.to_csv(out/'annotation_sampling_registry.csv',index=False,encoding='utf-8-sig')
 sweep=pd.read_csv(OLD/'sweep_log.csv');assert len(sweep)==72
 for field in ['val_macro_f1','val_bce']:sweep.pivot(index=['model','seed'],columns='epoch',values=field).to_csv(out/f'epoch_table_{field}.csv',encoding='utf-8-sig')
 write(out/'verification.json',{'passed':True,'validation_lock_sha256':sha(lockpath),'test_data_sha256':sha(testpath),'checkpoints':receipts,'n_test':5427,'n_labels':28,'scikit_learn_metrics_checked':len(rows),'taxonomy_totals_checked':len(taxrows)//5,'manual_annotation_rows':len(pool),'manual_annotation_completed':False,'no_training':True,'test_used_for_selection':False,'previous_test_seen':True})
 print(pd.DataFrame(stats)[['model','mode','n_runs','macro_f1_mean','macro_f1_std','micro_f1_mean','hamming_mean']].to_string(index=False))
 print(f'Done: {len(rows)} evaluations; {len(pool)} distinct annotation items; no human annotations invented.')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['freeze','evaluate']);args=parser.parse_args()
 freeze() if args.phase=='freeze' else evaluate()
