"""Tính lại bộ nộp từ scores và nhãn gốc, giữ rõ nguồn từng kiến trúc."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.paths import ROOT
from src.datasets.goemotions import load_goemotions,multi_hot,sha256
from src.models.baseline import load_aligned_scores
from src.evaluation.metrics import evaluate_multilabel
from scripts.analysis.summarize_project import summarize

SEEDS=(42,123,2026)
METRICS=('macro_f1','micro_f1','macro_precision','macro_recall','micro_precision','micro_recall','hamming_loss')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def verify(root,output,export=False,full=False):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    frames,labels,manifest=load_goemotions(root,write_metadata=False)
    truth={k:multi_hot(v.labels.tolist(),28) for k,v in frames.items()}
    support=truth['train'].sum(0);rareidx=np.argsort(support,kind='stable')[:5];rarel=[labels[i] for i in rareidx]
    records=[];per=[];histories=[];inputs={};saved_scores={};configs={}
    def measure(system,seed,folder,choices,paths):
        proto=folder/'final_protocol.json';inputs[str(proto.relative_to(root))]=sha256(proto)
        for split,path in paths.items():
            scores=load_aligned_scores(path,frames[split].id.tolist(),labels);inputs[str(path.relative_to(root))]=sha256(path)
            saved_scores[(system,seed,split)]=scores
            for choice in choices:
                m=evaluate_multilabel(truth[split],scores,labels,choice['thresholds'])
                records.append(dict(system=system,seed=seed,split=split,threshold_mode=choice['threshold_mode'],n_samples=len(scores),**{k:m[k] for k in METRICS}))
                per.extend(dict(system=system,architecture=system.removeprefix('C_'),seed=seed,split=split,threshold_mode=choice['threshold_mode'],train_support=int(support[r['label_id']]),**r) for r in m['per_label'])
    base=root/'data/processed/baseline';proto=read(base/'final_protocol.json')
    for variant in ['standard','balanced']:
        folder=base/('full' if variant=='standard' else 'balanced/full')
        measure('A_'+variant,42,base,[c for c in proto['configurations'] if c['variant']==variant],{'validation':folder/'validation_scores.npz','test':base/f'final/{variant}_test_scores.npz'})
    base=root/'data/processed/zero_shot/full';proto=read(base/'final_protocol.json')
    measure('B_bart_mnli',None,base,proto['configurations'],{s:base/(s+'_scores.npz') for s in ['validation','test']})
    for arch in ['bert','roberta','distilbert']:
        for seed in SEEDS:
            folder=root/f'data/processed/transformers/{arch}/seed_{seed}/full/standard';meta=read(folder/'run_metadata.json');proto=read(folder/'final_protocol.json')
            if not meta['completed'] or meta['smoke'] or meta['label_names']!=labels:raise ValueError('Run C không hợp lệ')
            inputs[str((folder/'run_metadata.json').relative_to(root))]=sha256(folder/'run_metadata.json')
            measure('C_'+arch,seed,folder,proto['configurations'],{s:folder/(s+'_scores.npz') for s in ['validation','test']})
            configs[(arch,seed)]=meta
            for row in meta['history']:
                histories.append(dict(architecture=arch,seed=seed,epoch=row['epoch'],train_loss=row['train_loss'],validation_macro_f1=row['macro_f1']))
    runs=pd.DataFrame(records);means=pd.DataFrame(summarize(records));allper=pd.DataFrame(per)
    expected=pd.read_csv(root/'reports/project_results/all_runs.csv')
    keys=['system','seed','split','threshold_mode'];left=runs.copy();right=expected.copy();left.seed=left.seed.fillna(-1);right.seed=right.seed.fillna(-1)
    merged=left.merge(right,on=keys,suffixes=('_new','_saved'),validate='one_to_one')
    if len(merged)!=72 or len(runs)!=72 or len(means)!=36:raise ValueError('Thiếu hoặc trùng cấu hình bộ nộp')
    delta=max(float(np.abs(merged[k+'_new']-merged[k+'_saved']).max()) for k in METRICS)
    if delta>1e-12:raise ValueError('Chỉ số khác bộ chốt: '+str(delta))
    expected_mean=pd.read_csv(root/'reports/project_results/mean_std.csv')
    for _,r in means.iterrows():
        other=expected_mean[(expected_mean.system==r.system)&(expected_mean.split==r.split)&(expected_mean.threshold_mode==r.threshold_mode)].iloc[0]
        for k in METRICS:
            for suffix in ['_mean','_std']:
                a,b=r[k+suffix],other[k+suffix]
                if pd.isna(a) and pd.isna(b):continue
                if pd.isna(a) or pd.isna(b) or abs(a-b)>1e-12:
                    raise ValueError('Sai mean/std: '+k)
    # Đại diện chọn theo validation @0,5; không dùng test chọn seed.
    reps={}
    for arch in ['bert','roberta','distilbert']:
        q=runs[(runs.system=='C_'+arch)&(runs.split=='validation')&(runs.threshold_mode=='fixed')]
        reps[arch]=int(q.sort_values(['macro_f1','seed'],ascending=[False,True]).iloc[0].seed)
    counts=[];pairs=[];examples=[];exampleids=['eczj48j','ed0jr9i','eczcvgx']
    y=truth['test'].astype(bool)
    for arch,seed in reps.items():
        score=saved_scores[('C_'+arch,seed,'test')];p=score>=.5;fn=y&~p;fp=~y&p;tp=y&p
        masks={'partial_multi_label':(y.sum(1)>=2)&tp.any(1)&fn.any(1),'rare_false_negative':fn[:,rareidx].any(1),'missed_extra_pair':fn.any(1)&fp.any(1)}
        eligible={'partial_multi_label':int((y.sum(1)>=2).sum()),'rare_false_negative':int(y[:,rareidx].any(1).sum()),'missed_extra_pair':len(y)}
        for category,mask in masks.items():
            n=int(mask.sum());counts.append(dict(architecture=arch,seed=seed,category=category,error_samples=n,total_samples=len(y),eligible_samples=eligible[category],fraction_all_samples=n/len(y),fraction_eligible=n/eligible[category]))
        for i,label in enumerate(labels):
            for j,extra in enumerate(labels):
                mask=fn[:,i]&fp[:,j];n=int(mask.sum())
                if n:pairs.append(dict(architecture=arch,seed=seed,missed_label=label,extra_label=extra,sample_count=n,missed_label_total_fn=int(fn[:,i].sum()),fraction_of_missed_label_fn=n/int(fn[:,i].sum()),example_id=str(frames['test'].iloc[np.flatnonzero(mask)[0]].id)))
        for exampleid in exampleids:
            ix=int(np.flatnonzero(frames['test'].id.eq(exampleid).to_numpy())[0]);text=frames['test'].iloc[ix].text
            examples.append(dict(architecture=arch,seed=seed,id=exampleid,text=text,true_labels=[l for l,z in zip(labels,y[ix]) if z],predicted_labels=[l for l,z in zip(labels,p[ix]) if z],missed_labels=[l for l,z in zip(labels,fn[ix]) if z],extra_labels=[l for l,z in zip(labels,fp[ix]) if z],scores_by_label={l:float(z) for l,z in zip(labels,score[ix])}))
    pairsdf=pd.DataFrame(pairs);wide=pairsdf.pivot_table(index=['missed_label','extra_label'],columns='architecture',values='sample_count',aggfunc='sum',fill_value=0).reset_index();wide['total']=wide[['bert','roberta','distilbert']].sum(axis=1);top=wide.sort_values(['total','missed_label','extra_label'],ascending=[False,True,True]).head(5)
    cper=allper[(allper.system.str.startswith('C_'))&(allper.split=='test')].drop(columns=['system','split']);rare=cper[cper.label.isin(rarel)]
    if export:
        for name,df in [('all_runs',runs),('mean_std',means),('learning_curves',pd.DataFrame(histories)),('per_label',allper),('per_label_test',cper),('rare_labels_test',rare),('error_counts',pd.DataFrame(counts)),('error_pairs',pairsdf),('top_error_pairs',top)]:df.to_csv(output/(name+'.csv'),index=False)
        write(output/'report_examples.json',examples)
        a=allper[(allper.system=='A_standard')&(allper.split=='test')&(allper.threshold_mode=='fixed')].set_index('label');b=allper[(allper.system=='A_balanced')&(allper.split=='test')&(allper.threshold_mode=='tuned')].set_index('label')
        write(output/'a_rare.json',dict(labels=rarel,before=[float(a.loc[l,'f1']) for l in rarel],after=[float(b.loc[l,'f1']) for l in rarel]))
    if full:
        from scripts.verification.reproduce_report import verify_scores
        verify_scores(root)
    result=dict(passed=True,records=72,summary_groups=36,curve_points=len(histories),rare_label_rows=len(rare),error_groups=len(counts),representatives=reps,metric_max_absolute_difference=delta,weights_checked=full,training_repeated=False,threshold_search_repeated=False,score_and_protocol_sha256=inputs)
    write(output/'verification.json',result);print('PASS bộ nộp: 72 dòng, 36 nhóm, 30 điểm đường học; sai lệch',delta,flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts-root',type=Path);p.add_argument('--output',type=Path,default=ROOT/'results/final_submission/report_data');p.add_argument('--export',action='store_true');p.add_argument('--full',action='store_true');a=p.parse_args()
    if a.artifacts_root is None:
        from src.final_submission import prepare_runtime
        a.artifacts_root=prepare_runtime()
    verify(a.artifacts_root,a.output,a.export,a.full)
if __name__=='__main__':main()
