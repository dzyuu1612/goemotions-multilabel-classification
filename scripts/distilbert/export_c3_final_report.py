"""Export the report dataset using archived A/B/C1/C2 and Huy's evaluated C3."""
import io
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.paths import ROOT
from src.datasets.goemotions import load_goemotions,multi_hot,sha256
from src.models.baseline import load_aligned_scores
from src.models.distilbert_study import read_json,write_json
from src.evaluation.metrics import evaluate_multilabel
from scripts.analysis.analyze_project_errors import error_masks

ARCHIVE='3c44309'
EVAL=ROOT/'data/processed/c3_distilbert/test_evaluation_20261021'
OUT=ROOT/'reports/c3_huy_final_20261021'
KEYS=['macro_f1','micro_f1','macro_precision','macro_recall','micro_precision','micro_recall','hamming_loss']

def archived(path):
    raw=subprocess.check_output(['git','show',f'{ARCHIVE}:{path}'],cwd=ROOT)
    dest=OUT/'archived_sources'/Path(path).name
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=read_json(EVAL/'final_protocol.json');fingerprint=sha256(EVAL/'final_protocol.json')
    validation=read_json(EVAL/'validation_results.json')
    frames,names,manifest=load_goemotions(ROOT,write_metadata=False)
    truth=multi_hot(frames['test'].labels.tolist());train=multi_hot(frames['train'].labels.tolist()).sum(0)
    rare_ids=sorted(range(28),key=lambda j:(int(train[j]),j))[:5]
    old_runs=archived('reports/project_results/all_runs.csv')
    runs=old_runs[old_runs.system!='C_distilbert'].to_dict('records')
    tests={};c3_per_label=[]
    for seed in protocol['seeds']:
        result=read_json(EVAL/f'seed_{seed}/test_results.json')
        assert result['protocol_sha256']==fingerprint
        assert result['scores_sha256']==sha256(EVAL/f'seed_{seed}/test_scores.npz')
        scores=load_aligned_scores(EVAL/f'seed_{seed}/test_scores.npz',frames['test'].id,names)
        limits=next(r['thresholds'] for r in protocol['runs'] if r['seed']==seed)
        for mode in ('fixed','global','tuned'):
            recalculated=evaluate_multilabel(truth,scores,names,limits[mode])
            assert recalculated==result['results'][mode]
        tests[seed]=result['results']
        for split,metrics in [('validation',validation[str(seed)]),('test',tests[seed])]:
            for mode,m in metrics.items():
                runs.append({'system':'C_distilbert','seed':seed,'split':split,'threshold_mode':mode,
                             **{k:m[k] for k in KEYS},'n_samples':m['n_samples']})
                for row in m['per_label']:
                    c3_per_label.append({'seed':seed,'split':split,'threshold_mode':mode,**row})
    df=pd.DataFrame(runs);assert len(df)==72
    df.to_csv(OUT/'all_runs.csv',index=False)
    summary=[]
    for (system,split,mode),part in df.groupby(['system','split','threshold_mode'],sort=False):
        summary.append({'system':system,'split':split,'threshold_mode':mode,'n_runs':len(part),
                        **{f'{k}_{stat}':float(part[k].mean() if stat=='mean' else part[k].std(ddof=1)) for k in KEYS for stat in ('mean','std')}})
    pd.DataFrame(summary).to_csv(OUT/'mean_std.csv',index=False)
    pd.DataFrame(c3_per_label).to_csv(OUT/'c3_per_label.csv',index=False)
    rare=[]
    for split in ('validation','test'):
        for j in rare_ids:
            before=[];after=[]
            for seed in protocol['seeds']:
                data=validation[str(seed)] if split=='validation' else tests[seed]
                before.append(data['fixed']['per_label'][j]['f1']);after.append(data['tuned']['per_label'][j]['f1'])
            rare.append({'label':names[j],'split':split,'train_support':int(train[j]),
                         'before_mean':float(np.mean(before)),'before_std':float(np.std(before,ddof=1)),
                         'after_mean':float(np.mean(after)),'after_std':float(np.std(after,ddof=1)),
                         'delta_mean':float(np.mean(np.array(after)-np.array(before)))})
    pd.DataFrame(rare).to_csv(OUT/'c3_rare_before_after.csv',index=False)
    representative=protocol['representative_seed']
    scores=load_aligned_scores(EVAL/f'seed_{representative}/test_scores.npz',frames['test'].id,names)
    masks,pred,fn,fp=error_masks(truth,scores,rare_ids,.5)
    counts={k:int(v.sum()) for k,v in masks.items()}
    old_counts=archived('reports/errors_test_standard_fixed/counts.csv')
    for k,n in counts.items():old_counts.loc[(old_counts.architecture=='distilbert')&(old_counts.category==k),'error_samples']=n
    c3mask=old_counts.architecture=='distilbert'
    old_counts.loc[c3mask,'seed']=representative
    old_counts.loc[c3mask,'fraction_all_samples']=old_counts.loc[c3mask,'error_samples']/5427
    old_counts.loc[c3mask,'fraction_eligible']=old_counts.loc[c3mask,'error_samples']/old_counts.loc[c3mask,'eligible_samples']
    old_counts.to_csv(OUT/'error_counts.csv',index=False)
    old_pairs=archived('reports/errors_test_standard_fixed/pairs.csv')
    pair_counts=fn.astype(np.int64).T@fp.astype(np.int64)
    rows=old_pairs[old_pairs.architecture!='distilbert'][['architecture','missed_label','extra_label','sample_count']].to_dict('records')
    for a,b in np.argwhere(pair_counts>0):rows.append({'architecture':'distilbert','missed_label':names[a],'extra_label':names[b],'sample_count':int(pair_counts[a,b])})
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'error_pairs.csv',index=False)
    pivot=pairs.pivot_table(index=['missed_label','extra_label'],columns='architecture',values='sample_count',aggfunc='sum',fill_value=0).reset_index()
    pivot['total']=pivot[['bert','roberta','distilbert']].sum(axis=1)
    top=pivot.sort_values(['total','missed_label','extra_label'],ascending=[False,True,True]).head(5)
    top.to_csv(OUT/'top_error_pairs.csv',index=False)
    old_examples=archived('reports/errors_test_standard_fixed/examples.csv')
    examples=[]
    for ident in old_examples.id.unique():
        i=int(np.flatnonzero(frames['test'].id.to_numpy()==ident)[0])
        examples.append({'id':ident,'text':frames['test'].iloc[i].text,
                         'true_labels':[names[j] for j in np.flatnonzero(truth[i])],
                         'predicted_labels':[names[j] for j in np.flatnonzero(pred[i])],
                         'missed_labels':[names[j] for j in np.flatnonzero(fn[i])],
                         'extra_labels':[names[j] for j in np.flatnonzero(fp[i])],
                         'scores':{name:float(scores[i,j]) for j,name in enumerate(names)},
                         'error_groups':{k:bool(mask[i]) for k,mask in masks.items()}})
    curves=archived('reports/project_results/learning_curves.csv')
    curves=curves[curves.architecture!='distilbert'].to_dict('records')
    for seed in protocol['seeds']:
        for h in read_json(ROOT/f'data/processed/c3_distilbert/full/seed_{seed}/history.json'):
            curves.append({'architecture':'distilbert','seed':seed,'status':'ok','reason':'','selected_epoch':3,**h})
    curves=pd.DataFrame(curves);curves.to_csv(OUT/'learning_curves.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    for arch,label,color in [('bert','BERT','#315b81'),('roberta','RoBERTa','#607080'),('distilbert','DistilBERT (Huy)','#b16d35')]:
        g=curves[curves.architecture==arch].groupby('epoch').validation_macro_f1.agg(['mean','std'])
        ax.plot(g.index,g['mean'],marker='o',label=label,color=color);ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.13,color=color)
    ax.set(xlabel='Epoch',ylabel='Validation Macro-F1 @ 0.5',xticks=[1,2,3,4]);ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'learning_curves_validation.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['macro_f1','micro_f1']):
        for offset,mode,label,color in [(-.18,'fixed','Fixed @ 0.5','#377eb8'),(.18,'tuned','Per-label (validation)','#ff8b22')]:
            values=[next(r for r in summary if r['system']=='C_'+arch and r['split']=='test' and r['threshold_mode']==mode) for arch in ['bert','roberta','distilbert']]
            ax.bar(np.arange(3)+offset,[r[key+'_mean'] for r in values],.36,yerr=[r[key+'_std'] for r in values],capsize=3,label=label,color=color)
        ax.set(title=key,xticks=np.arange(3),xticklabels=['BERT','RoBERTa','DistilBERT\n(Huy)'],ylabel='Mean ± sample std');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Test: checkpoints and thresholds frozen on validation');fig.tight_layout();fig.savefig(OUT/'fixed_tuned_test.png',dpi=220);plt.close(fig)
    result={'provenance':{'archive_commit':ARCHIVE,'unchanged_systems':['A','B','C1','C2'],
                          'C3':'Huy original full runs; test evaluation 2026-10-21','protocol_sha256':fingerprint},
            'representative_seed':representative,'c3_counts':counts,'summary':summary,'runs':runs,'rare':rare,
            'top_pairs':top.to_dict('records'),'examples':examples}
    def finite_json(value):
        if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list):return [finite_json(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    write_json(OUT/'report_data.json',finite_json(result))
    print(json.dumps({'C3_test_summary':[r for r in summary if r['system']=='C_distilbert'],
                      'counts':counts,'examples':[e for e in examples if e['id'] in ['eczj48j','ed0jr9i','eczcvgx']]},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
