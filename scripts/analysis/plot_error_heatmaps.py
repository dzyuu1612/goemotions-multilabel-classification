"""Recreate GoEmotions charts inspired by the XED report from saved evidence.

No training, new inference, threshold fitting or benchmark file is modified.
FN/FP co-errors are not a single-label confusion matrix. A sentence may count
in several cells and a missed label with no extra prediction counts in none.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.paths import ROOT


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo-scores',type=Path)
    a=p.parse_args();out=a.output.resolve();data=ROOT/'results/final_submission/report_data'
    if out.is_relative_to((ROOT/'results/final_submission').resolve()):
        p.error('Use a new output directory outside frozen results')
    out.mkdir(parents=True,exist_ok=True)
    labels=json.loads((ROOT/'data/labels.json').read_text(encoding='utf-8'))
    pairs=pd.read_csv(data/'error_pairs.csv');per=pd.read_csv(data/'per_label_test.csv')
    means=pd.read_csv(data/'mean_std.csv')
    architectures=['bert','roberta','distilbert'];names=['C1 BERT','C2 RoBERTa','C3 DistilBERT']
    seeds=[123,2026,123]
    matrices=[];all_rows=[]
    for arch,seed in zip(architectures,seeds):
        q=pairs[pairs.architecture==arch]
        assert set(q.seed)=={seed} and not q.duplicated(['missed_label','extra_label']).any()
        matrix=np.zeros((28,28),dtype=int)
        for row in q.itertuples():
            assert row.sample_count>=0 and int(row.sample_count)==row.sample_count
            matrix[labels.index(row.missed_label),labels.index(row.extra_label)]=row.sample_count
        assert not np.diag(matrix).any()
        matrices.append(matrix)
        for i,missed in enumerate(labels):
            for j,extra in enumerate(labels):
                all_rows.append(dict(architecture=arch,seed=seed,missed_label=missed,extra_label=extra,count=int(matrix[i,j])))
    pd.DataFrame(all_rows).to_csv(out/'error_pairs_28x28.csv',index=False,encoding='utf-8-sig')
    pooled=sum(matrices);importance=pooled.sum(0)+pooled.sum(1)
    chosen=sorted(range(28),key=lambda i:(-importance[i],i))[:10]
    shown_labels=[labels[i] for i in chosen]
    small=[m[np.ix_(chosen,chosen)] for m in matrices];vmax=max(int(m.max()) for m in small)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.unicode_minus':False,
                         'savefig.facecolor':'white','axes.titleweight':'bold'})
    manifest={'schema_version':1,'threshold_mode_error_pairs':'fixed_0.5','split':'test','n_samples':5427,
              'representatives':dict(zip(architectures,seeds)),
              'label_selection':'10 labels with largest pooled row+column FN/FP co-error counts across three validation-selected representatives; descriptive only',
              'shown_labels':shown_labels,'all_labels':labels,'common_count_vmax':vmax,'charts':[],
              'input_sha256':{str(x.relative_to(ROOT)):digest(x) for x in [data/'error_pairs.csv',data/'per_label_test.csv',data/'mean_std.csv',ROOT/'data/labels.json']},
              'matrices':[{'architecture':arch,'seed':seed,'full_matrix':m.tolist(),'shown_matrix':s.tolist(),
                            'all_pair_events':int(m.sum()),'shown_pair_events':int(s.sum())}
                          for arch,seed,m,s in zip(architectures,seeds,matrices,small)]}
    def save(fig,name,description):
        path=out/(name+'.png');fig.savefig(path,dpi=240,bbox_inches='tight',pad_inches=.15);plt.close(fig)
        manifest['charts'].append({'file':path.name,'sha256':digest(path),'description':description})
    for arch,name,seed,m in zip(architectures,names,seeds,small):
        fig,ax=plt.subplots(figsize=(9.7,8.1))
        im=ax.imshow(m,cmap='Reds',vmin=0,vmax=vmax)
        ax.set_xticks(range(10),shown_labels,rotation=45,ha='right');ax.set_yticks(range(10),shown_labels)
        ax.set_xlabel('Nhãn dự đoán thừa (FP)');ax.set_ylabel('Nhãn thật bị bỏ sót (FN)')
        ax.set_title(f'{name} / seed {seed}\nĐồng xuất hiện FN và FP trên cùng câu',pad=13,fontsize=16)
        for i in range(10):
            for j in range(10):ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if m[i,j]>vmax*.55 else '#202B33',fontsize=12)
        fig.colorbar(im,ax=ax,shrink=.82,label='Số câu có cặp FN–FP')
        fig.tight_layout();save(fig,f'coerror_{arch}','10/28 labels, same labels/order and color scale across C1/C2/C3, threshold 0.5, one representative per architecture')
    # Per-label means keep full seed sets and both threshold policies.
    values=np.zeros((28,6));summary=[]
    for j,arch in enumerate(architectures):
        for k,mode in enumerate(['fixed','tuned']):
            q=per[(per.architecture==arch)&(per.threshold_mode==mode)]
            assert set(q.seed)=={42,123,2026} and len(q)==84
            grouped=q.groupby('label').f1.agg(['mean','std'])
            for i,label in enumerate(labels):
                values[i,2*j+k]=grouped.loc[label,'mean']
                summary.append(dict(architecture=arch,mode=mode,label=label,f1_mean=float(grouped.loc[label,'mean']),f1_std=float(grouped.loc[label,'std']),n_seeds=3))
            expected=means[(means.system=='C_'+arch)&(means.split=='test')&(means.threshold_mode==mode)].iloc[0].macro_f1_mean
            assert abs(float(values[:,2*j+k].mean())-expected)<1e-12
    pd.DataFrame(summary).to_csv(out/'f1_per_label_mean_std.csv',index=False,encoding='utf-8-sig')
    manifest['f1_values']=values.tolist();manifest['f1_columns']=['C1 0,5','C1 riêng','C2 0,5','C2 riêng','C3 0,5','C3 riêng']
    fig,ax=plt.subplots(figsize=(9.5,11.0));im=ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(6),manifest['f1_columns']);ax.set_yticks(range(28),labels);ax.tick_params(axis='y',labelsize=12)
    ax.set_title('F1 từng nhãn trên test\nTrung bình đủ ba seed của mỗi kiến trúc',fontsize=17,pad=15)
    for i in range(28):
        for j in range(6):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=11,color='white' if values[i,j]>.58 else '#182B3A')
    fig.colorbar(im,ax=ax,shrink=.65,label='F1 (0 đến 1)');fig.tight_layout()
    save(fig,'f1_all_labels','28 labels; test mean over seeds 42/123/2026; fixed versus frozen per-label validation thresholds; standard deviations in CSV')
    if a.demo_scores:
        demo=json.loads(a.demo_scores.read_text(encoding='utf-8'))
        assert demo['no_gold_labels'] is True
        case=next(r for r in demo['predictions'] if r['architecture']=='bert' and r['case']==2 and r['mode']=='tuned')
        rows=sorted(case['rows'],key=lambda r:-r['score'])
        pd.DataFrame(rows).to_csv(out/'demo_case2_all28.csv',index=False,encoding='utf-8-sig')
        top=rows[:12]
        fig,ax=plt.subplots(figsize=(10.5,6.4));ys=np.arange(12)
        ax.barh(ys,[r['score'] for r in top],height=.62,color='#21618C',label='Điểm sigmoid')
        ax.scatter([r['threshold'] for r in top],ys,color='#C3621C',marker='D',s=68,zorder=3,label='Ngưỡng riêng từ validation')
        for i,r in enumerate(top):ax.text(.985,i,f"{r['score']:.4f}",va='center',ha='right',fontsize=11)
        ax.set_yticks(ys,[r['label']+(' *' if r['selected'] else '') for r in top]);ax.invert_yaxis()
        ax.set_xlim(0,1);ax.set_xlabel('Điểm và ngưỡng trên cùng thang 0–1');ax.grid(axis='x',alpha=.2)
        ax.set_title('Câu demo mới / BERT seed 123\n12 nhãn có điểm cao nhất; dấu * là nhãn đạt ngưỡng',fontsize=16,pad=13)
        ax.legend(loc='lower right',frameon=False);fig.tight_layout()
        save(fig,'demo_scores_thresholds','Authored case 2, GPU float32 saved inference; no gold labels or accuracy claim; top 12/28 scores; full scores in CSV')
        manifest['demo_case']=case;manifest['demo_top12']=top;manifest['demo_scores_sha256']=digest(a.demo_scores)
    (out/'chart_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'charts':len(manifest['charts']),'common_count_vmax':vmax,'shown_labels':shown_labels},ensure_ascii=True))


if __name__=='__main__':main()
