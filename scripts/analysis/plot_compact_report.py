"""Draw the three explanatory figures used by the compact report from frozen data."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, math, sys
from src.paths import ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=ROOT/"results/final_submission/compact_report")
W=parser.parse_args().output.resolve()
W.mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=ROOT/"results/final_submission"
EDA=ROOT/"reports/tables"
F=W/'figures'; F.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.titlesize':19,
    'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16,
    'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
    'savefig.facecolor':'white'})
manifest={'sources':{},'figures':{},'training_or_threshold_search':False}
for path in [EDA/'labels_per_sample.csv',EDA/'multilabel_summary.csv',R/'report_data/mean_std.csv',R/'report_data/all_runs.csv',R/'external_probe/predictions.json',R/'external_probe/new10.json']:
    manifest['sources'][path.relative_to(ROOT).as_posix()]=sha(path)
def save(fig,name,description,data):
    path=F/(name+'.png');fig.savefig(path,dpi=260,bbox_inches='tight',pad_inches=.14);plt.close(fig)
    manifest['figures'][name]={'path':path.name,'sha256':sha(path),'meaning':description,'plotted_data':data}

# 1: percentages use each split's own denominator. Every sample remains in a group.
card=rows(EDA/'labels_per_sample.csv');summary=rows(EDA/'multilabel_summary.csv')
groups=['1 nhãn','2 nhãn','≥3 nhãn'];splits=['train','validation','test']
splitnames=['Train','Validation','Test'];colors=['#21618C','#B35B1F','#327C64']
fig,ax=plt.subplots(figsize=(9.3,4.2));x=np.arange(3);width=.22;plotdata=[]
for offset,(split,label,color) in enumerate(zip(splits,splitnames,colors)):
    subset=[r for r in card if r['split']==split];total=sum(int(r['n_samples']) for r in subset)
    original=next(r for r in summary if r['split']==split)
    assert total==int(original['n_samples'])
    counts=[sum(int(r['n_samples']) for r in subset if int(r['n_labels'])==1),
            sum(int(r['n_samples']) for r in subset if int(r['n_labels'])==2),
            sum(int(r['n_samples']) for r in subset if int(r['n_labels'])>=3)]
    assert counts==[int(original[k]) for k in ['one_label','two_labels','three_or_more']]
    assert sum(counts)==total
    percentages=[100*n/total for n in counts]
    for row in subset:assert math.isclose(float(row['percent']),100*int(row['n_samples'])/total,abs_tol=1e-10)
    bars=ax.bar(x+(offset-1)*.27,percentages,width,color=color,label=f'{label} ({total:,})'.replace(',','.'))
    for bar,value in zip(bars,percentages):
        ax.text(bar.get_x()+bar.get_width()/2,value+1.05,f'{value:.2f}'.replace('.',','),ha='center',va='bottom',fontsize=15)
    plotdata.append({'split':split,'n_samples':total,'counts_1_2_ge3':counts,'percent_1_2_ge3':percentages,'multi_label_percent':100*sum(counts[1:])/total})
ax.set_xticks(x,groups);ax.set_ylabel('Tỷ lệ trong từng tập (%)');ax.set_ylim(0,96)
ax.set_yticks([0,20,40,60,80]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=15,bbox_to_anchor=(.53,1.025))
fig.tight_layout(rect=[0,0,1,.91]);save(fig,'label_cardinality','Distribution of true labels per sentence; 3,4,5 labels grouped as >=3, split-specific percentages.',plotdata)

# 2: two operating points, no threshold interpolation or PR-curve claim.
means=rows(R/'report_data/mean_std.csv');runs=rows(R/'report_data/all_runs.csv')
models=['bert','roberta','distilbert'];names=['C1 BERT','C2 RoBERTa','C3 DistilBERT']
fig,axes=plt.subplots(1,3,figsize=(9.3,4.4),sharex=True,sharey=True);plotdata=[]
for ax,model,name in zip(axes,models,names):
    points=[]
    for mode in ['fixed','tuned']:
        row=next(r for r in means if r['system']=='C_'+model and r['split']=='test' and r['threshold_mode']==mode)
        samples=[r for r in runs if r['system']=='C_'+model and r['split']=='test' and r['threshold_mode']==mode]
        assert len(samples)==3 and {int(float(r['seed'])) for r in samples}=={42,123,2026}
        record={'model':model,'mode':mode}
        for metric in ['micro_precision','micro_recall']:
            values=[float(r[metric]) for r in samples]
            record[metric+'_mean']=float(row[metric+'_mean']);record[metric+'_std']=float(row[metric+'_std'])
            assert math.isclose(record[metric+'_mean'],float(np.mean(values)),abs_tol=1e-12)
            assert math.isclose(record[metric+'_std'],float(np.std(values,ddof=1)),abs_tol=1e-12)
        points.append(record);plotdata.append(record)
    xvals=[r['micro_recall_mean'] for r in points];yvals=[r['micro_precision_mean'] for r in points]
    ax.annotate('',(xvals[1],yvals[1]),(xvals[0],yvals[0]),arrowprops={'arrowstyle':'->','color':'#485963','lw':2.2},zorder=2)
    for i,(row,color,marker) in enumerate(zip(points,['#7B8790','#21618C'],['o','s'])):
        xx=row['micro_recall_mean'];yy=row['micro_precision_mean']
        ax.errorbar(xx,yy,xerr=row['micro_recall_std'],yerr=row['micro_precision_std'],fmt=marker,color=color,capsize=4,markersize=8,lw=1.4,zorder=3)
        value=f"P {yy:.4f}\nR {xx:.4f}".replace('.',',')
        ax.annotate(value,(xx,yy),xytext=(8,14) if i==0 else (-6,-9),textcoords='offset points',ha='left' if i==0 else 'right',va='center' if i==0 else 'top',fontsize=14.5,color='#333')
    ax.set_title(name,fontweight='bold');ax.set_xlim(.44,.76);ax.set_ylim(.40,.85)
    ax.set_xticks([.45,.60,.75]);ax.set_yticks([.40,.50,.60,.70,.80]);ax.grid(alpha=.2)
axes[0].set_ylabel('Micro-Precision')
fig.supxlabel('Micro-Recall',fontsize=17,y=.02)
legend=[Line2D([],[],color='#7B8790',marker='o',ls='',markersize=8,label='Ngưỡng 0,5'),Line2D([],[],color='#21618C',marker='s',ls='',markersize=8,label='Ngưỡng riêng')]
fig.legend(handles=legend,loc='upper center',ncol=2,frameon=False,fontsize=16,bbox_to_anchor=(.55,1.035))
fig.tight_layout(rect=[0,.04,1,.91]);save(fig,'precision_recall','Fixed-to-tuned micro precision/recall; same axes; each point is mean of all three seeds, horizontal/vertical bars are sample std ddof=1.',plotdata)

# 3: counts are derived from the exact saved predictions, not re-evaluated models.
predictions=read(R/'external_probe/predictions.json');fixture=read(R/'external_probe/new10.json')
truth_by_id={row['id']:set(row['labels']) for row in fixture['rows']}
labels=['anger','fear','joy','sadness','surprise'];supports=[sum(label in truth for truth in truth_by_id.values()) for label in labels]
assert supports==[1,4,1,4,1]
fig,axes=plt.subplots(1,3,figsize=(9.3,3.65),sharey=True);plotdata=[]
for ax,model,name,seed in zip(axes,models,names,[123,2026,123]):
    selected=[r for r in predictions if r['model']==model and r['seed']==seed and r['mode']=='tuned']
    assert len(selected)==10 and {r['id'] for r in selected}==set(truth_by_id)
    for row in selected:
        truth=truth_by_id[row['id']];prediction=set(row['mapped5'])
        assert set(row['truth'])==truth and set(row['fn'])==truth-prediction and set(row['fp'])==prediction-truth
    arr=np.array([[sum(label in r[key] for r in selected) for key in ['fn','fp']] for label in labels])
    ax.imshow(arr,vmin=0,vmax=4,cmap='Blues',aspect='auto')
    ax.set_xticks([0,1],['FN','FP']);ax.set_yticks(range(5),[f'{label} ({n})' for label,n in zip(labels,supports)])
    ax.set_title(name+'\nseed '+str(seed),fontsize=18,fontweight='bold',pad=10)
    ax.tick_params(axis='y',length=0)
    for i in range(5):
        for j in range(2):ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>=3 else '#111',fontsize=21,fontweight='bold')
    plotdata.append({'model':model,'seed':seed,'mode':'tuned','labels':labels,'true_supports':supports,'fn_fp_counts':arr.tolist()})
fig.tight_layout(w_pad=1.5);save(fig,'external_errors','True-label support in parentheses; each cell is FN/FP count over ten saved sentences for one representative checkpoint; same color scale 0..4.',plotdata)
(W/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Three figures generated from frozen measurements. All data checks passed; no training or threshold changes.')
