"""Rebuild retained report charts from frozen CSVs; never fit models/thresholds."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.paths import ROOT
from scripts.analysis.report_style import vi,SPLIT_COLORS,MODEL_COLORS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();f=out/'figures';t=out/'tables'
 if out.is_relative_to((ROOT/'results/final_submission').resolve()):p.error('Output must be outside frozen results')
 f.mkdir(parents=True,exist_ok=True);t.mkdir(parents=True,exist_ok=True)
 d=ROOT/'results/final_submission/report_data';eda=ROOT/'reports/tables'
 means=pd.read_csv(d/'mean_std.csv');runs=pd.read_csv(d/'all_runs.csv');per=pd.read_csv(d/'per_label_test.csv');curves=pd.read_csv(d/'learning_curves.csv')
 models=['bert','roberta','distilbert'];names=['C1 BERT','C2 RoBERTa','C3 DistilBERT']
 representatives={m:int(runs[(runs.system=='C_'+m)&(runs.split=='validation')&(runs.threshold_mode=='fixed')].sort_values(['macro_f1','seed'],ascending=[False,True]).iloc[0].seed) for m in models}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':15,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
 manifest={'training':False,'threshold_search':False,'figures':{},'source_sha256':{}}
 for path in list(d.glob('*.csv'))+list(d.glob('*diagnostic.json'))+[eda/'label_distribution.csv',eda/'labels_per_sample.csv']:
  manifest['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 fmt=FuncFormatter(lambda x,_:f'{x:g}'.replace('.',','))
 def save(fig,name):
  fig.savefig(f/(name+'.png'),dpi=260,bbox_inches='tight',pad_inches=.12);plt.close(fig)
  manifest['figures'][name]=hashlib.sha256((f/(name+'.png')).read_bytes()).hexdigest()
 def group(arch,mode):return means[(means.system=='C_'+arch)&(means.split=='test')&(means.threshold_mode==mode)].iloc[0]
 # Rebuild explanatory diagrams so their threshold terminology matches the report.
 from matplotlib.patches import FancyBboxPatch
 def box(ax,x,y,w,h,label):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',facecolor='#F4F7FA',edgecolor='#516579'))
  ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=15,linespacing=1.4)
 fig,ax=plt.subplots(figsize=(11,4.2));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4.2))
 box(ax,.1,1.4,1.8,1.1,'Bình luận\ntiếng Anh')
 for y,label in [(3.1,'A: TF-IDF + OvR LR'),(1.65,'B: BART-MNLI'),(.2,'C: BERT / RoBERTa / DistilBERT')]:
  box(ax,2.4,y,4.4,.9,label);ax.annotate('',(2.35,y+.45),(1.95,1.95),arrowprops={'arrowstyle':'->','color':'#516579'})
  ax.annotate('',(7.3,1.95),(6.85,y+.45),arrowprops={'arrowstyle':'->','color':'#516579'})
 box(ax,7.35,1.25,3.5,1.4,'28 điểm → áp ngưỡng\nNgưỡng 0,5 / Ngưỡng riêng\n→ tập nhãn')
 fig.tight_layout();save(fig,'01_hinh_1_1')
 fig,ax=plt.subplots(figsize=(11,5.4));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,5.4))
 for y,name,tok,enc,head in [(3.8,'C1 BERT','WordPiece\ncased','12 block\nhidden 768','[CLS] → pooler Dense + tanh\nDropout → Linear 28'),(2.15,'C2 RoBERTa','Byte-level\nBPE','12 block\nhidden 768','<s> → Dropout → Dense + tanh\nDropout → Linear 28'),(.5,'C3 DistilBERT','WordPiece\nuncased','6 block\nhidden 768','Token đầu → Dense + ReLU\nDropout → Linear 28')]:
  ax.text(.1,y+1.1,name,weight='bold',fontsize=14)
  for x,w,label in [(.1,1.6,tok),(2.1,1.8,enc),(4.3,3.8,head),(8.5,2.3,'28 logits → sigmoid\nNgưỡng 0,5 /\nNgưỡng riêng')]:box(ax,x,y,w,.9,label)
  for x in [1.75,3.95,8.15]:ax.annotate('',(x+.3,y+.45),(x,y+.45),arrowprops={'arrowstyle':'->','color':'#516579'})
 fig.tight_layout();save(fig,'02_hinh_2_1')
 q=pd.read_csv(eda/'label_distribution.csv').sort_values('train_count',ascending=True)
 fig,ax=plt.subplots(figsize=(9.2,9.5));bars=ax.barh(q.label,q.train_count,color='#167D9A')
 ax.bar_label(bars,labels=[f'{x:,}'.replace(',','.') for x in q.train_count],padding=4,fontsize=13)
 ax.tick_params(labelsize=13);ax.set_xlim(0,q.train_count.max()*1.16);ax.set_xlabel('Số mẫu dương trên tập train');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'label_distribution')
 card=pd.read_csv(eda/'labels_per_sample.csv');fig,ax=plt.subplots(figsize=(9.2,3.7));x=np.arange(3)
 for i,(split,color) in enumerate(zip(['train','validation','test'],SPLIT_COLORS)):
  q=card[card.split==split];n=q.n_samples.sum();vals=[q[q.n_labels==1].n_samples.sum(),q[q.n_labels==2].n_samples.sum(),q[q.n_labels>=3].n_samples.sum()];vals=np.array(vals)*100/n
  bars=ax.bar(x+(i-1)*.25,vals,.22,label=split.capitalize(),color=color);ax.bar_label(bars,labels=[vi(v,2) for v in vals],padding=3,fontsize=11)
 ax.set_xticks(x,['1 nhãn','2 nhãn','≥3 nhãn']);ax.set_ylim(0,98);ax.set_ylabel('Tỷ lệ trong từng tập (%)');ax.legend(ncol=3,frameon=False);fig.tight_layout();save(fig,'label_cardinality')
 fig,axes=plt.subplots(2,3,figsize=(11,5.6),sharex='col')
 for j,arch in enumerate(models):
  for seed,color in zip([42,123,2026],['#6E44A5','#C18B00','#277DA8']):
   q=curves[(curves.architecture==arch)&(curves.seed==seed)].sort_values('epoch')
   for r,key in [(0,'train_loss'),(1,'validation_macro_f1')]:axes[r,j].plot(q.epoch,q[key],'-o',color=color,label=f'Seed {seed}',markersize=4)
  axes[0,j].set_title(names[j]);axes[1,j].set_xlabel('Epoch')
  for r in [0,1]:axes[r,j].grid(alpha=.15);axes[r,j].set_xticks(range(1,5 if arch=='bert' else 4));axes[r,j].yaxis.set_major_formatter(fmt)
 axes[0,0].set_ylabel('Loss train');axes[1,0].set_ylabel('Macro-F1 validation');fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=3,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.93]);save(fig,'seed_curves')
 diag=[json.loads((d/(m+'_diagnostic.json')).read_text(encoding='utf-8')) for m in models]
 fig,axes=plt.subplots(2,1,figsize=(9.2,6.0));x=np.arange(3);bw=.30
 for ax,key,title,lim in zip(axes,['macro_f1','bce'],['Macro-F1 tại ngưỡng 0,5','BCE không trọng số'],[.86,.12]):
  for j,(split,color) in enumerate(zip(['train','validation'],SPLIT_COLORS)):
   vals=[next(r[key] for r in z['rows'] if r['split']==split) for z in diag]
   bars=ax.bar(x+(j-.5)*bw,vals,bw,color=color,label=split.capitalize());ax.bar_label(bars,labels=[vi(v) for v in vals],padding=4,fontsize=12)
  ax.set_title(title);ax.set_ylim(0,lim);ax.set_xticks(x,[f'C{i+1} / {representatives[m]}' for i,m in enumerate(models)]);ax.yaxis.set_major_formatter(fmt);ax.grid(axis='y',alpha=.15)
 axes[0].legend(ncol=2,frameon=False);fig.tight_layout();save(fig,'train_validation')
 fig,axes=plt.subplots(1,3,figsize=(10.8,4.0),sharex=True,sharey=True)
 for ax,arch,name in zip(axes,models,names):
  rows=[group(arch,m) for m in ['fixed','tuned']];xx=[r.micro_recall_mean for r in rows];yy=[r.micro_precision_mean for r in rows]
  ax.annotate('',(xx[1],yy[1]),(xx[0],yy[0]),arrowprops={'arrowstyle':'->','color':'#444','lw':2})
  for i,(r,c,mk) in enumerate(zip(rows,['#777','#167D9A'],['o','s'])):
   ax.errorbar(xx[i],yy[i],xerr=r.micro_recall_std,yerr=r.micro_precision_std,fmt=mk,color=c,capsize=3,label=['Ngưỡng 0,5','Ngưỡng riêng'][i])
   ax.annotate(f'P {vi(yy[i])}\nR {vi(xx[i])}',(xx[i],yy[i]),xytext=(5,12) if i==0 else (-5,-12),textcoords='offset points',ha='left' if i==0 else 'right',va='bottom' if i==0 else 'top',fontsize=10.5)
  ax.set_title(name);ax.set_xlim(.43,.77);ax.set_ylim(.40,.86);ax.set_xlabel('Micro-Recall');ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt);ax.grid(alpha=.15)
 axes[0].set_ylabel('Micro-Precision');fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.89]);save(fig,'precision_recall')
 rare=['grief','pride','relief','nervousness','embarrassment'];rows=[];fig,axes=plt.subplots(2,2,figsize=(10.3,8.0),sharex=True)
 ar=json.loads((d/'a_rare.json').read_text(encoding='utf-8'))
 for ax,arch,name in zip(axes.ravel(),['A']+models,['A thường 0,5 → A cân bằng ngưỡng riêng']+names):
  before=[];after=[]
  for i,label in enumerate(rare):
   if arch=='A':v0=ar['before'][i];v1=ar['after'][i];sd0=sd1=0
   else:
    q=per[(per.architecture==arch)&(per.label==label)];a0=q[q.threshold_mode=='fixed'].f1;a1=q[q.threshold_mode=='tuned'].f1;assert len(a0)==len(a1)==3;v0=a0.mean();v1=a1.mean();sd0=a0.std(ddof=1);sd1=a1.std(ddof=1)
   ax.plot([v0,v1],[i,i],color='#AAA');ax.errorbar(v0,i-.09,xerr=sd0,fmt='o',color='#777',capsize=2);ax.errorbar(v1,i+.09,xerr=sd1,fmt='s',color='#167D9A',capsize=2)
   rows.append(dict(model=arch,label=label,before=v0,after=v1,std_before=sd0,std_after=sd1))
  ax.set_yticks(range(5),[f'{l} (n={n})' for l,n in zip(rare,[6,16,11,23,37])]);ax.invert_yaxis();ax.set_xlim(-.1,.72);ax.set_title(name,fontsize=13.5);ax.tick_params(labelsize=13.5);ax.set_xlabel('F1 test');ax.xaxis.set_major_formatter(fmt);ax.grid(axis='x',alpha=.15)
 fig.tight_layout();save(fig,'rare_labels');pd.DataFrame(rows).to_csv(t/'rare_f1_before_after.csv',index=False)
 err=pd.read_csv(d/'error_counts.csv');err.to_csv(t/'error_counts_verified.csv',index=False);fig,axes=plt.subplots(3,1,figsize=(9.5,5.3))
 for ax,cat,title in zip(axes,['partial_multi_label','rare_false_negative','missed_extra_pair'],['Câu đa nhãn đúng một phần, còn bỏ sót','Bỏ sót ít nhất một nhãn hiếm','Đồng thời có FN và FP']):
  for i,(arch,color) in enumerate(zip(models,MODEL_COLORS)):
   r=err[(err.architecture==arch)&(err.category==cat)].iloc[0];v=100*r.fraction_eligible
   ax.barh(i,v,color=color);ax.text(v+1,i,f'{int(r.error_samples)}/{int(r.eligible_samples)} = {vi(v,2)}%',va='center',fontsize=11)
  ax.set_yticks(range(3),names);ax.invert_yaxis();ax.set_xlim(0,127);ax.set_xticks([0,25,50,75,100]);ax.set_title(title,fontsize=13);ax.grid(axis='x',alpha=.15)
 axes[-1].set_xlabel('Tỷ lệ trong nhóm đủ điều kiện (%)');fig.tight_layout();save(fig,'error_groups')
 top=pd.read_csv(d/'top_error_pairs.csv');fig,ax=plt.subplots(figsize=(9.5,4.3));x=np.arange(5)
 for i,(arch,name,color) in enumerate(zip(models,names,MODEL_COLORS)):
  bars=ax.barh(x+(i-1)*.22,top[arch],.20,label=name,color=color);ax.bar_label(bars,padding=3,fontsize=11)
 ax.set_yticks(x,[f'{a} → {b}' for a,b in zip(top.missed_label,top.extra_label)]);ax.invert_yaxis();ax.set_xlim(0,90);ax.set_xlabel('Số câu có cặp FN → FP');fig.legend(*ax.get_legend_handles_labels(),ncol=3,frameon=False,loc='upper center');fig.tight_layout(rect=[0,0,1,.89]);save(fig,'error_pairs')
 fig,axes=plt.subplots(1,2,figsize=(9.2,3.6));x=np.arange(3)
 for ax,metric,title in zip(axes,['macro_f1','micro_f1'],['Macro-F1 test','Micro-F1 test']):
  for i,(mode,c,name) in enumerate(zip(['fixed','tuned'],['#888','#167D9A'],['Ngưỡng 0,5','Ngưỡng riêng'])):
   vals=[group(m,mode)[metric+'_mean'] for m in models];sd=[group(m,mode)[metric+'_std'] for m in models];ax.bar(x+(i-.5)*.3,vals,.28,yerr=sd,capsize=3,label=name,color=c)
  ax.set_xticks(x,['C1','C2','C3']);ax.set_ylim(0,.72);ax.set_title(title);ax.yaxis.set_major_formatter(fmt);ax.grid(axis='y',alpha=.15)
 fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False);fig.tight_layout(rect=[0,0,1,.88]);save(fig,'test_f1_summary')
 pairs=pd.read_csv(d/'error_pairs.csv');labels=json.loads((ROOT/'data/labels.json').read_text(encoding='utf-8'));mats=[]
 for arch in models:
  mat=np.zeros((28,28),int)
  for row in pairs[pairs.architecture==arch].itertuples():mat[labels.index(row.missed_label),labels.index(row.extra_label)]=row.sample_count
  mats.append(mat)
 pooled=sum(mats);importance=pooled.sum(0)+pooled.sum(1);chosen=sorted(range(28),key=lambda i:(-importance[i],i))[:10]
 vmax=max(int(m.max()) for m in mats)
 for arch,name,mat in zip(models,names,mats):
  seed=representatives[arch]
  q=mat[np.ix_(chosen,chosen)];fig,ax=plt.subplots(figsize=(8.7,7.3));im=ax.imshow(q,cmap='Blues',vmin=0,vmax=vmax);ax.set_xticks(range(10),[labels[i] for i in chosen],rotation=45,ha='right');ax.set_yticks(range(10),[labels[i] for i in chosen]);ax.set_xlabel('Nhãn dự đoán thừa (FP)');ax.set_ylabel('Nhãn thật bị bỏ sót (FN)');ax.set_title(f'{name} / seed {seed} / ngưỡng 0,5')
  for i in range(10):
   for j in range(10):ax.text(j,i,str(q[i,j]),ha='center',va='center',color='white' if q[i,j]>.55*vmax else '#222',fontsize=13)
  fig.colorbar(im,ax=ax,shrink=.75,label='Số câu');fig.tight_layout();save(fig,'coerror_'+arch)
 vals=np.zeros((28,6));tab=[]
 for j,arch in enumerate(models):
  for k,mode in enumerate(['fixed','tuned']):
   q=per[(per.architecture==arch)&(per.threshold_mode==mode)];g=q.groupby('label').f1.agg(['mean','std']);assert len(q)==84
   for i,label in enumerate(labels):vals[i,2*j+k]=g.loc[label,'mean'];tab.append(dict(architecture=arch,mode=mode,label=label,f1_mean=g.loc[label,'mean'],f1_std=g.loc[label,'std']))
   assert abs(vals[:,2*j+k].mean()-group(arch,mode).macro_f1_mean)<1e-12
 pd.DataFrame(tab).to_csv(t/'f1_per_label_mean_std.csv',index=False)
 fig,ax=plt.subplots(figsize=(8.8,11.5));im=ax.imshow(vals,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(28),labels);ax.set_xticks(range(6),['C1\n0,5','C1\nNgưỡng\nriêng','C2\n0,5','C2\nNgưỡng\nriêng','C3\n0,5','C3\nNgưỡng\nriêng'],fontsize=12);ax.set_title('F1 test từng nhãn — trung bình ba seed')
 for i in range(28):
  for j in range(6):ax.text(j,i,vi(vals[i,j],3),ha='center',va='center',color='white' if vals[i,j]>.58 else '#222',fontsize=12)
 cb=fig.colorbar(im,ax=ax,shrink=.6,label='F1');cb.ax.yaxis.set_major_formatter(fmt);fig.tight_layout();save(fig,'f1_all_labels')
 probe_path=ROOT/'results/final_submission/external_probe/summary.json'
 probe=json.loads(probe_path.read_text(encoding='utf-8'))
 manifest['source_sha256'][probe_path.relative_to(ROOT).as_posix()]=hashlib.sha256(probe_path.read_bytes()).hexdigest()
 selected=[next(r for r in probe if r['model']==arch and r['seed']==representatives[arch] and r['mode']=='fixed') for arch in models]
 counts=[np.array([[r['fn'],r['fp']] for r in row['per_label']]) for row in selected]
 fig,axes=plt.subplots(1,3,figsize=(10.4,3.6),sharey=True)
 limit=max(1,max(int(m.max()) for m in counts))
 for ax,arch,name,mat,row in zip(axes,models,names,counts,selected):
  ax.imshow(mat,cmap='Blues',vmin=0,vmax=limit,aspect='auto')
  ax.set_xticks([0,1],['FN','FP']);ax.set_yticks(range(5),[f"{r['label']} ({r['support']})" for r in row['per_label']])
  ax.set_title(f'{name}\nSeed {representatives[arch]}',fontsize=15)
  for i in range(5):
   for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=18,color='white' if mat[i,j]>.55*limit else '#222')
 fig.tight_layout();save(fig,'brighter_errors')
 (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Charts:',len(manifest['figures']))
if __name__=='__main__':main()
