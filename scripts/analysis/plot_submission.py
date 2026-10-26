from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W=Path(__file__).resolve().parents[2]/'results/final_submission'; D=W/'report_data'; F=W/'figures'; F.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':17,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','axes.unicode_minus':False})
colors=['#21618C','#B35B1F','#327C64']; models=['bert','roberta','distilbert']; names=['C1 BERT','C2 RoBERTa','C3 DistilBERT']; modes=['fixed','global','tuned']; ticks=['0,5','Chung','Từng nhãn']
a=pd.read_csv(D/'all_runs.csv'); means=pd.read_csv(D/'mean_std.csv'); per=pd.read_csv(D/'rare_labels_test.csv'); curves=pd.read_csv(D/'learning_curves.csv')
manifest={}
def save(fig,name,sources):
 fig.savefig(F/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.12);plt.close(fig)
 manifest[name]={'sources':sources,'sha256':hashlib.sha256((F/(name+'.png')).read_bytes()).hexdigest()}
def group(system,split,mode):
 return means[(means.system==system)&(means.split==split)&(means.threshold_mode==mode)].iloc[0]
# A: categorical threshold rules, not a continuous threshold sweep.
fig,axes=plt.subplots(2,3,figsize=(12.3,6.4),sharex=True)
for row,split in enumerate(['validation','test']):
 for j,(metric,title,lim) in enumerate([('macro_f1','Macro-F1 ↑',(.15,.56)),('micro_f1','Micro-F1 ↑',(.3,.60)),('hamming_loss','Hamming Loss ↓',(.02,.065))]):
  ax=axes[row,j]
  for s,label,c in [('A_standard','A thường',colors[0]),('A_balanced','A cân bằng',colors[1])]:
   vals=[group(s,split,m)[metric+'_mean'] for m in modes]
   ax.plot(range(3),vals,marker='o',lw=2,color=c,label=label)
  ax.set_ylim(*lim);ax.set_xticks(range(3),ticks);ax.grid(alpha=.2);ax.set_title(title if row==0 else '')
  if j==0:ax.set_ylabel('Validation' if row==0 else 'Test')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,1.02));fig.tight_layout(rect=[0,0,1,.94]);save(fig,'a_thresholds',['all_runs.csv','mean_std.csv'])
# B: very high Hamming is shown in its own panel.
fig,axes=plt.subplots(1,3,figsize=(12.3,3.7))
for ax,metric,title,lim in zip(axes,['macro_f1','micro_f1','hamming_loss'],['Macro-F1 ↑','Micro-F1 ↑','Hamming Loss ↓'],[(.08,.19),(.08,.20),(.24,.57)]):
 for split,c,style in [('validation',colors[0],'-'),('test',colors[1],'--')]:
  ax.plot(range(3),[group('B_bart_mnli',split,m)[metric+'_mean'] for m in modes],marker='o',color=c,label=split,lw=2,linestyle=style)
 ax.set_xticks(range(3),ticks);ax.set_title(title);ax.set_ylim(*lim);ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.5,1.04),frameon=False);fig.tight_layout(rect=[0,0,1,.88]);save(fig,'b_thresholds',['mean_std.csv'])
# Actual tensor shapes explain max_length versus hidden size.
fig,ax=plt.subplots(figsize=(12.3,3.4));ax.set(xlim=(0,12.3),ylim=(0,3.4));ax.axis('off')
blocks=[(.1,2.1,'16 câu\nIDs và mask\n16 × 128'),(2.65,3.0,'Encoder pretrained\nHidden states\n16 × 128 × 768'),(6.1,2.5,'Head phân loại\n28 logits / câu\n16 × 28'),(9.05,3.1,'Sigmoid → ngưỡng\nQuyết định 28 nhãn\n16 × 28')]
for x,ww,text in blocks:
 ax.add_patch(FancyBboxPatch((x,1.25),ww,1.45,boxstyle='round,pad=.03',facecolor='#F3F6F8',edgecolor='#546A7B'))
 ax.text(x+ww/2,1.975,text,ha='center',va='center',fontsize=16,linespacing=1.6)
for x in [2.25,5.7,8.65]:ax.annotate('',(x+.3,1.96),(x,1.96),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.text(6.15,.75,'Ví dụ một batch đủ 16 câu, padding đến 128 token; nhãn thật cũng có dạng 16 × 28.',ha='center',fontsize=16)
ax.text(6.15,.24,'128 là số token tối đa; 768 là hidden size. C1/C2 có 12 lớp encoder; C3 có 6 lớp.',ha='center',fontsize=16)
save(fig,'tensor_shapes',['scores.json: parameters','encoding_audit.json (verified in previous revision)'])
# Per-seed learning traces, retains each measured epoch.
fig,axes=plt.subplots(2,3,figsize=(12.3,6.7),sharex='col')
for j,m in enumerate(models):
 for seed,c in zip([42,123,2026],colors):
  q=curves[(curves.architecture==m)&(curves.seed==seed)].sort_values('epoch')
  for r,key in [(0,'train_loss'),(1,'validation_macro_f1')]:axes[r,j].plot(q.epoch,q[key],marker='o',color=c,label=f'Seed {seed}',lw=1.8,markersize=4)
 axes[0,j].set_title(names[j],fontweight='bold');axes[0,j].set_ylim(.025,.20);axes[1,j].set_ylim(.10,.53)
 for r in [0,1]:axes[r,j].grid(alpha=.2);axes[r,j].set_xticks(range(1,5 if m=='bert' else 4))
 axes[1,j].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss train theo log');axes[1,0].set_ylabel('Macro-F1 validation tại 0,5')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.02));fig.tight_layout(rect=[0,0,1,.95]);save(fig,'seed_curves',['learning_curves.csv'])
# Checkpoint diagnostic uses eval-mode predictions on full splits.
diag=[read(D/(m+'_diagnostic.json')) for m in models];fig,axes=plt.subplots(1,2,figsize=(12.3,4.2));x=np.arange(3);bw=.34
for ax,key,title,ylim in [(axes[0],'macro_f1','Macro-F1 tại 0,5 ↑',(0,.86)),(axes[1],'bce','BCE không trọng số ↓',(0,.145))]:
 for j,(split,c) in enumerate([('train',colors[0]),('validation',colors[1])]):
  vals=[next(r[key] for r in d['rows'] if r['split']==split) for d in diag]
  bars=ax.bar(x+(j-.5)*bw,vals,bw,label=split,color=c)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+ylim[1]*.02,f'{v:.4f}',ha='center',fontsize=14)
 ax.set_title(title);ax.set_ylim(*ylim);ax.set_xticks(x,['C1 / 123','C2 / 2026','C3 / 123']);ax.grid(axis='y',alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.04),ncol=2,frameon=False);fig.tight_layout(rect=[0,0,1,.90]);save(fig,'train_validation',[m+'_diagnostic.json' for m in models])
# Precision/recall trade-off shown as arrows. No interpolation of PR curve.
fig,ax=plt.subplots(figsize=(12.3,4.8))
for m,name,c in zip(models,names,colors):
 q=[group('C_'+m,'test',mode) for mode in ['fixed','tuned']]
 xx=[r.micro_recall_mean for r in q]; yy=[r.micro_precision_mean for r in q]
 ax.annotate('',(xx[1],yy[1]),(xx[0],yy[0]),arrowprops=dict(arrowstyle='->',color=c,lw=2))
 for j,marker in [(0,'o'),(1,'s')]:
  ax.errorbar(xx[j],yy[j],xerr=q[j].micro_recall_std,yerr=q[j].micro_precision_std,fmt=marker,color=c,capsize=4,markersize=8)
  offset={'bert':[(.006,.004),(.006,.008)],'roberta':[(.008,.012),(.006,.010)],'distilbert':[(-.016,.030),(.006,-.017)]}[m][j]
  ax.text(xx[j]+offset[0],yy[j]+offset[1],name,fontsize=14,color=c)
ax.plot([],[],marker='o',ls='',color='#444',label='0,5');ax.plot([],[],marker='s',ls='',color='#444',label='Từng nhãn')
ax.set(xlim=(.44,.73),ylim=(.49,.79),xlabel='Micro-Recall ↑',ylabel='Micro-Precision ↑');ax.grid(alpha=.2);ax.legend(loc='lower left',frameon=False,title='Mũi tên: 0,5 → từng nhãn',fontsize=14,title_fontsize=15)
fig.tight_layout();save(fig,'precision_recall',['mean_std.csv'])
# Five rare labels, A contrasts weight+threshold, C contrasts threshold only.
labels=['grief','pride','relief','nervousness','embarrassment']; supports=[6,16,11,23,37]
base=read(D/'a_rare.json');fig,axes=plt.subplots(2,2,figsize=(12.3,8.5),sharex=True)
for ax,key,title in zip(axes.ravel(),['A']+models,['A thay trọng số và ngưỡng']+names):
 for i,label in enumerate(labels):
  if key=='A':before=base['before'][i];after=base['after'][i];std0=std1=0
  else:
   before_q=per[(per.architecture==key)&(per.label==label)&(per.threshold_mode=='fixed')].f1
   after_q=per[(per.architecture==key)&(per.label==label)&(per.threshold_mode=='tuned')].f1
   before=before_q.mean();after=after_q.mean();std0=before_q.std(ddof=1);std1=after_q.std(ddof=1)
  ax.plot([before,after],[i,i],color='#8D9AA3',lw=1.5,zorder=1)
  for v,sd,dy,c,m in [(before,std0,-.10,'#7B8790','o'),(after,std1,.10,colors[0],'s')]:ax.errorbar(v,i+dy,xerr=sd,fmt=m,color=c,markersize=6,capsize=3,lw=1.4)
 ax.set_title(title);ax.set_yticks(range(5),[f'{l} (n={n})' for l,n in zip(labels,supports)]);ax.invert_yaxis();ax.set_xlim(-.10,.70);ax.set_xticks([0,.2,.4,.6]);ax.grid(axis='x',alpha=.2);ax.set_xlabel('F1 test ↑')
axes[0,0].plot([],[],color='#7B8790',marker='o',ls='',label='Trước');axes[0,0].plot([],[],color=colors[0],marker='s',ls='',label='Sau')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.02));fig.tight_layout(rect=[0,0,1,.95]);save(fig,'rare_labels',['rare_labels_test.csv','a_rare.json'])
# Error groups use category-specific denominators, never stacked as a partition.
err=pd.read_csv(D/'error_counts.csv');fig,axes=plt.subplots(3,1,figsize=(12.3,6.6),sharex=True)
for ax,category,title in zip(axes,['partial_multi_label','rare_false_negative','missed_extra_pair'],['Đúng một phần câu đa nhãn, còn bỏ sót (837 câu đủ điều kiện)','Bỏ sót ít nhất một nhãn hiếm (93 câu đủ điều kiện)','Đồng thời có FN và FP (5.427 câu test)']):
 for i,(m,name,c) in enumerate(zip(models,names,colors)):
  r=err[(err.architecture==m)&(err.category==category)].iloc[0];v=100*r.fraction_eligible
  ax.barh(i,v,height=.6,color=c);inside=v>70
  ax.text(v-1 if inside else v+1,i,f'{int(r.error_samples)}/{int(r.eligible_samples)} = {v:.2f}%',va='center',ha='right' if inside else 'left',color='white' if inside else '#222',fontsize=15)
 ax.set_yticks(range(3),names);ax.invert_yaxis();ax.set_title(title,fontsize=16,loc='left');ax.grid(axis='x',alpha=.2);ax.set_xlim(0,100)
axes[-1].set_xlabel('Tỷ lệ câu thuộc nhóm lỗi (%) ↓');fig.tight_layout();save(fig,'error_groups',['error_counts.csv'])
# FN→FP co-occurrence heatmap, absolute counts, not a confusion matrix.
pairs=pd.read_csv(D/'top_error_pairs.csv').head(5);arr=pairs[models].to_numpy(int);fig,ax=plt.subplots(figsize=(12.3,4.6));ax.imshow(arr,cmap='Blues',vmin=0,vmax=80,aspect='auto')
ax.set_xticks(range(3),names);ax.set_yticks(range(5),['FN '+r.missed_label+' → FP '+r.extra_label for _,r in pairs.iterrows()]);ax.set_xlabel('Checkpoint đại diện tại ngưỡng 0,5')
for i in range(5):
 for j in range(3):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=18,fontweight='bold',color='white' if arr[i,j]>=55 else '#111')
fig.tight_layout();save(fig,'error_pairs',['top_error_pairs.csv'])
# External small set: aggregate each checkpoint independently before summarizing seeds.
ext=pd.DataFrame([{k:v for k,v in r.items() if k!='per_label'} for r in read(D/'summary.json')]);systems=['A_standard','A_balanced','B']+models;extnames=['A thường','A cân bằng','BART MNLI']+names
fig,axes=plt.subplots(2,2,figsize=(12.3,8.2))
for ax,key,title,limit in zip(axes.ravel(),['macro_f1','micro_f1','hamming_loss','exact'],['Macro-F1 trên 5 nhóm ↑','Micro-F1 trên 5 nhóm ↑','Hamming Loss trên 50 quyết định ↓','Số câu đúng đủ trong 10 câu ↑'],[1,1,1,10]):
 for j,(mode,c,mk,dy) in enumerate([('fixed','#7B8790','o',-.12),('tuned',colors[0],'s',.12)]):
  for i,system in enumerate(systems):
   q=ext[(ext.model==system)&(ext['mode']==mode)][key]
   assert len(q)==(3 if system in models else 1), (system,mode,key,len(q))
   assert np.isfinite(q).all()
   v=q.mean();sd=q.std(ddof=1) if len(q)==3 else 0
   ax.errorbar(v,i+dy,xerr=sd,fmt=mk,color=c,capsize=3,markersize=6)
  ax.plot([],[],marker=mk,ls='',color=c,label='0,5' if mode=='fixed' else 'Từng nhãn đã khóa')
 ax.set_yticks(range(6),extnames);ax.invert_yaxis();ax.set_xlim(-limit*.02,limit);ax.set_title(title,fontsize=16);ax.grid(axis='x',alpha=.2)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.02));fig.tight_layout(rect=[0,0,1,.95]);save(fig,'external_metrics',['summary.json'])
# Local explanations anchored in raw 28-label scores and frozen per-label thresholds.
s=read(D/'scores.json');fig,axes=plt.subplots(2,3,figsize=(12.3,7.1))
for j,(model,seed) in enumerate(zip(models,[123,2026,123])):
 run=next(r for r in s['runs'] if r['model']==model and r['seed']==seed)
 for row,(case,ll) in enumerate([('N04',['fear','nervousness','surprise']),('N08',['fear','nervousness','sadness'])]):
  ax=axes[row,j];vals=[];th=[]
  for label in ll:
   idx=s['labels28'].index(label);vals.append(run['scores28'][s['ids'].index(case)][idx]);th.append(run['thresholds']['tuned'][idx])
  ax.barh(range(3),vals,color=[colors[0] if v>=t else '#AAB5BD' for v,t in zip(vals,th)],height=.45)
  ax.scatter(th,range(3),marker='|',s=230,linewidth=2,color='#B04727',zorder=4)
  for i,(v,t) in enumerate(zip(vals,th)):
   ax.text(.99,i-.34,f'{v:.4f} / {t:.2f}',ha='right',va='center',fontsize=14,bbox=dict(facecolor='white',edgecolor='none',alpha=.83,pad=1))
  ax.set_yticks(range(3),ll);ax.invert_yaxis();ax.set_xlim(0,1.04);ax.set_ylim(2.65,-.65);ax.set_title(f'{case} · {names[j]}',fontsize=16);ax.grid(axis='x',alpha=.2)
  if row==1:ax.set_xlabel('Điểm sigmoid (0–1)',fontsize=14)
fig.text(.5,1.01,'Cột xanh: score đạt ngưỡng; xám: chưa đạt; vạch đỏ: ngưỡng đã khóa.',ha='center',fontsize=15)
fig.tight_layout();save(fig,'case_scores',['scores.json'])
# Supplementary views use the same frozen run set as the main figures.
fig,ax=plt.subplots(figsize=(12.3,5.2))
for m,name,c in zip(models,names,colors):
 q=curves[curves.architecture==m].groupby('epoch').validation_macro_f1.agg(['mean','std'])
 ax.plot(q.index,q['mean'],marker='o',label=name,color=c,lw=2)
 ax.fill_between(q.index,q['mean']-q['std'],q['mean']+q['std'],color=c,alpha=.15)
ax.set(xlabel='Epoch',ylabel='Macro-F1 validation tại 0,5',ylim=(.1,.53));ax.set_xticks([1,2,3,4]);ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();save(fig,'validation_mean',['learning_curves.csv'])
fig,axes=plt.subplots(1,2,figsize=(12.3,5.0));x=np.arange(3);bw=.34
for ax,key,title in zip(axes,['macro_f1','micro_f1'],['Macro-F1 test ↑','Micro-F1 test ↑']):
 for j,(mode,c,label) in enumerate([('fixed','#7B8790','0,5'),('tuned',colors[0],'Từng nhãn')]):
  q=[group('C_'+m,'test',mode) for m in models]
  ax.bar(x+(j-.5)*bw,[r[key+'_mean'] for r in q],bw,yerr=[r[key+'_std'] for r in q],capsize=4,color=c,label=label)
 ax.set_xticks(x,['C1','C2','C3']);ax.set_ylim(0,.7);ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=[0,0,1,.90]);save(fig,'test_f1_summary',['mean_std.csv'])
pred=read(W/'external_probe/predictions.json');fig,axes=plt.subplots(1,3,figsize=(11.7,3.9))
for ax,(m,seed),name in zip(axes,[('bert',123),('roberta',2026),('distilbert',123)],names):
 q=[r for r in pred if r['model']==m and r['seed']==seed and r['mode']=='tuned'];arr=np.array([[sum(label in r[k] for r in q) for k in ['fn','fp']] for label in ['anger','fear','joy','sadness','surprise']])
 ax.imshow(arr,vmin=0,vmax=4,cmap='Blues',aspect='auto');ax.set_xticks([0,1],['FN','FP']);ax.set_yticks(range(5),['anger','fear','joy','sadness','surprise']);ax.set_title(f'{name} / {seed}',fontsize=17)
 for i in range(5):
  for j in range(2):ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>=3 else 'black',weight='bold',fontsize=14)
fig.tight_layout();save(fig,'external_errors',['external_probe/predictions.json'])

(W/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('Created',len(manifest),'plots, no training or threshold changes.')

