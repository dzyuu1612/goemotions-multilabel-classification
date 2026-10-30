from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,gc,time
import numpy as np
import pandas as pd
import torch
from src.apps.demo import TransformerDemo
from src.apps.model_selector import BaselineDemo,ZeroShotDemo
from src.paths import ROOT
import argparse
from src.final_submission import prepare_runtime
from src.models.transformer import load_transformer_run
from src.models.zero_shot import map_pipeline_scores
from src.evaluation.protocols import validate_protocol
from src.evaluation.metrics import evaluate_multilabel
from src.datasets.goemotions import load_goemotions,multi_hot,sha256

parser=argparse.ArgumentParser(description='Replay the frozen human-labeled ten-sentence probe without training or threshold tuning.')
parser.add_argument('--artifacts-root',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'artifacts/external_probe_replay')
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
parser.add_argument('--skip-diagnostics',action='store_true')
args=parser.parse_args()
W=ROOT/'results/final_submission/external_probe';A=args.artifacts_root or prepare_runtime()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
torch.set_num_threads(4)
fixture=read(W/'new10.json');plan=read(W/'plan.json')
assert sha256(W/'new10.json')==plan['fixture_sha256']
frames,labels,manifest=load_goemotions(A,write_metadata=False)
texts=[r['text'] for r in fixture['rows']];human=np.array([r['truth'] for r in fixture['rows']],dtype=int)
group_ids=[[labels.index(n) for n in fixture['mapping'][c]] for c in fixture['labels']]
out=args.output;out.mkdir(parents=True,exist_ok=True)
reps={m:max([42,123,2026],key=lambda s:(read(A/f'data/processed/transformers/{m}/seed_{s}/full/standard/validation_metrics.json')['macro_f1'],-s)) for m in ['bert','roberta','distilbert']}

def infer(predictor,items,truth=None,label=''):
    points=[];loss=0.;n=len(items)
    for start in range(0,n,64):
        inputs=predictor.tokenizer(items[start:start+64],padding=True,truncation=True,
            max_length=predictor.max_length,return_tensors='pt').to(predictor.device)
        with torch.inference_mode():
            logits=predictor.model(**inputs).logits.float()
            scores=torch.sigmoid(logits).cpu().numpy()
            if truth is not None:
                y=torch.as_tensor(truth[start:start+len(scores)],dtype=torch.float32,device=logits.device)
                loss+=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum'))
        points.append(scores)
        if n>10000 and (start//64)%150==0:print(label,start,'/',n,flush=True)
    return np.concatenate(points),loss/(n*28) if truth is not None else None

bp=A/'data/processed/baseline/final_protocol.json';baseline_protocol=read(bp)
runs=[]
for model,seed in [('A_standard',None),('A_balanced',None),('B',None)]+[(m,s) for m in ['bert','roberta','distilbert'] for s in [42,123,2026]]:
    name=f'{model}_{seed}'
    saved=out/(name+'.json')
    if saved.exists():
        v=read(saved)
        if v['fixture_sha256']!=plan['fixture_sha256']:raise ValueError('Old output uses another fixture')
        if model.startswith('A_'):
            variant=model[2:];folder=A/baseline_protocol['runs'][variant]['folder'];protofile=bp;weight=folder/'model.joblib'
        elif model=='B':
            protofile=A/'data/processed/zero_shot/full/final_protocol.json';weight=None
        else:
            folder=A/f'data/processed/transformers/{model}/seed_{seed}/full/standard';protofile=folder/'final_protocol.json';weight=folder/'checkpoint/model.safetensors'
        if v['protocol_sha256']!=sha256(protofile) or (weight is not None and v['weights_sha256']!=sha256(weight)):
            raise ValueError('Cached probe output belongs to a different model or protocol')
        runs.append(v);continue
    print('RUN',name,flush=True)
    started=datetime.now(timezone.utc).isoformat()
    if model.startswith('A_'):
        variant=model[2:];predictor=BaselineDemo(A,variant)
        p=A/baseline_protocol['runs'][variant]['folder']/'model.joblib'
        digest=sha256(p);assert digest==baseline_protocol['runs'][variant]['artifact_sha256']['model.joblib']
        points=predictor.model.predict_proba(texts)
        tuned=next(c['thresholds'] for c in baseline_protocol['configurations'] if c['variant']==variant and c['threshold_mode']=='tuned')
        protocol_sha=sha256(bp)
        params={'features':len(predictor.model.named_steps['tfidf'].vocabulary_),'class_weight':'balanced' if variant=='balanced' else None}
    elif model=='B':
        folder=A/'data/processed/zero_shot/full';proto=read(folder/'final_protocol.json')
        validate_protocol(folder,proto,labels)
        predictor=ZeroShotDemo(A,'cpu')
        p=Path(predictor.classifier.model.name_or_path)/'model.safetensors';digest=sha256(p)
        points=map_pipeline_scores(predictor.classifier(texts,candidate_labels=labels,multi_label=True,
                   hypothesis_template=predictor.template,batch_size=8),labels)
        tuned=next(c['thresholds'] for c in proto['configurations'] if c['threshold_mode']=='tuned')
        protocol_sha=sha256(folder/'final_protocol.json')
        params={'checkpoint':proto['checkpoint'],'revision':proto['model_revision'],'template':predictor.template}
    else:
        folder=A/f'data/processed/transformers/{model}/seed_{seed}/full/standard'
        meta=load_transformer_run(folder,require_full=True);proto=read(folder/'final_protocol.json')
        validate_protocol(folder,proto,labels);checkpoint=folder/'checkpoint'
        predictor=TransformerDemo.from_checked_checkpoint(checkpoint,meta,args.device)
        p=checkpoint/'model.safetensors';digest=sha256(p)
        points,_=infer(predictor,texts)
        tuned=next(c['thresholds'] for c in proto['configurations'] if c['threshold_mode']=='tuned')
        protocol_sha=sha256(folder/'final_protocol.json')
        params={k:meta['config'][k] for k in ['epochs','learning_rate','batch_size','max_length']}
        params.update(hidden_size=getattr(predictor.model.config,'hidden_size',getattr(predictor.model.config,'dim',None)),
            parameters=sum(p.numel() for p in predictor.model.parameters()))
        if seed==reps[model] and not args.skip_diagnostics:
            diagnostic=[]
            for split in ['train','validation']:
                truth=multi_hot(frames[split].labels.tolist(),28)
                s,bce=infer(predictor,frames[split].text.tolist(),truth,model+' '+split)
                np.savez_compressed(out/f'{model}_{split}_diagnostic.npz',scores=s,ids=frames[split].id.to_numpy(dtype=str),label_names=np.array(labels,dtype=str))
                met=evaluate_multilabel(truth,s,labels,.5)
                diagnostic.append(dict(split=split,n=len(s),bce=bce,
                    **{k:met[k] for k in ['macro_f1','micro_f1','hamming_loss']}))
            write(out/f'{model}_diagnostic.json',dict(seed=seed,model=model,rows=diagnostic,
                    mode='Frozen checkpoint in eval mode, no training, no dropout, unweighted mean BCE. Not epoch training loss.'))
    assert predictor.labels==labels
    assert sha256(p)==digest
    v=dict(model=model,seed=seed,fixture_sha256=plan['fixture_sha256'],started_at_utc=started,
        weights_sha256=digest,protocol_sha256=protocol_sha,parameters=params,scores28=np.asarray(points).tolist(),
        thresholds={'fixed':.5,'tuned':tuned},weights_unchanged=True)
    write(saved,v);runs.append(v)
    del predictor
    gc.collect();torch.cuda.empty_cache()

summary=[];predictions=[]
for run in runs:
    s=np.array(run['scores28']);assert s.shape==(10,28) and np.isfinite(s).all()
    for mode,limits in run['thresholds'].items():
        p28=s>=np.asarray(limits)
        p5=np.stack([p28[:,ix].any(axis=1) for ix in group_ids],axis=1).astype(int)
        met=evaluate_multilabel(human,p5.astype(float),fixture['labels'],.5)
        summary.append(dict(model=run['model'],seed=run['seed'],mode=mode,n=10,n_labels=5,
             exact=int((p5==human).all(axis=1).sum()),tp=int((p5*human).sum()),
             fp=int((p5*(1-human)).sum()),fn=int(((1-p5)*human).sum()),
             **{k:met[k] for k in ['micro_precision','micro_recall','micro_f1','macro_f1','hamming_loss']},per_label=met['per_label']))
        for i,row in enumerate(fixture['rows']):
            names=np.array(fixture['labels']);actual=human[i].astype(bool);guessed=p5[i].astype(bool)
            predictions.append(dict(id=row['id'],source_id=row['source_id'],model=run['model'],seed=run['seed'],mode=mode,
                truth=row['labels'],raw28=np.array(labels)[p28[i]].tolist(),mapped5=names[guessed].tolist(),
                fn=names[actual & ~guessed].tolist(),fp=names[~actual & guessed].tolist()))
write(out/'summary.json',summary);write(out/'predictions.json',predictions)
write(out/'scores.json',dict(ids=[r['id'] for r in fixture['rows']],labels28=labels,runs=runs))
pd.DataFrame([{k:v for k,v in r.items() if k!='per_label'} for r in summary]).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
write(out/'receipt.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),human_source_labels=True,
    plan_sha256=sha256(W/'plan.json'),fixture_sha256=sha256(W/'new10.json'),runs=12,score_vectors=120,
    predictions=240,threshold_tuning=False,training=False,labels_evaluated=fixture['labels'],
    mapping='Threshold 28 scores with frozen limits, then OR within published GoEmotions Ekman groups; ignore disgust and neutral for this five-label evaluation.',
    all_weights_unchanged=all(r['weights_unchanged'] for r in runs)))
print('COMPLETE: 12 runs, 120 score vectors, 240 predictions; cached outputs checked against current checkpoint and protocol hashes.',flush=True)
