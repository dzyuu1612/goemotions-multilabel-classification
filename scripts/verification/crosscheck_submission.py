"""Independent sklearn check of the frozen benchmark and external ten sentences."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,hamming_loss
from src.notebook_support import artifacts_root
from src.datasets.goemotions import load_goemotions,multi_hot
from src.models.baseline import load_aligned_scores
from src.paths import ROOT

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def metric(y,p):
    return {'macro_f1':f1_score(y,p,average='macro',zero_division=0),'micro_f1':f1_score(y,p,average='micro',zero_division=0),
            'macro_precision':precision_score(y,p,average='macro',zero_division=0),'macro_recall':recall_score(y,p,average='macro',zero_division=0),
            'micro_precision':precision_score(y,p,average='micro',zero_division=0),'micro_recall':recall_score(y,p,average='micro',zero_division=0),'hamming_loss':hamming_loss(y,p)}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts-root',type=Path);args=parser.parse_args()
    root=args.artifacts_root or artifacts_root();published=ROOT/'results/final_submission';external=published/'external_probe'
    frames,labels,_=load_goemotions(root,write_metadata=False);truth={k:multi_hot(v.labels.tolist(),28) for k,v in frames.items()}
    records=pd.read_csv(published/'report_data/all_runs.csv');count=0;maxdiff=0
    def check(system,seed,folder,paths,choices):
        nonlocal count,maxdiff
        for split,path in paths.items():
            scores=load_aligned_scores(path,frames[split].id.tolist(),labels)
            for choice in choices:
                q=records[(records.system==system)&(records.split==split)&(records.threshold_mode==choice['threshold_mode'])]
                q=q[q.seed.isna()] if seed is None else q[q.seed==seed]
                assert len(q)==1
                actual=metric(truth[split],scores>=np.asarray(choice['thresholds']))
                for k,v in actual.items():
                    diff=abs(v-q.iloc[0][k]);maxdiff=max(maxdiff,diff);assert diff<1e-12,(system,seed,split,k)
                count+=1
    base=root/'data/processed/baseline';protocol=read(base/'final_protocol.json')
    for variant in ['standard','balanced']:
        folder=base/('full' if variant=='standard' else 'balanced/full')
        check('A_'+variant,42,folder,{'validation':folder/'validation_scores.npz','test':base/f'final/{variant}_test_scores.npz'},[c for c in protocol['configurations'] if c['variant']==variant])
    folder=root/'data/processed/zero_shot/full'
    check('B_bart_mnli',None,folder,{s:folder/(s+'_scores.npz') for s in ['validation','test']},read(folder/'final_protocol.json')['configurations'])
    for model in ['bert','roberta','distilbert']:
        for seed in [42,123,2026]:
            folder=root/f'data/processed/transformers/{model}/seed_{seed}/full/standard'
            check('C_'+model,seed,folder,{s:folder/(s+'_scores.npz') for s in ['validation','test']},read(folder/'final_protocol.json')['configurations'])
    fixture=read(external/'new10.json');scores=read(external/'scores.json');predictions=read(external/'predictions.json');summaries=read(external/'summary.json')
    assert hashlib.sha256((external/'new10.json').read_bytes()).hexdigest()==read(external/'plan.json')['fixture_sha256']
    assert scores['ids']==[r['id'] for r in fixture['rows']]
    extcount=0
    for run in scores['runs']:
        for mode in ['fixed','tuned']:
            q=[r for r in predictions if r['model']==run['model'] and r['seed']==run['seed'] and r['mode']==mode];assert len(q)==10
            y=np.array([r['truth'] for r in fixture['rows']],dtype=int)
            p28=np.asarray(run['scores28'])>=np.asarray(run['thresholds'][mode])
            p=np.stack([p28[:,[scores['labels28'].index(label) for label in fixture['mapping'][group]]].any(1) for group in fixture['labels']],axis=1)
            for r,source,yy,pp in zip(q,fixture['rows'],y,p):
                assert r['id']==source['id'] and r['truth']==source['labels']
                assert r['mapped5']==[l for l,b in zip(fixture['labels'],pp) if b]
                assert r['fn']==[l for l,a,b in zip(fixture['labels'],yy,pp) if a and not b]
                assert r['fp']==[l for l,a,b in zip(fixture['labels'],yy,pp) if b and not a]
            reference=next(r for r in summaries if r['model']==run['model'] and r['seed']==run['seed'] and r['mode']==mode)
            values=metric(y,p);values['exact']=int((y==p).all(1).sum())
            for k,v in values.items():
                if k not in reference:continue
                diff=abs(v-reference[k]);maxdiff=max(maxdiff,diff);assert diff<1e-12
            extcount+=1
    splits=0
    for model in ['bert','roberta','distilbert']:
        diagnostic=read(external/f'{model}_diagnostic.json')
        for split in ['train','validation']:
            path=external/f'{model}_{split}_diagnostic.npz'
            with np.load(path,allow_pickle=False) as z:
                s=z['scores'];assert z['ids'].tolist()==frames[split].id.tolist();assert z['label_names'].tolist()==labels
            ref=next(r for r in diagnostic['rows'] if r['split']==split);values=metric(truth[split],s>=.5)
            for k in ['macro_f1','micro_f1','hamming_loss']:assert abs(values[k]-ref[k])<1e-12
            splits+=1
    result={'passed':True,'benchmark_metric_configurations':count,'external_metric_configurations':extcount,'predictions_checked':len(predictions),'diagnostic_splits_checked':splits,'metric_max_abs_difference':float(maxdiff),'frozen_fixture_sha256':hashlib.sha256((external/'new10.json').read_bytes()).hexdigest(),'training_repeated':False,'threshold_search_repeated':False}
    (published/'independent_crosscheck.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('PASS independent sklearn:',count,'benchmark configurations,',extcount,'external configurations,',splits,'diagnostic splits; max difference',maxdiff)
if __name__=='__main__':main()
