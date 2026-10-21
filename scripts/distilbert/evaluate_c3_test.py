"""Evaluate Huy's three existing C3 checkpoints; freeze validation choices first.

No training and no mutation of original run.json or training artifacts.
python -m scripts.distilbert.evaluate_c3_test --device cuda
"""
import argparse
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.paths import ROOT
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.models.baseline import load_aligned_scores, tune_global_threshold, tune_thresholds
from src.models.distilbert_study import read_json, load_bundle, EmotionDataset, make_loader
from src.evaluation.distilbert_thresholds import load_c3_thresholds
from src.evaluation.metrics import evaluate_multilabel

SEEDS = (42, 123, 2026)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def freeze(output):
    frames, names, data = load_goemotions(ROOT, write_metadata=False, splits=('train', 'validation'))
    truth = multi_hot(frames['validation'].labels.tolist())
    sources, validation_results, signature = [], {}, None
    for seed in SEEDS:
        folder = ROOT / f'data/processed/c3_distilbert/full/seed_{seed}'
        meta = read_json(folder/'run.json')
        if (meta['status'] != 'complete' or meta['mode'] != 'full' or meta['config']['seed'] != seed
                or meta['n_train'] != 43410 or meta['n_validation'] != 5426
                or meta['label_names'] != names or meta['data_revision'] != REVISION):
            raise ValueError(f'Invalid full training run: {seed}')
        for filename, expected in meta['artifact_sha256'].items():
            if sha256(folder/filename) != expected:
                raise ValueError(f'Artifact hash mismatch: {seed}/{filename}')
        config = {k:v for k,v in meta['config'].items() if k != 'seed'}
        current = (config, meta['source_sha256'], meta['data_sha256'])
        if signature is not None and signature != current:
            raise ValueError('Do not mix training configurations or datasets')
        signature = current
        scores = load_aligned_scores(folder/'validation_scores.npz', frames['validation'].id, names)
        fixed = evaluate_multilabel(truth, scores, names, .5)
        if fixed != read_json(folder/'validation_metrics.json'):
            raise ValueError('Saved validation metrics do not reproduce')
        tuned = load_c3_thresholds(folder, meta, 'per_label')
        if not np.array_equal(tuned, tune_thresholds(truth, scores)):
            raise ValueError('Saved per-label thresholds do not reproduce on validation')
        global_threshold, curve = tune_global_threshold(truth, scores)
        thresholds = {'fixed': .5, 'global': float(global_threshold), 'tuned': tuned.tolist()}
        sources.append({'seed':seed, 'run_dir':folder.relative_to(ROOT).as_posix(),
                        'metadata_sha256':sha256(folder/'run.json'),
                        'model_sha256':meta['artifact_sha256']['best/model.safetensors'],
                        'validation_scores_sha256':sha256(folder/'validation_scores.npz'),
                        'thresholds':thresholds, 'best_epoch':meta['best_epoch'],
                        'validation_macro_f1_fixed':fixed['macro_f1']})
        validation_results[str(seed)] = {mode:evaluate_multilabel(truth,scores,names,t) for mode,t in thresholds.items()}
    protocol = {'schema':'huy-c3-final-test-v1','data_revision':REVISION,'label_names':names,
                'seeds':list(SEEDS),'runs':sources,
                'selection_data':'validation','checkpoint_selection':'highest validation Macro-F1@0.5; earliest epoch on tie',
                'representative_seed':max(sources,key=lambda r:(r['validation_macro_f1_fixed'],-r['seed']))['seed'],
                'representative_selection':'highest validation Macro-F1@0.5; smaller seed on tie',
                'test_samples':5427,'threshold_grid':[round(i*.05,2) for i in range(1,20)],
                'source_sha256':sha256(Path(__file__)),
                'note':'Existing test data was previously explored by the group; no test-driven model or threshold selection.'}
    path = output/'final_protocol.json'
    if path.exists():
        saved = read_json(path)
        if {k:v for k,v in saved.items() if k!='frozen_at_utc'} != protocol:
            raise ValueError('Existing test protocol differs; refusing to overwrite')
        protocol = saved
    else:
        protocol['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        save(path,protocol)
    save(output/'validation_results.json',validation_results)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/c3_distilbert/test_evaluation_20261021')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    protocol=freeze(output)
    fingerprint=sha256(output/'final_protocol.json')
    print(f'FROZEN all three runs; representative seed={protocol["representative_seed"]}',flush=True)
    # Test labels are read only after all three choices have been frozen.
    frames,names,data=load_goemotions(ROOT,write_metadata=False,splits=('test',))
    frame=frames['test'];truth=multi_hot(frame.labels.tolist())
    for frozen in protocol['runs']:
        seed=frozen['seed'];folder=ROOT/frozen['run_dir'];dest=output/f'seed_{seed}'
        dest.mkdir(exist_ok=True)
        result_path=dest/'test_results.json';score_path=dest/'test_scores.npz'
        if result_path.exists():
            saved=read_json(result_path)
            if saved['protocol_sha256']!=fingerprint or saved['scores_sha256']!=sha256(score_path):
                raise ValueError('Existing test evaluation has changed')
            scores=load_aligned_scores(score_path,frame.id,names)
            recomputed={mode:evaluate_multilabel(truth,scores,names,t) for mode,t in frozen['thresholds'].items()}
            if recomputed != saved['results']:raise ValueError('Existing test metrics mismatch')
            print(f'Seed {seed}: verified saved test evaluation',flush=True)
            continue
        bundle=load_bundle(folder,args.device)
        dataset=EmotionDataset(frame,bundle['tokenizer'],bundle['metadata']['config']['max_length'])
        loader=make_loader(dataset,bundle['tokenizer'],16,shuffle=False)
        blocks=[]
        with torch.inference_mode():
            for step,batch in enumerate(loader,1):
                inputs={k:v.to(args.device) for k,v in batch.items() if k!='labels'}
                blocks.append(bundle['model'](**inputs).logits.float().sigmoid().cpu().numpy())
                if step%100==0 or step==len(loader):print(f'Seed {seed}: test batches {step}/{len(loader)}',flush=True)
        scores=np.concatenate(blocks)
        if scores.shape!=(5427,28) or not np.isfinite(scores).all():raise ValueError('Invalid test scores')
        temporary=dest/'test_scores.tmp.npz'
        np.savez_compressed(temporary,ids=frame.id.to_numpy(dtype=str),label_names=np.asarray(names,dtype=str),scores=scores)
        temporary.replace(score_path)
        results={mode:evaluate_multilabel(truth,scores,names,t) for mode,t in frozen['thresholds'].items()}
        save(result_path,{'seed':seed,'split':'test','protocol_sha256':fingerprint,
                          'scores_sha256':sha256(score_path),'data_sha256':data['files'][0]['sha256'],
                          'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
                          'environment':{'python':platform.python_version(),'torch':torch.__version__,
                                         'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None},
                          'results':results})
        print(f'Seed {seed}: fixed Macro-F1={results["fixed"]["macro_f1"]:.6f}; tuned={results["tuned"]["macro_f1"]:.6f}',flush=True)
        del bundle,dataset,loader,blocks;gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    print('PASS: three test evaluations complete; original training runs unchanged.',flush=True)


if __name__=='__main__':
    main()
