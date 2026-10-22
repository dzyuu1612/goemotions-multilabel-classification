"""Suy luận văn bản ngoài GoEmotions; không fit hoặc tìm lại ngưỡng.

python -m scripts.analysis.evaluate_external_texts --artifacts-root PATH --verify-source
python -m scripts.analysis.evaluate_external_texts --verify-only
Nhãn cảm xúc trong fixture do AI đề xuất, chưa được con người thẩm định.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.datasets.goemotions import load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel, validate_multilabel_inputs, validate_thresholds
from src.paths import ROOT

FIXTURE = ROOT / 'examples/data/external_reviews_20.json'
OUTPUT = ROOT / 'results/external_review_probe'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def check_fixture(fixture, labels):
    rows = fixture['rows']
    if (len(rows) != 30 or len({r['id'] for r in rows}) != 30
            or len({r['text'] for r in rows}) != 30
            or sum(r['kind'] == 'external_review' for r in rows) != 20
            or sum(r['kind'] == 'controlled_probe' for r in rows) != 10
            or any(not r['text'].strip() or not r['labels'] or not set(r['labels']) <= set(labels) for r in rows)
            or fixture['human_review_status'] != 'NOT_REVIEWED'):
        raise ValueError('Fixture sai kích thước, mapping hoặc nguồn nhãn')
    return rows


def verify_source(fixture):
    source = fixture['source_dataset']
    archive = urllib.request.urlopen(source['archive_url'], timeout=60).read()
    if hashlib.sha256(archive).hexdigest() != source['archive_sha256']:
        raise ValueError('Archive nguồn đã thay đổi')
    with ZipFile(BytesIO(archive)) as z:
        for row in fixture['rows']:
            if row['kind'] != 'external_review':
                continue
            name = next(n for n in z.namelist() if n == 'sentiment labelled sentences/' + row['source_file'])
            raw = z.read(name)
            original, sentiment = raw.decode('utf-8').splitlines()[row['source_line'] - 1].rsplit('\t', 1)
            if (hashlib.sha256(raw).hexdigest() != row['source_file_sha256']
                    or original != row['text'] or int(sentiment) != row['source_sentiment']):
                raise ValueError('Câu trích không khớp dòng nguồn')


def boundary_check(rows, frames):
    result = {}
    for split, frame in frames.items():
        exact, folded = set(frame.text), set(map(normalized, frame.text))
        result[split] = {'exact_matches': [r['id'] for r in rows if r['text'] in exact],
                         'normalized_matches': [r['id'] for r in rows if normalized(r['text']) in folded]}
    # Record overlaps rather than silently dropping samples after seeing scores.
    return result


def summarize(rows, labels, records):
    metrics, predictions = [], []
    for record in records:
        scores = np.asarray(record['scores'])
        if scores.shape != (len(rows), 28):
            raise ValueError('Sai shape điểm hoặc thứ tự ID')
        for mode, limits in record['thresholds'].items():
            validate_thresholds(limits, 28)
            truth = multi_hot([[labels.index(label) for label in r['labels']] for r in rows], 28)
            validate_multilabel_inputs(truth, scores)
            predicted = scores >= np.asarray(limits)
            for kind in ('external_review', 'controlled_probe'):
                ix = np.array([r['kind'] == kind for r in rows])
                m = evaluate_multilabel(truth[ix], scores[ix], labels, limits)
                metrics.append(dict(model=record['model'], seed=record['seed'], kind=kind, mode=mode,
                    n_samples=int(ix.sum()), exact_match_count=int(np.all(predicted[ix] == truth[ix], axis=1).sum()),
                    tp=sum(p['tp'] for p in m['per_label']), fp=sum(p['fp'] for p in m['per_label']),
                    fn=sum(p['fn'] for p in m['per_label']), supported_labels=int((truth[ix].sum(axis=0) > 0).sum()),
                    **{key: m[key] for key in ('micro_precision','micro_recall','micro_f1','macro_f1','hamming_loss','empty_prediction_count')}))
            for i, row in enumerate(rows):
                actual = truth[i].astype(bool)
                predictions.append(dict(id=row['id'], kind=row['kind'], model=record['model'], seed=record['seed'], mode=mode,
                    reference_labels='; '.join(row['labels']),
                    predicted_labels='; '.join(np.asarray(labels)[predicted[i]]),
                    fn='; '.join(np.asarray(labels)[actual & ~predicted[i]]),
                    fp='; '.join(np.asarray(labels)[~actual & predicted[i]]),
                    exact_match=bool(np.array_equal(actual, predicted[i])), ambiguous=row['ambiguous']))
    return metrics, predictions


def verify_saved(output=OUTPUT):
    fixture, plan, saved = read(FIXTURE), read(output/'plan.json'), read(output/'scores.json')
    if sha256(FIXTURE) != plan['fixture_sha256'] or saved['fixture_sha256'] != plan['fixture_sha256']:
        raise ValueError('Không khớp bộ câu/nhãn đã khóa trước suy luận')
    rows = check_fixture(fixture, saved['labels'])
    if saved['ids'] != [r['id'] for r in rows] or len(saved['runs']) != 12:
        raise ValueError('Sai thứ tự ID hoặc thiếu run')
    expected = {('A_standard', None), ('A_balanced', None), ('B', None)} | {
        (a, s) for a in ('bert','roberta','distilbert') for s in (42,123,2026)}
    if {(r['model'],r['seed']) for r in saved['runs']} != expected:
        raise ValueError('Sai kiến trúc/seed')
    metrics, predictions = summarize(rows, saved['labels'], saved['runs'])
    if metrics != read(output/'metrics.json') or predictions != read(output/'predictions.json'):
        raise ValueError('Kết quả xuất không khớp điểm chưa làm tròn')
    print(f'PASS: {len(metrics)} metric rows, {len(predictions)} predictions, 12 saved runs; AI reference labels unreviewed.')
    return metrics, predictions


def run(args):
    import torch
    from src.apps.demo import TransformerDemo
    from src.apps.model_selector import BaselineDemo, ZeroShotDemo
    from src.current_report import checked_c3, current_c3_root
    from src.models.transformer import load_transformer_run
    from src.evaluation.protocols import validate_protocol
    from src.models.zero_shot import map_pipeline_scores

    torch.set_num_threads(4)
    root = args.artifacts_root.resolve()
    fixture, plan = read(FIXTURE), read(OUTPUT/'plan.json')
    if sha256(FIXTURE) != plan['fixture_sha256']:
        raise ValueError('Fixture thay đổi sau khi khóa')
    frames, labels, _ = load_goemotions(root, write_metadata=False)
    rows = check_fixture(fixture, labels)
    texts = [r['text'] for r in rows]
    boundary = boundary_check(rows, frames)
    if args.verify_source:
        verify_source(fixture)
    bp = root/'data/processed/baseline/final_protocol.json'
    base_protocol = read(bp)
    specs = [('A_standard', None), ('A_balanced', None), ('B', None)] + [
        (a, s) for a in ('bert','roberta','distilbert') for s in (42,123,2026)]
    records, explanation = [], []
    for model, seed in specs:
        print('Running', model, seed, flush=True)
        started = datetime.now(timezone.utc).isoformat()
        params, truncations = {}, []
        if model.startswith('A_'):
            variant = model.removeprefix('A_')
            predictor = BaselineDemo(root, variant)
            folder = root/base_protocol['runs'][variant]['folder']
            artifact = folder/'model.joblib'
            digest = sha256(artifact)
            if digest != base_protocol['runs'][variant]['artifact_sha256']['model.joblib']:
                raise ValueError('Sai model A so với protocol')
            limits = next(c['thresholds'] for c in base_protocol['configurations']
                          if c['variant'] == variant and c['threshold_mode'] == 'tuned')
            points = predictor.model.predict_proba(texts)
            vectorizer = predictor.model.named_steps['tfidf']
            params = dict(n_features=len(vectorizer.vocabulary_),class_weight='balanced' if variant == 'balanced' else None)
            # Exact linear decomposition for the first review, label approval.
            clf = next(step for step in predictor.model.named_steps.values() if hasattr(step,'estimators_'))
            est = clf.estimators_[labels.index('approval')]
            x = vectorizer.transform([texts[0]])
            contributions = x.multiply(est.coef_).toarray()[0]
            logit = float(contributions.sum()+est.intercept_[0])
            if not np.isclose(1/(1+np.exp(-logit)),points[0,labels.index('approval')]):
                raise ValueError('Giải thích tuyến tính không khớp suy luận')
            names = vectorizer.get_feature_names_out()
            terms = sorted(np.flatnonzero(contributions),key=lambda j:abs(contributions[j]),reverse=True)
            explanation.append(dict(model=model,id=rows[0]['id'],label='approval',intercept=float(est.intercept_[0]),
                contribution_sum=float(contributions.sum()),logit=logit,score=float(points[0,labels.index('approval')]),
                terms=[dict(feature=names[j],contribution=float(contributions[j])) for j in terms]))
            protocol_path = bp
        elif model == 'B':
            folder = root/'data/processed/zero_shot/full'
            protocol_path = folder/'final_protocol.json'
            protocol = read(protocol_path)
            validate_protocol(folder, protocol, labels)
            predictor = ZeroShotDemo(root, 'cpu')
            points = map_pipeline_scores(predictor.classifier(texts,candidate_labels=labels,multi_label=True,
                hypothesis_template=predictor.template,batch_size=8),labels)
            limits = next(c['thresholds'] for c in protocol['configurations'] if c['threshold_mode']=='tuned')
            snapshot = Path(predictor.classifier.model.name_or_path)
            artifact = snapshot/'model.safetensors';digest=sha256(artifact)
            params = dict(checkpoint=protocol['checkpoint'],revision=protocol['model_revision'],
                          hypothesis_template=predictor.template,max_length=1024)
        else:
            if model == 'distilbert':
                folder = current_c3_root()/f'seed_{seed}'
                meta, protocol = checked_c3(current_c3_root(),seed)
                checkpoint = folder/'best'
            else:
                folder = root/f'data/processed/transformers/{model}/seed_{seed}/full/standard'
                meta = load_transformer_run(folder, require_full=True)
                protocol = read(folder/'final_protocol.json')
                validate_protocol(folder, protocol, labels)
                checkpoint = folder/'checkpoint'
            protocol_path=folder/'final_protocol.json'
            artifact=checkpoint/'model.safetensors';digest=sha256(artifact)
            predictor=TransformerDemo.from_checked_checkpoint(checkpoint,meta,'cpu')
            points=[]
            for row,text in zip(rows,texts):
                n=len(predictor.tokenizer(text,add_special_tokens=True)['input_ids'])
                if n>predictor.max_length:truncations.append(row['id'])
                inputs=predictor.tokenizer(text,truncation=True,max_length=predictor.max_length,return_tensors='pt')
                with torch.inference_mode():
                    scores=torch.sigmoid(predictor.model(**inputs).logits.float())[0].numpy()
                points.append(scores)
            points=np.asarray(points)
            limits=next(c['thresholds'] for c in protocol['configurations'] if c['threshold_mode']=='tuned')
            params={k:meta['config'][k] for k in ('learning_rate','epochs','batch_size','max_length')}
            params.update(selected_epoch=meta.get('best_epoch',meta.get('selected_epoch')),
                          n_parameters=sum(p.numel() for p in predictor.model.parameters()))
        if predictor.labels!=labels:
            raise ValueError('Mapping mô hình khác dữ liệu')
        if sha256(artifact)!=digest:
            raise ValueError('Trọng số đã thay đổi trong suy luận')
        records.append(dict(model=model,seed=seed,started_at_utc=started,model_sha256=digest,
            protocol_sha256=sha256(protocol_path),parameters=params,truncated_ids=truncations,
            scores=np.asarray(points,dtype=float).tolist(),thresholds={'fixed':.5,'tuned':limits},
            weights_unchanged=True))
        del predictor
        gc.collect()
    saved=dict(fixture_sha256=sha256(FIXTURE),ids=[r['id'] for r in rows],labels=labels,runs=records)
    write(OUTPUT/'scores.json',saved)
    metrics,predictions=summarize(rows,labels,records)
    write(OUTPUT/'metrics.json',metrics);write(OUTPUT/'predictions.json',predictions)
    pd.DataFrame(metrics).to_csv(OUTPUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(OUTPUT/'predictions.csv',index=False,encoding='utf-8-sig')
    write(OUTPUT/'linear_explanation.json',explanation)
    write(OUTPUT/'receipt.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        fixture_sha256=sha256(FIXTURE),plan_sha256=sha256(OUTPUT/'plan.json'),
        runner_sha256=sha256(Path(__file__)),boundary=boundary,source_verified=args.verify_source,
        n_external=20,n_controlled=10,runs=12,score_rows=360,threshold_decisions=720,
        no_training=True,no_threshold_selection=True,device='cpu',dtype='float32',
        reference_status=fixture['human_review_status'],
        limitations=['Purposive small sample; AI-proposed references are unreviewed.',
                     'Source sentiment labels are not 28-class emotion gold labels.',
                     'Near duplicates and pretraining overlap are not ruled out.',
                     'External and controlled results are reported separately; no population or overfitting claim.']))
    verify_saved()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root',type=Path,default=ROOT)
    p.add_argument('--verify-source',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if args.verify_only:verify_saved()
    else:run(args)


if __name__=='__main__':
    main()
