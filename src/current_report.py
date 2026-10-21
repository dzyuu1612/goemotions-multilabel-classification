"""Kiểm bộ số báo cáo 2.1: A/B/C1/C2 đã lưu và C3 chạy lại trên RTX 5060.

Không thay artifact huấn luyện hoặc tìm lại ngưỡng. Thiếu bundle C3 mới sẽ dừng,
không âm thầm lấy DistilBERT lịch sử thay cho bộ kết quả hiện tại.
"""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.distilbert.reproduce_c3 import SEEDS, SOURCES, checked_run, require
from scripts.verification.reproduce_report import compare_averages, compare_metrics, record_key
from src.datasets.goemotions import REVISION, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel
from src.evaluation.protocols import read_json
from src.models.baseline import load_aligned_scores
from src.models.transformer import METRIC_NAMES
from src.notebook_support import benchmark
from src.paths import ROOT


def current_c3_root():
    configured = os.environ.get('GOEMOTIONS_C3_RUN_ROOT')
    folder = Path(configured).expanduser() if configured else ROOT / 'reports/c3_current_environment_2026-10-21'
    folder = folder.resolve()
    if not (folder / 'completion.json').is_file():
        raise FileNotFoundError(
            'Thiếu bộ C3 dùng cho báo cáo 2.1. Đặt GOEMOTIONS_C3_RUN_ROOT tới thư mục '
            'có completion.json và seed_42/seed_123/seed_2026; không dùng bộ C3 cũ thay thế.')
    return folder


def checked_c3(folder, seed):
    """Ràng buộc checkpoint, scores và protocol với lượt huấn luyện full đã khóa."""
    folder = Path(folder)
    require(seed in SEEDS, 'C3 chỉ dùng ba seed đã khai báo')
    plan = read_json(folder / 'training_plan.json')['plan']
    require(plan['data_revision'] == REVISION and plan['seeds'] == list(SEEDS), 'Sai dataset/seed C3')
    require(plan['source_sha256'] == {name: sha256(ROOT / name) for name in SOURCES},
            'Source huấn luyện khác nguồn đã khóa')
    config = dict(plan['config'], seed=seed)
    run = folder / f'seed_{seed}'
    metadata = checked_run(run, config, plan['source_sha256'])
    protocol = read_json(run / 'final_protocol.json')
    require(protocol['seed'] == seed and protocol['data_revision'] == REVISION
            and protocol['label_names'] == metadata['label_names']
            and protocol['test_used_for_selection'] is False, 'Sai protocol C3')
    for key, name in [('run_metadata_sha256', 'run.json'),
                      ('model_sha256', 'best/model.safetensors'),
                      ('validation_scores_sha256', 'validation_scores.npz')]:
        require(protocol[key] == sha256(run / name), f'Artifact khác protocol C3: {name}')
    modes = [c['threshold_mode'] for c in protocol['configurations']]
    require(modes == ['fixed', 'global', 'tuned'], 'Thiếu/lặp chế độ ngưỡng C3')
    receipt = read_json(run / 'test_results.json')
    require(receipt['n_samples'] == 5427
            and receipt['protocol_sha256'] == sha256(run / 'final_protocol.json')
            and receipt['scores_sha256'] == sha256(run / 'test_scores.npz'), 'Sai biên nhận test C3')
    return metadata, protocol


def compare_report_records(actual, expected):
    """Không cho trộn thêm bộ C3 cũ, bỏ seed hoặc giữ nguyên bảng điểm đã thay."""
    by_key = {record_key(row): row for row in expected}
    require(len(actual) == len(expected) == len(by_key) == 72, 'Cần đúng 72 dòng báo cáo')
    require(len({record_key(row) for row in actual}) == 72, 'Trùng cấu hình/seed/split')
    for row in actual:
        other = by_key.get(record_key(row))
        require(other is not None and row['n_samples'] == other['n_samples'], 'Sai dòng/số mẫu báo cáo')
        compare_metrics(row, other, record_key(row))


@lru_cache(maxsize=2)
def verify_current_report(artifact_root, run_root):
    """Tính lại 72 dòng và 36 nhóm, giữ rõ nguồn của từng kiến trúc."""
    old, frames, names = benchmark(str(Path(artifact_root).resolve()))
    folder = Path(run_root).resolve()
    complete = read_json(folder / 'completion.json')
    require(complete['training_repeated_from_pretrained'] is True
            and complete['seeds'] == list(SEEDS) and complete['test_used_for_selection'] is False
            and (complete['n_train'], complete['n_validation'], complete['n_test']) == (43410, 5426, 5427),
            'C3 phải là ba lượt full đã hoàn thành')
    require(complete['training_plan_sha256'] == sha256(folder / 'training_plan.json'), 'Plan C3 thay đổi')
    for name, digest in complete['outputs_sha256'].items():
        require(sha256(folder / name) == digest, f'Output C3 thay đổi: {name}')
    snapshot = read_json(folder / 'report_data/report_data.json')
    require(snapshot['source_hashes']['new_completion'] == sha256(folder / 'completion.json'),
            'Bộ số báo cáo không thuộc lượt C3 này')
    records = [row for row in old['records'] if row['system'] != 'C_distilbert']
    histories = [row for row in old['histories'] if row['architecture'] != 'distilbert']
    for seed in SEEDS:
        meta, protocol = checked_c3(folder, seed)
        require(meta['label_names'] == names, 'Sai thứ tự nhãn C3')
        history = [dict(row, macro_f1=row['validation_macro_f1'], micro_f1=row['validation_micro_f1'])
                   for row in read_json(folder / f'seed_{seed}/history.json')]
        config = dict(meta['config'], epochs=meta['config']['epochs'],
                      model_revision=meta['config']['revision'])
        histories.append(dict(architecture='distilbert', seed=seed, selected_epoch=meta['best_epoch'],
                              model_revision=meta['config']['revision'], config=config, history=history))
        for split in ('validation', 'test'):
            scores = load_aligned_scores(folder / f'seed_{seed}/{split}_scores.npz', frames[split].id, names)
            truth = multi_hot(frames[split].labels.tolist(), len(names))
            for choice in protocol['configurations']:
                metrics = evaluate_multilabel(truth, scores, names, choice['thresholds'])
                records.append(dict(system='C_distilbert', seed=seed, split=split,
                    threshold_mode=choice['threshold_mode'], n_samples=len(truth),
                    **{key: metrics[key] for key in METRIC_NAMES}))
    compare_report_records(records, snapshot['all_runs'])
    averages = compare_averages(records, snapshot['mean_std'])
    ranked = sorted((r for r in averages if r['system'].startswith('C_') and r['split'] == 'validation'
                     and r['threshold_mode'] == 'fixed'), key=lambda r: -r['macro_f1_mean'])
    require(ranked[0]['system'] == 'C_' + old['selection']['architecture'], 'Lựa chọn demo đã thay đổi')
    return dict(records=records, averages=averages, histories=histories,
                selection=old['selection'], report_snapshot=snapshot), frames, names


def replay_current_c3(folder, seed, frames, names, split='test'):
    """Suy luận mới trên toàn bộ split bằng checkpoint C3 RTX 5060 của báo cáo."""
    import torch
    from src.models.distilbert_study import EmotionDataset, load_bundle, make_loader, score_loader

    require(split in ('validation', 'test'), 'Chỉ replay validation hoặc test')
    folder = Path(folder)
    meta, protocol = checked_c3(folder, seed)
    require(names == meta['label_names'], 'Sai thứ tự nhãn replay C3')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    run = folder / f'seed_{seed}'
    bundle = load_bundle(run, device)
    dataset = EmotionDataset(frames[split], bundle['tokenizer'], meta['config']['max_length'])
    loader = make_loader(dataset, bundle['tokenizer'], meta['config']['eval_batch_size'])
    scores = score_loader(bundle['model'], loader, device)
    previous = load_aligned_scores(run / f'{split}_scores.npz', frames[split].id, names)
    np.testing.assert_allclose(scores, previous, rtol=0, atol=1e-6)
    truth = multi_hot(frames[split].labels.tolist(), len(names))
    rows = []
    for choice in protocol['configurations']:
        actual = evaluate_multilabel(truth, scores, names, choice['thresholds'])
        expected = evaluate_multilabel(truth, previous, names, choice['thresholds'])
        difference = compare_metrics(actual, expected, f'C3/{seed}/{split}/{choice["threshold_mode"]}')
        rows.append(dict(system='C_distilbert', seed=seed, split=split,
                         threshold_mode=choice['threshold_mode'], n_samples=len(truth),
                         **{key: actual[key] for key in METRIC_NAMES},
                         score_max_abs_diff=float(np.max(np.abs(scores - previous))),
                         metric_max_abs_diff=difference, device=device))
    output = ROOT / 'reports/notebook_replay'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'C_distilbert_{seed}_{split}.npz', ids=frames[split].id.to_numpy(dtype=str),
                        scores=scores, label_names=np.asarray(names, dtype=str))
    del bundle, dataset, loader
    if device == 'cuda':
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)
