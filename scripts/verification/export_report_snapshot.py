"""Công bố bảng/hình và biên nhận báo cáo 2.1; không đưa trọng số lớn vào Git."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from src.current_report import current_c3_root, verify_current_report
from src.datasets.goemotions import REVISION, sha256
from src.notebook_support import artifacts_root
from src.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/report_2_1')
    args = parser.parse_args()
    historical = (args.artifacts_root or artifacts_root()).resolve()
    c3 = (args.run_root or current_c3_root()).resolve()
    result, frames, labels = verify_current_report(str(historical), str(c3))
    verification = json.loads((c3 / 'report_data/word_verification.json').read_text(encoding='utf-8'))
    if not verification.get('passed') or verification.get('sha256') != sha256(args.report):
        raise ValueError('Cần kiểm bản Word hiện tại bằng verify_current_c3_report trước khi công bố.')
    data = result['report_snapshot']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    exported = []

    def copy(source, relative):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        exported.append(relative)

    def write(relative, value):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        exported.append(relative)

    for name in ('all_runs.csv', 'mean_std.csv', 'per_label_test.csv', 'rare_labels_test.csv',
                 'error_counts.csv', 'error_pairs.csv', 'top_error_pairs.csv',
                 'learning_curves.csv', 'learning_curves.png', 'test_f1.png'):
        copy(c3 / 'report_data' / name, name)
    write('report_snapshot.json', dict(data_revision=REVISION, label_names=labels,
          sizes={split: len(frame) for split, frame in frames.items()},
          records=result['records'], averages=result['averages'],
          representatives={name: {key: value for key, value in item.items() if key != 'run_dir'}
                           for name, item in data['representatives'].items()},
          rare_labels=data['rare_labels'], error_counts=data['error_counts'],
          report_examples=data['report_examples'], top_error_pairs=data['top_pairs']))
    write('word_verification.json', {key: value for key, value in verification.items() if key != 'path'})
    demo_path = ROOT / 'reports/demo_verification.json'
    demo = json.loads(demo_path.read_text(encoding='utf-8'))
    if (demo.get('model_inference_status') != 'PASS'
            or (demo['architecture'], demo['seed']) != (result['selection']['architecture'], result['selection']['seed'])
            or demo['selection_sha256'] != sha256(historical / 'data/processed/transformers/selected_model.json')
            or not all(demo['input_checks'].values())):
        raise ValueError('Biên nhận demo không khớp mô hình đã chọn hoặc phép kiểm đầu vào chưa đạt.')
    copy(demo_path, 'demo_verification.json')
    for architecture in ('bert', 'roberta', 'distilbert'):
        for seed in (42, 123, 2026):
            folder = (c3 / f'seed_{seed}' if architecture == 'distilbert' else
                      historical / f'data/processed/transformers/{architecture}/seed_{seed}/full/standard')
            metadata = 'run.json' if architecture == 'distilbert' else 'run_metadata.json'
            for name in (metadata, 'final_protocol.json', 'test_results.json'):
                copy(folder / name, f'receipts/{architecture}/seed_{seed}/{name}')
    for name in ('training_plan.json', 'completion.json', 'selection.json', 'history.csv'):
        copy(c3 / name, 'receipts/distilbert/' + name)
    manifest = dict(schema_version=1, exported_at_utc=datetime.now(timezone.utc).isoformat(),
        report_filename=args.report.name, report_sha256=sha256(args.report),
        source_report_data_sha256=sha256(c3 / 'report_data/report_data.json'),
        scope='Report 2.1: checked A/B results; C1/C2 RTX 5060 runs from 2026-10-08; C3 RTX 5060 runs from 2026-10-21. No RTX 3050 results in the main comparison.',
        records=72, summary_groups=36, hash_policy='sha256:raw-bytes; results/report_2_1/** -text',
        model_weights_in_git=False,
        files_sha256={name: sha256(output / name) for name in sorted(exported)})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASS: exported {len(exported)} verified result/receipt files; 72 records / 36 groups.')


if __name__ == '__main__':
    main()
