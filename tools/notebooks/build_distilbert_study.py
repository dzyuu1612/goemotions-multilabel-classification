"""Tạo notebook C3 của báo cáo 2.1; Run All kiểm bundle RTX 5060 đã chạy."""
from src.paths import ROOT
import nbformat as nbf


def main():
    cells = []
    cells.append(nbf.v4.new_markdown_cell(r"""# C3 DistilBERT — thiết kế của Nhật Huy, chạy trên RTX 5060
Ba lượt full seed 42, 123, 2026 đã được train từ pretrained ngày 21/10/2026 trên máy
RTX 5060 của Duy. Đây là bộ C3 dùng trong báo cáo 2.1, thay bộ C3 lịch sử trong bảng chính.
Không gộp thêm kết quả RTX 3050 thành sáu seed. Run All kiểm artifact và suy luận lại
validation/test từ checkpoint đã có; code và lệnh train được trình bày riêng."""))
    cells.append(nbf.v4.new_code_cell(r"""from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/current_report.py').is_file())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.current_report import current_c3_root, verify_current_report, checked_c3, replay_current_c3
from src.notebook_support import artifacts_root
from src.datasets.goemotions import multi_hot, sha256, REVISION
from src.evaluation.protocols import read_json
from src.evaluation.metrics import evaluate_multilabel
from src.models.transformer import METRIC_NAMES
from scripts.analysis.summarize_project import summarize
C3_RUN, ARTIFACTS = current_c3_root(), artifacts_root()
verified, frames, names = verify_current_report(str(ARTIFACTS), str(C3_RUN))
snapshot = verified['report_snapshot']
completion = read_json(C3_RUN/'completion.json')
torch.set_num_threads(4)
print('Python:', sys.version.split()[0], '| torch:', torch.__version__)
print('Thiết bị replay:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Revision dữ liệu:', REVISION)"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 1. Thiết lập đã khóa trước khi train
Official split: 43.410 train / 5.426 validation / 5.427 test; 28 nhãn.
Checkpoint chọn theo Macro-F1 validation @0,5; ngưỡng riêng cũng chỉ chọn trên validation.
Muốn train bộ mới, dùng `python -m scripts.distilbert.reproduce_c3 --output reports/c3_new_run`.
Đường dẫn mới giúp giữ nguyên checkpoint và kết quả của báo cáo đã nộp."""))
    cells.append(nbf.v4.new_code_cell(r"""plan = read_json(C3_RUN/'training_plan.json')['plan']
assert plan['config'] == read_json(ROOT/'configs/distilbert_full.json')
assert plan['seeds'] == [42,123,2026] and not plan['test_used_for_selection']
display(pd.Series(plan['config']))
display(pd.DataFrame([dict(split=s, rows=len(df), unique_ids=df.id.nunique()) for s,df in frames.items()]))
display(pd.Series(completion['environment']))"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 2. Kiểm checkpoint, hash và protocol của đủ ba seed"""))
    cells.append(nbf.v4.new_code_cell(r"""receipts, runs = [], {}
for seed in (42,123,2026):
    meta, protocol = checked_c3(C3_RUN, seed)
    runs[seed] = meta
    assert meta['n_train'] == 43410 and meta['n_validation'] == 5426 and not meta['test_used']
    receipts.append(dict(seed=seed, best_epoch=meta['best_epoch'], gpu=meta['environment']['gpu'],
        n_train=meta['n_train'], n_validation=meta['n_validation'],
        model_sha256=meta['artifact_sha256']['best/model.safetensors'],
        checkpoint_selection='validation Macro-F1 @0.5'))
display(pd.DataFrame(receipts))
print('PASS: cả ba bundle mới có thật, đủ hash; không thay bằng checkpoint RTX 3050.')"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Loss và Macro-F1 validation qua ba epoch đã chạy"""))
    cells.append(nbf.v4.new_code_cell(r"""import matplotlib.pyplot as plt
history = pd.read_csv(C3_RUN/'history.csv')
display(history)
fig, axes = plt.subplots(1,2,figsize=(10,3.5))
for seed, group in history.groupby('seed'):
    axes[0].plot(group.epoch, group.train_loss, marker='o', label=f'seed {seed}')
    axes[1].plot(group.epoch, group.validation_macro_f1, marker='o', label=f'seed {seed}')
axes[0].set_title('Train loss — RTX 5060'); axes[1].set_title('Validation Macro-F1 @0.5')
for ax in axes: ax.set_xlabel('Epoch đã chạy'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()
print('Ba seed chọn epoch 3; chưa có đối chứng cho kết luận đã hội tụ hoặc tối ưu siêu tham số.')"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 4. Vòng lặp train và số cập nhật thực tế
Cell hiển thị implementation đã tạo bộ C3 này. FP16 có thể bỏ một bước khi gradient
không hữu hạn; vì vậy số bước optimizer thực tế được đối chiếu riêng với số dự kiến."""))
    cells.append(nbf.v4.new_code_cell(r"""import inspect
from IPython.display import Code
from scripts.distilbert import train_distilbert
display(Code(inspect.getsource(train_distilbert.main), language='python'))
steps = []
for seed, meta in runs.items():
    assert meta['optimizer_steps'] + len(meta['skipped_amp_steps']) == meta['planned_optimizer_steps']
    steps.append(dict(seed=seed, optimizer_updates=meta['optimizer_steps'],
        planned_updates=meta['planned_optimizer_steps'], skipped_amp_updates=len(meta['skipped_amp_steps']),
        training_seconds=meta['fit_and_validation_seconds']))
display(pd.DataFrame(steps))"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 5. Suy luận mới toàn bộ validation và test
Nạp lại từng checkpoint đã khóa, chạy model thật rồi đối chiếu scores/metric với
artifact báo cáo. Không tìm lại ngưỡng; cell dừng nếu scores hoặc metric không khớp."""))
    cells.append(nbf.v4.new_code_cell(r"""fresh_parts = []
for seed in (42,123,2026):
    for split in ('validation','test'):
        fresh_parts.append(replay_current_c3(C3_RUN, seed, frames, names, split))
        print('PASS real inference:', seed, split, len(frames[split]))
fresh = pd.concat(fresh_parts, ignore_index=True)
assert len(fresh) == 18
display(fresh)"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 6. Mean và sample std của đủ ba seed — đối chiếu báo cáo 2.1"""))
    cells.append(nbf.v4.new_code_cell(r"""averages = summarize(fresh.to_dict('records'))
for row in averages:
    expected = next(r for r in snapshot['mean_std'] if r['system'] == 'C_distilbert'
                    and r['split'] == row['split'] and r['threshold_mode'] == row['threshold_mode'])
    for metric in METRIC_NAMES:
        for suffix in ('_mean','_std'):
            assert abs(row[metric+suffix] - expected[metric+suffix]) <= 1e-12
display(pd.DataFrame(averages)[['split','threshold_mode','n_runs','macro_f1_mean','macro_f1_std',
                              'micro_f1_mean','micro_f1_std','hamming_loss_mean']])
print('PASS: sáu nhóm mean ± sample std (ddof=1) khớp bộ số báo cáo 2.1.')"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 7. Ngưỡng và năm nhãn hiếm
Nhãn hiếm chọn theo support trên train. Điểm validation sau chọn ngưỡng có thể lạc quan;
bảng test dưới được tính lại từ scores mới, không dùng test thay ngưỡng."""))
    cells.append(nbf.v4.new_code_cell(r"""from src.models.baseline import load_aligned_scores
support = multi_hot(frames['train'].labels.tolist()).sum(axis=0)
rare_ids = np.argsort(support)[:5].tolist()
threshold_rows, rare_rows = [], []
for seed in (42,123,2026):
    _, protocol = checked_c3(C3_RUN, seed)
    scores = load_aligned_scores(ROOT/f'reports/notebook_replay/C_distilbert_{seed}_test.npz', frames['test'].id, names)
    truth = multi_hot(frames['test'].labels.tolist())
    for choice in protocol['configurations']:
        metric = evaluate_multilabel(truth, scores, names, choice['thresholds'])
        rare_rows.extend(dict(seed=seed, mode=choice['threshold_mode'], **metric['per_label'][j]) for j in rare_ids)
        if choice['threshold_mode'] == 'tuned':
            threshold_rows.extend(dict(seed=seed,label=label,threshold=value) for label,value in zip(names,choice['thresholds']))
display(pd.DataFrame(threshold_rows).pivot(index='label',columns='seed',values='threshold').reindex(names))
display(pd.DataFrame(rare_rows).groupby(['label','mode']).f1.agg(['mean','std']))"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 8. Phân tích lỗi và các ví dụ trong báo cáo — kiểm bằng scores vừa suy luận"""))
    cells.append(nbf.v4.new_code_cell(r"""from scripts.analysis.analyze_project_errors import analyze_arrays
representative = snapshot['representatives']['distilbert']
seed = representative['seed']
scores = load_aligned_scores(ROOT/f'reports/notebook_replay/C_distilbert_{seed}_test.npz', frames['test'].id, names)
counts, pairs, _, _ = analyze_arrays(frames['test'], names, rare_ids, support,
    {'distilbert':representative}, {'distilbert':scores}, {'distilbert':0.5})
for row in counts:
    expected = next(r for r in snapshot['error_counts'] if r['architecture']=='distilbert' and r['category']==row['category'])
    assert row == expected
display(pd.DataFrame(counts))
examples = []
for example in (e for e in snapshot['report_examples'] if e['architecture']=='distilbert'):
    index = int(np.flatnonzero(frames['test'].id.to_numpy()==example['id'])[0])
    predicted = [names[j] for j in np.flatnonzero(scores[index]>=0.5)]
    assert predicted == example['predicted_labels'] and frames['test'].text.iloc[index] == example['text']
    examples.append({k:v for k,v in example.items() if k!='scores'})
display(pd.DataFrame(examples))
print('PASS: ba nhóm lỗi và ba ví dụ C3 khớp nội dung báo cáo.')"""))
    cells.append(nbf.v4.new_markdown_cell(r"""## 9. Thử dự đoán C3 và xác nhận mô hình demo cuối
Hàm dưới thực sự nạp C3 seed đại diện và trả 28 điểm cho câu nhập.
Demo Gradio chính của nhóm vẫn dùng BERT seed 123, chọn bằng validation; bảng này
chỉ minh họa checkpoint C3 vừa được kiểm, không đổi mô hình demo theo điểm test."""))
    cells.append(nbf.v4.new_code_cell(r"""from src.models.distilbert_study import load_bundle, predict_texts
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bundle = load_bundle(C3_RUN/f'seed_{seed}', device)
predictions = predict_texts(bundle, ['Thank you, I am so happy!', 'Why did this happen?'])
display(pd.DataFrame([{k:v for k,v in item.items() if k!='scores'} for item in predictions]))
display(pd.DataFrame([item['scores'] for item in predictions]).T)
assert all(len(item['scores']) == 28 for item in predictions)
del bundle
if device == 'cuda': torch.cuda.empty_cache()
print('Demo chính:', verified['selection']['architecture'], 'seed', verified['selection']['seed'])
print('Hoàn tất: 3 checkpoint C3 mới, 6 lượt suy luận full, 18 dòng metric; không train thêm trong Run All.')"""))
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {'goemotions': {'scope': 'Report 2.1 C3: three completed full RTX 5060 training runs; fresh validation/test inference, metrics, rare labels, errors and prediction; no retraining inside Run All'}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.13.9'}}
    nbf.write(notebook, ROOT/'notebooks/05_distilbert_huy.ipynb')


if __name__ == "__main__":
    main()
