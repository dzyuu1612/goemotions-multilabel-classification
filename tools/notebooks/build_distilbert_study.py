"""Build C3 notebook from the same verified nine-run benchmark as C1/C2."""
import nbformat as nbf
from src.paths import ROOT


def main():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# C3 DistilBERT — phần Nhật Huy trong lượt chạy chung
Ba seed 42, 123, 2026 thuộc pipeline cục bộ A/B/C. Notebook kiểm config/hash,
suy luận thật validation/test và tổng hợp kết quả. Không thay bằng bộ C3 riêng
ngày 08/10 hoặc bảng báo cáo 2.1; không huấn luyện thêm khi Run All.''')
    code('''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/notebook_support.py').is_file())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.notebook_support import artifacts_root, benchmark, replay_split
from src.datasets.goemotions import multi_hot, sha256, REVISION
from src.evaluation.protocols import read_json, validate_protocol
from src.evaluation.metrics import evaluate_multilabel
from src.models.transformer import METRIC_NAMES, run_folder, load_transformer_run
from scripts.analysis.summarize_project import summarize
ARTIFACTS = artifacts_root()
verified, frames, names = benchmark(str(ARTIFACTS))
SEEDS = (42, 123, 2026)
torch.set_num_threads(4)
print('Python:', sys.version.split()[0], '| torch:', torch.__version__)
print('Thiết bị suy luận:', 'CUDA' if torch.cuda.is_available() else 'CPU')
print('Revision dữ liệu:', REVISION)''')
    md('''## 1. Cấu hình và ba checkpoint full
Mỗi seed bắt đầu từ DistilBERT pretrained, học 28 logits bằng BCE và chọn epoch bằng
Macro-F1 validation tại 0,5. Muốn tạo checkpoint: chạy
`python -m scripts.transformers.run_transformer_seeds --architectures distilbert --device cuda --resume`.
Lệnh train được chạy riêng; cell dưới chỉ kiểm các run đã hoàn tất.''')
    code('''runs, receipts = {}, []
for seed in SEEDS:
    folder = run_folder(ARTIFACTS, 'distilbert', seed)
    meta = load_transformer_run(folder, require_full=True)
    assert meta['architecture'] == 'distilbert' and meta['seed'] == seed
    protocol = read_json(folder/'final_protocol.json')
    validate_protocol(folder, protocol, names)
    runs[seed] = meta
    receipts.append(dict(seed=seed, selected_epoch=meta['selected_epoch'],
        n_train=meta['sizes']['train'], n_validation=meta['sizes']['validation'],
        model_sha256=meta['artifact_sha256']['checkpoint/model.safetensors'],
        trained_at_utc=meta['created_at_utc'], completed_at_utc=meta['completed_at_utc'],
        training_seconds=meta['elapsed_seconds']))
assert all(r['config'] == runs[42]['config'] for r in runs.values())
display(pd.Series(runs[42]['config']))
display(pd.DataFrame(receipts))
display(pd.Series(runs[42]['environment']))''')
    md('## 2. Loss và chỉ số validation qua các epoch thực tế')
    code('''import matplotlib.pyplot as plt
history = pd.DataFrame([dict(seed=seed, selected_epoch=meta['selected_epoch'], **row)
                       for seed, meta in runs.items() for row in meta['history']])
display(history)
fig, axes = plt.subplots(1,2,figsize=(11,3.5))
for seed, group in history.groupby('seed'):
    axes[0].plot(group.epoch, group.train_loss, marker='o', label=f'seed {seed}')
    axes[1].plot(group.epoch, group.macro_f1, marker='o', label=f'seed {seed}')
axes[0].set_title('Train loss'); axes[1].set_title('Validation Macro-F1 @0.5')
for ax in axes: ax.set_xlabel('Epoch'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()
print('Epoch tốt nhất được ghi theo từng seed; chưa khẳng định siêu tham số tối ưu.')''')
    md('''## 3. Theo dõi tokenizer và BCE bằng một batch train thật
CLS → encoder 6 block → pre_classifier → ReLU → dropout → classifier 28 logits.
Distillation thuộc pretraining; nhóm fine-tune bằng BCE. Cell không gọi backward.''')
    code('''from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scripts.transformers.train_transformer import encoded_dataset, trim_padding_collate
folder = run_folder(ARTIFACTS, 'distilbert', 42)
tokenizer = AutoTokenizer.from_pretrained(folder/'checkpoint', local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(folder/'checkpoint', local_files_only=True).eval()
data, _ = encoded_dataset(frames['train'].head(4), tokenizer, names, runs[42]['config']['max_length'])
batch = trim_padding_collate([data[i] for i in range(4)])
with torch.inference_mode():
    out = model(**batch)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(out.logits, batch['labels'])
torch.testing.assert_close(out.loss, expected)
display(pd.Series(dict(input_shape=str(tuple(batch['input_ids'].shape)),
    labels_shape=str(tuple(batch['labels'].shape)), logits_shape=str(tuple(out.logits.shape)), bce=float(out.loss))))
display(pd.DataFrame({'token':tokenizer.convert_ids_to_tokens(batch['input_ids'][0]),
                      'id':batch['input_ids'][0].tolist()}))
del model, data, batch, out''')
    md('## 4. Code huấn luyện dùng cho lượt chạy này')
    code('''import inspect
from IPython.display import Code
from scripts.transformers import train_transformer
display(Code(inspect.getsource(train_transformer.main), language='python'))''')
    md('''## 5. Suy luận lại cả validation và test từ ba checkpoint
Mỗi split chạy đủ mẫu, giữ nguyên ngưỡng đã khóa. Dừng nếu metric không khớp
artifact. Không lấy số từ bảng xuất sẵn để giả làm kết quả suy luận mới.''')
    code('''parts = []
for seed in SEEDS:
    for split in ('validation','test'):
        parts.append(replay_split(ARTIFACTS, 'C_distilbert', seed, split))
        print('PASS real inference:', seed, split, len(frames[split]))
fresh = pd.concat(parts, ignore_index=True)
assert len(fresh) == 18
display(fresh)''')
    md('## 6. Mean và sample std qua ba seed')
    code('''averages = summarize(fresh.to_dict('records'))
for row in averages:
    expected = next(r for r in verified['averages'] if r['system']=='C_distilbert'
                    and r['split']==row['split'] and r['threshold_mode']==row['threshold_mode'])
    for metric in METRIC_NAMES:
        for suffix in ('_mean','_std'):
            assert abs(row[metric+suffix]-expected[metric+suffix]) <= 1e-12
display(pd.DataFrame(averages)[['split','threshold_mode','n_runs','macro_f1_mean','macro_f1_std',
                              'micro_f1_mean','micro_f1_std','hamming_loss_mean']])
print('PASS: sáu nhóm mean ± sample std (ddof=1) từ ba seed thật.')''')
    md('''## 7. Ngưỡng riêng và năm nhãn hiếm
Chọn năm nhãn có support train thấp nhất. So fixed/global/tuned trên cùng model;
hiển thị cả nhãn tăng và giảm điểm, không có thí nghiệm weighted BCE ở đây.''')
    code('''from src.models.baseline import load_aligned_scores
support = multi_hot(frames['train'].labels.tolist()).sum(axis=0)
rare_ids = np.argsort(support)[:5].tolist()
threshold_rows, rare_rows = [], []
for seed in SEEDS:
    folder = run_folder(ARTIFACTS, 'distilbert', seed)
    protocol = read_json(folder/'final_protocol.json')
    scores = load_aligned_scores(ROOT/f'reports/notebook_replay/C_distilbert_{seed}_test.npz', frames['test'].id, names)
    truth = multi_hot(frames['test'].labels.tolist())
    for choice in protocol['configurations']:
        metric = evaluate_multilabel(truth, scores, names, choice['thresholds'])
        rare_rows.extend(dict(seed=seed, mode=choice['threshold_mode'], **metric['per_label'][j]) for j in rare_ids)
        if choice['threshold_mode']=='tuned':
            threshold_rows.extend(dict(seed=seed,label=label,threshold=value) for label,value in zip(names,choice['thresholds']))
display(pd.DataFrame(threshold_rows).pivot(index='label',columns='seed',values='threshold').reindex(names))
display(pd.DataFrame(rare_rows).groupby(['label','mode']).f1.agg(['mean','std']))''')
    md('''## 8. Ba nhóm lỗi và ví dụ có ID
Seed đại diện chọn bằng validation; fixed 0,5. Các nhóm có thể chồng lấp:
thiếu một phần nhãn, bỏ sót nhãn hiếm, vừa bỏ sót vừa dự đoán thừa.''')
    code('''from scripts.analysis.analyze_project_errors import analyze_arrays
seed = max(SEEDS, key=lambda s:(runs[s]['metrics']['macro_f1'], -s))
scores = load_aligned_scores(ROOT/f'reports/notebook_replay/C_distilbert_{seed}_test.npz', frames['test'].id, names)
counts, pairs, rare, examples = analyze_arrays(frames['test'], names, rare_ids, support,
    {'distilbert':runs[seed]}, {'distilbert':scores}, {'distilbert':0.5})
display(pd.DataFrame(counts))
display(pd.DataFrame(examples))
print('Seed đại diện C3 theo validation:', seed)''')
    md('''## 9. Dự đoán câu nhập và phân biệt với demo chung
Dùng checkpoint C3 đại diện để thử câu. Demo chung chọn kiến trúc theo mean validation
Macro-F1 của cả ba C; kết quả chọn được đọc từ selected_model, không hard-code BERT123.''')
    code('''from src.apps.demo import TransformerDemo
folder = run_folder(ARTIFACTS, 'distilbert', seed)
predictor = TransformerDemo.from_checked_checkpoint(folder/'checkpoint', runs[seed], 'auto')
for text in ['Thank you, I am so happy!', 'Why did this happen?']:
    message, rows = predictor.predict(text)
    print(text, message)
    assert len(rows) == 28
    display(pd.DataFrame(rows, columns=['label','score','threshold','selected']))
del predictor
if torch.cuda.is_available(): torch.cuda.empty_cache()
print('Demo chính:', verified['selection']['architecture'], 'seed', verified['selection']['seed'])
print('Hoàn tất: 3 checkpoint C3 full, 6 lượt suy luận, 18 dòng metric; không train trong Run All.')''')
    n = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
        'goemotions':{'scope':'Local full benchmark C3: three seeds, six real inference passes, no training in Run All'}})
    nbf.validate(n)
    nbf.write(n, ROOT/'notebooks/05_distilbert_huy.ipynb')


if __name__ == '__main__':
    main()
