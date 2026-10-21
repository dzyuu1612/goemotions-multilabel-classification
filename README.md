# GoEmotions · Multi-label Emotion Classification

**Một bình luận. Nhiều cảm xúc. Một quy trình thực nghiệm có thể kiểm lại.**

![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![28 labels](https://img.shields.io/badge/GoEmotions-28%20labels-0F766E)
![NLP](https://img.shields.io/badge/NLP-Multi--label-7C3AED)

[Chạy thử](#quick-start) · [Baseline của Duy](#baseline) · [Notebook](#notebooks) · [Thực nghiệm](#experiments) · [Demo](#demo) · [Đọc code](#code-map)

## Kết luận kiểm tra fine-tune và số báo cáo

**Đã fine-tune ba C, chưa có bằng chứng đã tìm được cấu hình tốt nhất.**
Mỗi kiến trúc chạy một cấu hình với ba seed 42/123/2026. Đây là kiểm độ ổn định,
chưa phải tìm learning rate, epoch hoặc batch size. Cả 9 run đều chọn epoch cuối
(BERT 4, RoBERTa/DistilBERT 3); nên thử thêm trên validation trước khi kết luận đã hội tụ.
Không suy ra chắc chắn thêm epoch sẽ tốt hơn.

Số **test**, ba C là mean ± sample std của ba seed; ngưỡng riêng đã khóa từ validation:

| Hệ thống | Macro-F1 | Micro-F1 |
|---|---:|---:|
| A balanced, tuned | 0.4493 | 0.5277 |
| B BART-MNLI, tuned | 0.1609 | 0.1752 |
| C1 BERT, tuned | 0.5038 ± 0.0097 | 0.5900 ± 0.0029 |
| C2 RoBERTa, tuned | 0.4872 ± 0.0156 | 0.6048 ± 0.0057 |
| C3 DistilBERT, tuned | 0.4866 ± 0.0093 | 0.5935 ± 0.0047 |

BERT thắng theo Macro-F1; RoBERTa có Micro-F1 tuned cao hơn.
Demo vẫn dùng **BERT seed 123**, đã chọn trước bằng **validation Macro-F1 @0.5**,
không đổi người thắng theo test. B không fine-tune; A chưa tìm C/ngram tốt nhất.
Chỉnh ngưỡng không đồng nghĩa với fine-tune trọng số. Nghiên cứu C3 riêng của Huy
chỉ dùng validation, không gộp thành thêm seed hoặc số test của bảng chung.

### Tính lại đúng số đã báo cáo từ artifacts hiện có

Trong workspace này, code chính nằm ở `goemotions-source-only`; dataset/checkpoint/
scores lịch sử nằm ở thư mục cạnh nó `goemotions-multilabel-classification`.
Chạy từ thư mục code bằng Python đã có đủ thư viện:

```powershell
$py = "../goemotions-multilabel-classification/.venv-models/Scripts/python.exe"
& $py -m scripts.verification.reproduce_report --artifacts-root ../goemotions-multilabel-classification
```

Lệnh kiểm hash dataset/model/protocol, ghép scores theo ID, áp đúng ngưỡng đã khóa,
tính lại **72 dòng và 36 nhóm**, so toàn bộ 7 metric với CSV gốc ở sai số ≤ 1e-12.
Thiếu/sai dữ liệu sẽ dừng; không tải mạng, không tự tìm ngưỡng mới.
Kết quả kiểm ở `reports/reproduction/verification.json`; CSV tính lại nằm cùng thư mục.

Muốn **suy luận test thật lại** từ hai model A và cả 9 checkpoint C, thêm:

```powershell
& $py -m scripts.verification.reproduce_report --artifacts-root ../goemotions-multilabel-classification --inference-baseline --inference-c --device cuda
```

Có thể thêm `--report <đường-dẫn-báo-cáo-cuối-kỳ.docx>` để đối chiếu trực tiếp
111 ô metric trong 9 bảng kết quả Word. Output luôn tách khỏi artifacts gốc.
Lượt kiểm 20/10/2026 đã chạy cả hai A và chín C: scores/metric trùng bản cũ,
sai lệch lớn nhất bằng **0** trên cùng máy. B được tính lại từ scores NLI đã lưu;
hai lệnh trên không chạy lại suy luận B. Nghiên cứu C3 riêng của Huy thiếu bundle
checkpoint tại workspace này nên chưa suy luận lại được; không ảnh hưởng C3 chung.

Đây là tái lập metric/suy luận, **không phải huấn luyện lại** từ pretrained.
Huấn luyện lại cần cùng revision, seed, dependency và GPU; không bảo đảm giống bit
trên mọi máy. Muốn cải tiến mô hình, dùng thí nghiệm mới và validation; giữ bộ số
đã báo cáo cùng protocol cũ để đối chiếu.

![Văn bản đi qua A/B/C độc lập, rồi áp ngưỡng để trả nhiều cảm xúc](assets/overview.png)

## Đồ án làm gì?

Phân loại cảm xúc đa nhãn trong bình luận Reddit **tiếng Anh** bằng GoEmotions:
**27 cảm xúc + neutral**. Một câu có thể vừa cảm ơn, vừa vui hoặc lo lắng.
Mô hình trả 28 điểm; ngưỡng quyết định tập nhãn được chọn.

| Phần | Phương pháp | Vai trò |
|---|---|---|
| **A · Baseline** | TF-IDF + One-vs-Rest Logistic Regression | Mốc cổ điển; class weighting và threshold tuning |
| **B · Zero-shot** | BART-large-MNLI qua NLI | Dùng pretrained trực tiếp, không fine-tune bằng GoEmotions |
| **C · Fine-tune** | BERT / RoBERTa / DistilBERT | 28 logits + BCE, ba seed mỗi kiến trúc |
| **D · Demo** | Gradio | Nạp checkpoint C được chọn bằng validation |

A/B/C chạy độc lập, cùng dữ liệu/mapping/metric. A là mốc so sánh,
không phải đầu vào bắt buộc của Transformer.

<a id="quick-start"></a>
## 1. Chạy thử nhanh trên CPU

Cần **Python 3.12 hoặc 3.13**. Chạy lệnh từ thư mục có `pyproject.toml`.

```bash
git clone https://github.com/trangkhanh-ai/goemotions-multilabel-classification.git
cd goemotions-multilabel-classification
python -m venv .venv
```

| Hệ điều hành | Kích hoạt |
|---|---|
| PowerShell | `./.venv/Scripts/Activate.ps1` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Linux/macOS | `source .venv/bin/activate` |

```bash
python -m pip install -e .
python -m examples.baseline_from_scratch
```

Ví dụ NumPy in vocabulary, IDF, TF-IDF, loss và nhãn dự đoán.
**Dữ liệu tự tạo; không tải model/dataset, không dùng điểm này làm benchmark.**

Chạy notebook với pipeline scikit-learn và metric của baseline thật:

```bash
python -m pip install -e ".[notebooks]"
python -m scripts.data.run_notebooks --notebook 00_quick_start
```

Đọc [00_quick_start.ipynb](notebooks/00_quick_start.ipynb).
Bản có output được lưu cục bộ ở `reports/notebooks/00_quick_start.executed.ipynb`.

<a id="setup"></a>
## 2. Cài đúng phần cần dùng

| Nhu cầu | Lệnh |
|---|---|
| Baseline CPU | `python -m pip install -e .` |
| EDA + Jupyter | `python -m pip install -e ".[eda,notebooks]"` |
| B, ba C và Gradio | Cài PyTorch phù hợp, rồi `python -m pip install -e ".[transformers]"` |
| Đầy đủ | Cài PyTorch phù hợp, rồi `python -m pip install -e ".[eda,notebooks,transformers,dev]"` |
| Nghiên cứu C3 riêng | `python -m pip install -e ".[study,eda,notebooks]"` |

CPU: cài wheel CPU trước extra Transformer:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[transformers]"
```

NVIDIA: chọn wheel đúng GPU/driver tại [PyTorch](https://pytorch.org/get-started/locally/).
Kiểm đang dùng đúng Python và CUDA:

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

[requirements/](requirements/) chia dependency theo phần;
[models-lock.txt](requirements/models-lock.txt) ghi môi trường nghiên cứu trước.
Wheel CPU/CUDA vẫn cần phù hợp máy. Nếu PowerShell chặn kích hoạt, dùng trực tiếp
`./.venv/Scripts/python.exe` thay cho `python`.

<a id="baseline"></a>
## 3. Baseline A — phần của Duy

![Sáu bước: văn bản, TF-IDF, LR, scores, ngưỡng và tập cảm xúc](assets/baseline_steps.png)

Điểm trong sơ đồ là **minh họa**, không phải số GoEmotions đã đo.

1. **TF-IDF:** fit vocabulary/IDF chỉ trên train; val/test chỉ transform.
2. **OvR:** mỗi nhãn có một LR nhị phân riêng, tổng cộng 28 LR.
3. **Score:** `sigmoid(w_j · x + b_j)`; 28 điểm không cần cộng thành 1.
4. **Ngưỡng:** chọn mọi nhãn vượt ngưỡng, không dùng argmax ép một nhãn.
5. **Metric:** so tập nhãn với multi-hot thật bằng module chung.

TF-IDF: unigram/bigram, `min_df=2`, `max_features=100000`.
LR: `C=1`, `solver="liblinear"`, `max_iter=1000`, `random_state=42`.
Standard dùng trọng số mặc định; balanced dùng `class_weight="balanced"`
cho mỗi bài toán nhị phân, tính từ train.

### Train → analyze → predict

```bash
python -m scripts.baseline.run_baseline
python -m scripts.baseline.run_baseline --variant balanced
python -m scripts.baseline.analyze_baseline
python -m scripts.baseline.predict_baseline --variant balanced --threshold tuned --text "Thank you, but I am still worried."
```

Lần đầu tải snapshot train/validation và kiểm SHA-256.
Hai lần train tạo hai model độc lập; analyze cần cả hai bản **full**.
Predictor nạp model balanced và ngưỡng đúng model đó.

| `--threshold` | Luật quyết định |
|---|---|
| `fixed` | 0,5 cho mọi nhãn |
| `global` | Một ngưỡng chung chọn bằng Macro-F1 validation |
| `tuned` | Một ngưỡng mỗi nhãn chọn bằng F1 validation của nhãn đó |

Lưới 0,05–0,95, bước 0,05; hòa chọn gần 0,5 rồi ngưỡng lớn hơn.
Không chuyển ngưỡng standard sang balanced.

### Khóa cấu hình trước test

Sau khi đã chốt lựa chọn bằng validation:

```bash
python -m scripts.baseline.freeze_baseline
python -m scripts.baseline.evaluate_baseline_test
```

Tuned-validation dùng chính tập chọn ngưỡng, có thể lạc quan.
Test giữ model/ngưỡng đã khóa, không dùng test chọn lại.
Weighting/tuning có thể đổi P/R/Hamming theo nhiều chiều; báo cả nhãn giảm F1.
So standard fixed với balanced tuned thay cả weighting và ngưỡng, chưa cô lập từng kỹ thuật.

Kiểm luồng nhỏ: `python -m scripts.baseline.run_baseline --smoke`.
Smoke dùng 5.000 train/1.000 val, lưu riêng, không thay model full hoặc benchmark.

<a id="notebooks"></a>
## 4. Notebook: đọc và chạy theo thứ tự

| Notebook | Nội dung | Điều kiện |
|---|---|---|
| [00 · Chạy thử](notebooks/00_quick_start.ipynb) | Fit/predict/metric, dữ liệu tự tạo | Baseline + notebooks; không tải dữ liệu |
| [01 · EDA](notebooks/01_eda.ipynb) | Split, nhãn, multi-hot, mất cân bằng | Extra EDA; tải snapshot lần đầu |
| [02 · Baseline](notebooks/02_baseline.ipynb) | Sáu cấu hình A, score, ngưỡng và lỗi | Train hai A full + analyze trước |
| [03 · Zero-shot](notebooks/03_zero_shot.ipynb) | NLI và trạng thái/kết quả B | Extra Transformer; artifact B để đọc số |
| [04 · Transformers](notebooks/04_transformers.ipynb) | BCE, seed/checkpoint/protocol, ba C | Extra Transformer; artifact C để đọc số |
| [05 · DistilBERT Huy](notebooks/05_distilbert_huy.ipynb) | Nghiên cứu C3 riêng | Extra study và bundle/output riêng |
| [06 · EDA bổ sung](notebooks/06_eda_extra.ipynb) | Độ dài, tokenizer và câu hỏi dữ liệu | Sau 01, extra EDA |

Mở Jupyter từ gốc repo: `python -m jupyterlab`, chọn kernel của `.venv`.
Hoặc chạy notebook 02 bằng Python hiện tại, sau bước train/analyze A:

```bash
python -m scripts.data.run_notebooks --notebook 02_baseline
```

Runner giữ source sạch, ghi output vào `reports/notebooks/`.
B/C đọc artifact và in lệnh, **không tự train full khi Run All**.
Thông báo thiếu artifact trên clone mới nghĩa là bạn chưa chạy thí nghiệm đó.

<a id="data"></a>
## 5. Dữ liệu và đánh giá chung

| Hạng mục | Quy ước |
|---|---|
| Dataset | GoEmotions `simplified`, Reddit tiếng Anh |
| Revision | `add492243ff905527e67aeb8b80c082af02207c3` |
| Official split | **43.410 train / 5.426 val / 5.427 test** |
| Nhãn | [data/labels.json](data/labels.json), 28 vị trí |
| Nhãn thật | Multi-hot `N × 28` |
| Scores | `ids`, `scores` N×28, `label_names`; căn theo ID |
| Metric | Macro/Micro-F1, Precision/Recall, Hamming; per-label P/R/F1/support |

<details>
<summary>Phân bố nhãn thật trên train</summary>

![Phân bố 28 nhãn trên official train; một câu có thể được tính ở nhiều nhãn](assets/label_distribution.png)

Ảnh EDA đã tạo từ snapshot train cố định; không phải lỗi dự đoán.
Nhãn hiếm chọn từ train: grief 77, pride 111, relief 153, nervousness 164, embarrassment 303.
Code tạo: [build_eda.py](tools/notebooks/build_eda.py).

</details>

Giữ neutral theo annotation, không tự gán neutral khi không điểm nào vượt ngưỡng.
Không âm thầm chia lại/xóa mẫu khỏi official split. EDA kiểm trùng văn bản/ID
và giới hạn dữ liệu. Tiếng Việt/dữ liệu doanh nghiệp cần đánh giá riêng.

<a id="zero-shot"></a>
## 6. Zero-shot B

Cài extra Transformer trước. Smoke kiểm luồng; full tạo kết quả thực nghiệm:

```bash
python -m scripts.zero_shot.run_zero_shot --smoke --device cpu --dtype float32
python -m scripts.zero_shot.run_zero_shot --device cuda --dtype float16 --batch-size 16
```

CPU dùng `--device cpu --dtype float32`; full cần nhiều phép suy luận.
BART-MNLI giữ nguyên trọng số, đặt từng cảm xúc vào giả thuyết NLI, dùng đủ 28 nhãn.
HF sắp output theo score; code căn về mapping chuẩn. Neutral NLI khác neutral GoEmotions.
Nếu dùng nhãn val để chọn ngưỡng, ghi riêng hàng có hiệu chỉnh bằng nhãn đích.
NLI score chưa mặc nhiên là xác suất cảm xúc đã hiệu chuẩn.

<a id="experiments"></a>
## 7. Ba Transformer và thực nghiệm full

| Kiến trúc | Checkpoint | LR | Epoch tối đa |
|---|---|---:|---:|
| BERT | `google-bert/bert-base-cased` | 5e-5 | 4 |
| RoBERTa | `FacebookAI/roberta-base` | 2e-5 | 3 |
| DistilBERT | `distilbert/distilbert-base-uncased` | 2e-5 | 3 |

Batch 16, max length 128, accumulation 1; seed **42, 123, 2026**.
28 logits đi trực tiếp vào `BCEWithLogitsLoss`; sigmoid khi suy luận.
Checkpoint mỗi seed chọn Macro-F1 validation @0,5.
BERT tham khảo GoEmotions; C2/C3 là cấu hình nhóm. Ba seed đo biến động,
không chứng minh đã tối ưu LR. Khác LR/epoch/tokenizer là giới hạn so kiến trúc.

```bash
# Smoke tải checkpoint, chạy ít mẫu; không tạo benchmark full.
python -m scripts.transformers.train_transformer --architecture distilbert --seed 42 --smoke --device cpu
# Full: ba kiến trúc x ba seed, chạy tuần tự.
python -m scripts.transformers.run_transformer_seeds --device cuda --resume
```

Sau khi **hai baseline full + analyze A đã xong**, có thể chạy pipeline chung:

```bash
python -m scripts.pipeline.complete_project --device cuda
```

Pipeline: ba C × ba seed → B val → chọn C bằng val → khóa A/B/C → test → CSV/JSON,
metadata và nhóm lỗi. CPU đổi thành `--device cpu`; tốc độ tùy máy.
Đã có đủ run full/hash hợp lệ thì dùng `--skip-training`.
`--resume` bỏ qua run hoàn tất đã kiểm, không phục hồi optimizer của run dở.
Không chạy `--overwrite` lên run cần giữ. Một tác vụ GPU tại một thời điểm.

**Đọc kết quả đúng:** C có từng seed và mean ± sample std (`ddof=1`, n=3);
A/B n=1 không tự tạo std. Chọn C bằng mean val Macro-F1 @0,5, không chọn lại bằng test.
Ngưỡng gắn đúng model/revision/seed. Báo nhãn tăng/giảm, support, lỗi có ID.
Cặp FN/FP khác đồng xuất hiện nhãn thật; điểm toy/smoke không thay benchmark.

<a id="demo"></a>
## 8. Demo D — một câu có thể nhiều cảm xúc

![Demo Gradio đã chạy: dự đoán gratitude, joy và bảng score/ngưỡng 28 nhãn](assets/demo.png)

Ảnh đã kiểm ngày **08/10/2026**, BERT seed 123 được chọn bằng validation.
Đây là minh chứng một lần chạy, không khẳng định app đang mở trên máy bạn.
Sau khi pipeline tạo checkpoint C và `selected_model.json` ở máy:

```bash
python -m scripts.transformers.verify_demo --device auto
python app.py --device auto
```

Trong workspace có sẵn artifacts bên cạnh, dùng biến `$py` ở phần tái lập:

```powershell
& $py app.py --artifacts-root ../goemotions-multilabel-classification --device cuda
```

Kiểm hàm dự đoán và luồng nhập sau khi đổi cấu trúc:
`& $py -m scripts.transformers.verify_demo --artifacts-root ../goemotions-multilabel-classification --device cuda`.

Mở `http://127.0.0.1:7860/`. App cần checkpoint thật; clone source không có trọng số.
Ngưỡng mặc định 0,5; ngưỡng riêng cần `--thresholds` đúng protocol run đã chọn.

<a id="distilbert-study"></a>
### DistilBERT riêng của Nhật Huy

[scripts/distilbert/](scripts/distilbert/), [configs/](configs/) và
[distilbert_study.py](src/models/distilbert_study.py) giữ nghiên cứu riêng.
Không gộp khác protocol/thiết bị thành sáu seed C3 chính.

```bash
python -m scripts.distilbert.train_distilbert --help
python -m scripts.distilbert.predict_distilbert --help
```

`scripts/distilbert/run_c3_full.py` chạy full trực tiếp, không có `--help`; đọc config trước.
Sau khi đủ run riêng, `python -m scripts.distilbert.analyze_c3_full` và
`python -m scripts.distilbert.plot_c3_results` xuất CSV/JSON/hình.
`app_distilbert_huy.py` là Streamlit riêng, cần bundle C3; `app.py` là demo benchmark chính.

<a id="code-map"></a>
## 9. Source gọn, mỗi phần một nhiệm vụ

```text
.
├── README.md             # Toàn bộ giới thiệu và hướng dẫn
├── assets/               # Sơ đồ, ảnh demo và phân bố nhãn
├── src/
│   ├── datasets/         # Dữ liệu, hash, mapping, multi-hot
│   ├── models/           # A, B, C và nghiên cứu C3 riêng
│   ├── evaluation/       # Metric, ngưỡng và protocol
│   ├── apps/             # Implementation Gradio và Streamlit
│   └── learning/         # Ví dụ thuật toán NumPy
├── scripts/
│   ├── data/             # EDA và chạy notebook
│   ├── baseline/         # Train, analyze, predict, freeze, test A
│   ├── zero_shot/        # BART-MNLI
│   ├── transformers/     # Ba C, seed, test, chọn/kiểm demo
│   ├── distilbert/       # Nghiên cứu C3 riêng
│   ├── analysis/         # CSV/JSON, lỗi và metadata
│   ├── verification/     # Tính lại số và suy luận kiểm báo cáo
│   └── pipeline/         # Ghép quy trình
├── notebooks/            # 00 chạy thử; 01–06 đồ án
├── examples/             # Ví dụ CPU không cần Internet
├── tests/                # Kiểm metric và protocol
├── configs/              # Cấu hình C3 riêng
├── data/                 # Git chỉ giữ mapping/revision
├── requirements/         # Dependency theo nhu cầu
├── tools/                # Kiểm source/link và tạo notebook trong notebooks/
└── app.py                # Điểm chạy demo, gọi src/apps/demo.py
```

| Muốn hiểu | File |
|---|---|
| Tạo TF-IDF + LR | [run_baseline.py](scripts/baseline/run_baseline.py), `build_model` |
| Căn scores/ngưỡng A | [baseline.py](src/models/baseline.py) |
| Metric dùng chung | [metrics.py](src/evaluation/metrics.py) |
| Khóa lựa chọn trước test | [protocols.py](src/evaluation/protocols.py) |
| Freeze/test A | [freeze_baseline.py](scripts/baseline/freeze_baseline.py), [evaluate_baseline_test.py](scripts/baseline/evaluate_baseline_test.py) |
| Chọn C cho demo | [select_best_transformer.py](scripts/transformers/select_best_transformer.py), [app.py](app.py) |
| Train C từng bước | [train_transformer.py](scripts/transformers/train_transformer.py) |
| Kiểm số báo cáo và chạy lại suy luận | [reproduce_report.py](scripts/verification/reproduce_report.py) |

**Output cục bộ:** `data/raw/` và `data/tokenizers/` chứa cache;
`data/processed/baseline/`, `zero_shot/`, `transformers/` chứa model/scores/protocol;
`reports/project_results/` chứa bảng; `reports/notebooks/` chứa notebook đã chạy.
Git bỏ qua những output này. Word/PDF, bài báo, nguồn viết báo cáo và tài liệu
riêng giữ ngoài repo. **Chỉ một Markdown được đưa vào Git: README.md.**

<a id="checks"></a>
## 10. Kiểm code

```bash
python -m pip install -e ".[dev]"
python tools/check_repository.py
python -m unittest discover -s tests -v
python -m examples.baseline_from_scratch
```

Kiểm Torch cần extra Transformer. Tests dùng dữ liệu giả/mock, không tải checkpoint
hoặc train GoEmotions. Checker chặn docs/output ngoài phạm vi, notebook có output,
import thiếu và link README hỏng. Xóa output/execution count nếu sửa notebook bằng Jupyter;
runner giữ bản có output riêng. Đổi code tính điểm/protocol phải kiểm lại, không đổi tên run cũ.

<a id="troubleshooting"></a>
## 11. Lỗi thường gặp

| Vấn đề | Xử lý |
|---|---|
| `No module named src` | Chạy ở gốc repo, cài editable `python -m pip install -e .` |
| Notebook thiếu thư viện | Kernel phải dùng đúng `.venv`, cài extra tương ứng |
| Notebook 02 thiếu file | Train hai A full + analyze; smoke không thay full |
| Demo thiếu checkpoint | Hoàn tất C và bước chọn C bằng validation |
| CUDA không có/hết VRAM | Kiểm wheel/driver; thử smoke CPU; đổi config full phải ghi thành run riêng |
| Hash/protocol không khớp | Dừng, kiểm artifact nguồn; không bỏ qua kiểm tra để lấy điểm |

<a id="sources"></a>
## 12. Nguồn, phạm vi và ghi nhận

- Bài nền tảng: [GoEmotions, Demszky et al., ACL 2020](https://aclanthology.org/2020.acl-main.372/).
- [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions).
- Baseline: [TF-IDF](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [OvR](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), [Logistic Regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [threshold tuning](https://scikit-learn.org/1.7/modules/classification_threshold.html).
- B: [BART-MNLI model card](https://huggingface.co/facebook/bart-large-mnli).
- C/D: [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [Gradio](https://github.com/gradio-app/gradio).

Cách bố trí giới thiệu/cài đặt/ví dụ/nguồn tham khảo
[scikit-learn](https://github.com/scikit-learn/scikit-learn),
[Transformers](https://github.com/huggingface/transformers) và
[Gradio](https://github.com/gradio-app/gradio). Hình quy trình do nhóm tạo;
không sao chép logo hay mã từ README của các dự án đó.

**Nhóm 4 người:** Duy — baseline A, điều phối B và bàn giao;
Quốc Khánh — BERT C1; Đức Trí — RoBERTa C2/zero-shot theo phân công;
Nhật Huy — DistilBERT C3, nghiên cứu riêng và demo.
Tỷ lệ đóng góp do nhóm xác nhận, không suy từ tác giả commit.

**Tác giả Git bản cập nhật:** `nguyndy` · `baoduynguyen1612@gmail.com`.
Nhóm chưa công bố giấy phép riêng cho source; dữ liệu/checkpoint/thư viện theo nguồn tương ứng.
Score chưa mặc nhiên là xác suất nghiệp vụ; áp dụng ngoài Reddit tiếng Anh cần kiểm chứng riêng.
