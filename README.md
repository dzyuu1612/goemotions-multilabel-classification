# GoEmotions phân loại cảm xúc đa nhãn

## Bản nâng cấp ngày 29/10/2026

Bổ sung biểu đồ ngày **30/10/2026**: ma trận đồng xuất hiện FN/FP của ba C,
bảng màu F1 theo đủ 28 nhãn và biểu đồ điểm/ngưỡng của câu demo. Ý tưởng trình
bày tham khảo báo cáo XED của nhóm Quý; toàn bộ số liệu lấy từ GoEmotions.
Giữ cùng 10 nhãn và thang màu cho ba ma trận. Chúng đếm cặp lỗi trên cùng câu,
không phải ma trận đơn nhãn hoặc chứng minh cải thiện mô hình. Word/slide bổ sung
được lưu riêng, không ghi đè bản ngày 29/10.

Tái tạo biểu đồ từ CSV đã chốt, không train hoặc tìm ngưỡng:

```bash
python -m scripts.analysis.plot_error_heatmaps --output artifacts/charts_20261030
```

Thêm `--demo-scores /path/to/new_cases_c123.json` để vẽ điểm/ngưỡng cho case 2
từ bộ câu mới đã lưu. File này nằm ở `verification/` của gói nâng cấp ngày 29/10.
Script xuất năm PNG khi có demo scores, ba bảng CSV và manifest ghi SHA-256 nguồn.
F1 heatmap dùng mean đủ ba seed; std giữ trong CSV. Ma trận lỗi dùng từng
checkpoint đại diện BERT/123, RoBERTa/2026, DistilBERT/123 ở ngưỡng 0,5.

Nhánh `codex/goemotions-upgrade-20261029` được tạo từ `a7798d9` của source
hiện có. Giữ nguyên benchmark `results/final_submission`; thay đổi mới bổ sung
khả năng chẩn đoán huấn luyện và đóng gói suy luận. Chưa chạy tìm siêu tham số
full, nên chưa có kết quả chứng minh cấu hình mới tăng F1.

### Demo không huấn luyện lại

Sau khi khôi phục checkpoint như hướng dẫn bên dưới, tạo một tệp mô hình và
một gói web cục bộ:

```bash
python -m scripts.verification.package_demo --output artifacts/portable_release
```

Gói `GoEmotions_Web.zip` chứa `GoEmotions_BERT.goemotions`, giao diện và bộ khởi
động Windows. Tệp `.goemotions` chứa BERT seed 123 đã fine-tune, tokenizer,
mapping, max_length và 28 ngưỡng từ validation, kèm SHA-256 từng thành phần.
Không mở notebook, train hoặc tìm ngưỡng khi demo. Giải nén rồi chạy
`START_DEMO.cmd`. Máy cần Python 3.12/3.13 và Internet cho lần cài dependency
đầu; sau đó suy luận cục bộ offline ở `http://127.0.0.1:7870`.
Đây là gói web có mô hình, chưa phải MSI chứa sẵn Python.

Có thể chạy trực tiếp bằng môi trường đã cài dependency:

```bash
python deployment/web_demo.py --model artifacts/portable_release/GoEmotions_Web/GoEmotions_BERT.goemotions
```

Web cho nhập câu mới, đổi ngưỡng riêng/0,5, xem token và xuất bảng CSV.
CSV chỉ chứa điểm dự đoán, không chứa trọng số mô hình. Câu demo tự soạn không
có gold label nên không dùng để tính accuracy. Các mô hình so sánh A/B/C vẫn
có ở Gradio gốc. Web đóng gói chỉ dùng checkpoint C được chọn bằng validation.

### Thử LR và ngân sách epoch cho C1 C2 C3

```bash
# Chỉ tạo plan.json với 9 cấu hình pilot, chưa train.
python -m scripts.transformers.plan_improvements --output-root artifacts/improvements/pilot --device cuda
# Kiểm kỹ thuật 64 train/32 validation, 1 epoch cho mỗi C.
python -m scripts.transformers.plan_improvements --output-root artifacts/improvements/smoke --device cuda --smoke --execute
# Chạy full pilot khi có ngân sách GPU.
python -m scripts.transformers.plan_improvements --output-root artifacts/improvements/pilot --device cuda --execute
# Sau khi đủ full pilot: chọn bằng validation, tạo kế hoạch xác nhận 3 seed/C.
python -m scripts.transformers.plan_improvements --stage confirm --pilot-root artifacts/improvements/pilot --output-root artifacts/improvements/confirm --device cuda
```

Thêm `--execute` ở bước confirm để chạy. Pilot dùng LR 2e-5/3e-5/5e-5, trần
8 epoch, batch 16, patience 2 và min_delta 0,001 Macro-F1. Đây là các giả thuyết
khảo sát. Confirm dùng cùng một cấu hình đã chọn cho ba seed 42/123/2026,
khóa revision theo pilot. Không coi ba smoke run là benchmark hoặc kết luận
cải thiện. Tăng trần epoch cũng thay lịch LR tuyến tính nên phải ghi cả scheduler.
Sau confirm, dùng các công cụ freeze/evaluate của dự án với root kết quả mới.
Test cũ đã được quan sát, nên muốn đánh giá độc lập cần holdout mới.

Trainer có thêm `--output-root`, `--patience`, `--min-delta` và `--diagnostics`.
Mỗi epoch lưu `validation_bce_eval` và `learning_rate_end`; bật diagnostics
thì đo thêm `train_bce_eval`, `train_macro_f1_eval`, `train_micro_f1_eval` và gap.
Eval tắt dropout. BCE eval không trọng số để đối chiếu giữa split, không thay
loss train có dropout/weighted objective. Diagnostics cần thêm một lượt suy
luận train mỗi epoch. Best checkpoint luôn theo val Macro-F1 @0,5; patience là
quy tắc dừng khi không có cải thiện đáng kể, không phải chứng minh overfit.

Seed 42/123/2026 không có tính chất làm F1 cao hơn; chúng cố định các lượt ngẫu
nhiên để đo dao động. LR nhỏ giúp điều chỉnh biểu diễn pretrained thận trọng,
nhưng quá nhỏ có thể học chậm. Chín run cũ đều chọn epoch cuối; chỉ nhìn train
loss giảm chưa xác nhận hội tụ hoặc không overfit. Nguồn giải thích:
[BERT, Appendix A.3](https://aclanthology.org/N19-1423/),
[GoEmotions, §5.3](https://aclanthology.org/2020.acl-main.372/),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html),
[tokenizer theo checkpoint](https://huggingface.co/docs/transformers/v4.57.1/tokenizer_summary).

Kiểm bản nâng cấp:

```bash
python -m unittest tests.test_training_control tests.test_portable tests.test_transformer tests.test_metrics -v
python tools/check_repository.py
python -m examples.pretrained_math
```

[Notebook 07](notebooks/07_improvement_workflow.ipynb) tách các bước C1/C2/C3,
để cờ huấn luyện mặc định tắt. Ví dụ `pretrained_math` kiểm gradient bằng sai phân
hữu hạn và in attention/BCE/SGD đúng các phép tính ở Phụ lục C.

Bản Word và PPTX được tạo từ bản gốc GoEmotions, giữ ngoài Git. Phụ lục C đối
chiếu ba PDF môn học, giải thích pretrained/LR/seed/tokenizer, tính tay gradient,
so sánh phạm vi XED, phân tích lỗi và hướng chạy cải tiến. So sánh XED dựa trên
báo cáo được cung cấp, chưa có code/checkpoint XED để kiểm độc lập. Không xếp
hạng F1 giữa hai dataset khác nhau.

Bộ nộp ngày **26/10/2026** dùng **A/B/BERT/RoBERTa của Duy** và **toàn bộ ba seed
DistilBERT của Huy**. Báo cáo dùng bản Duy làm nền. Nguồn được ghi rõ trong
[provenance.json](results/final_submission/provenance.json); không ghép từng seed
hoặc từng chỉ số cao hơn từ hai bộ.

Nhóm đã xem test của cả hai bộ trước khi chốt nguồn này. Vì vậy, lựa chọn nguồn
không độc lập với test; so sánh cuối mang tính mô tả. Trong từng run gốc,
checkpoint và ngưỡng vẫn giữ lựa chọn trên validation. Test đã được quan sát
trong EDA; không nhận đây là một tập chưa từng được xem.

[Notebook có output](notebooks/executed/) · [Bộ nộp chính thức](results/final_submission/)
· [Checkpoint đã huấn luyện](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/releases/tag/goemotions-final-duy-huy-20261026)
· [Demo](#demo) · [Cách đọc code](#code-map)

## Kết quả chính thức

Chỉ số **test**, ngưỡng riêng đã khóa từ validation. C dùng mean ± sample std
(`ddof=1`) của đủ ba seed 42, 123, 2026. A/B có một model cho mỗi cấu hình.

| Hệ | Macro-F1 | Micro-F1 | Hamming Loss ↓ |
|---|---:|---:|---:|
| A balanced tuned | 0.4493 | 0.5277 | 0.0467 |
| B BART-MNLI tuned | 0.1609 | 0.1752 | 0.2941 |
| C1 BERT tuned | 0.5038 ± 0.0097 | 0.5900 ± 0.0029 | 0.0373 ± 0.0003 |
| C2 RoBERTa tuned | 0.4872 ± 0.0156 | 0.6048 ± 0.0057 | 0.0365 ± 0.0014 |
| C3 DistilBERT tuned | 0.4907 ± 0.0093 | 0.5974 ± 0.0023 | 0.0370 ± 0.0005 |

Demo mặc định BERT seed 123 theo Macro-F1 validation ở ngưỡng 0,5. BERT dẫn
Macro-F1; RoBERTa có Micro-F1 sau tuning cao hơn. Đây là một cấu hình mỗi kiến
trúc với ba seed, chưa phải tìm siêu tham số tối ưu hoặc đối chứng chỉ thay encoder.

| Minh chứng | Tệp |
|---|---|
| 72 dòng metric và 36 nhóm mean/std | [all_runs.csv](results/final_submission/report_data/all_runs.csv), [mean_std.csv](results/final_submission/report_data/mean_std.csv) |
| 30 điểm đường học | [learning_curves.csv](results/final_submission/report_data/learning_curves.csv) |
| F1 nhãn hiếm, số đếm lỗi và ví dụ | [report_data/](results/final_submission/report_data/) |
| 14 biểu đồ từ đúng bộ nộp | [figures/](results/final_submission/figures/), [mã vẽ](scripts/analysis/plot_submission.py) |
| Kiểm số và hash checkpoint | [verification.json](results/final_submission/report_data/verification.json) |
| Mười câu BRIGHTER đã khóa và 240 dự đoán | [external_probe/](results/final_submission/external_probe/) |
| Kiểm dữ liệu trùng và train-only TF-IDF | [data_boundary_audit.json](results/final_submission/data_boundary_audit.json) |
| Mã hóa riêng các split và kiểm truncation/UNK | [encoding_audit.json](results/final_submission/encoding_audit.json) |
| Đối chiếu metric độc lập | [independent_crosscheck.json](results/final_submission/independent_crosscheck.json) |
| Chênh lệch suy luận mới so với scores gốc | [replay_comparison.csv](results/final_submission/replay_comparison.csv) |
| Sáu lựa chọn demo, kiểm API mới nhất | [demo_api_recheck.json](results/final_submission/demo_api_recheck.json) |

### Bản Word rút gọn hoàn thiện

Bản rút gọn giữ 48 trang, dùng cùng bộ kết quả chính thức ở trên và đủ ba seed
DistilBERT của Huy. Bảng chính giữ bốn cấu hình baseline standard/balanced ×
ngưỡng 0,5/ngưỡng riêng để phân biệt tác động của weighting và tuning.
Phần nghiên cứu liên quan bổ sung BERT Macro-F1 khoảng 0,46 và BiLSTM khoảng
0,41 của bài GoEmotions; std 0,19 trong bài gốc là giữa các nhãn, khác std giữa
ba seed. Không suy ra kết luận vượt bài gốc từ hai thiết lập khác nhau.

[Biên nhận bản rút gọn](results/final_submission/compact_report/report_manifest.json)
ghi tên, hash và số trang của Word phát hành cục bộ. Biên nhận
[bản đầy đủ 84 trang](results/final_submission/report_manifest.json) được giữ riêng.
Ba hình bổ sung có [ảnh và dữ liệu vẽ](results/final_submission/compact_report/)
và [mã tái tạo](scripts/analysis/plot_compact_report.py):

```bash
python -m scripts.analysis.plot_compact_report
```

Lệnh đọc CSV/JSON đã lưu để vẽ phân bố số nhãn, hai điểm Precision–Recall trước/sau
tuning và FN/FP trên mười câu BRIGHTER; không train, tìm ngưỡng hoặc ghi đè
scores. [Biên nhận vẽ](results/final_submission/compact_report/plot_manifest.json)
ghi hash nguồn, số liệu từng hình và phạm vi checkpoint. BRIGHTER được dẫn theo
[bản ACL 2025 đã xuất bản](https://aclanthology.org/2025.acl-long.436/).

## Khôi phục bộ nộp và chạy demo

Dùng Python 3.12 hoặc 3.13 trong môi trường riêng. Cài PyTorch phù hợp máy theo
phần setup bên dưới; các lệnh sau chạy ở thư mục chứa `pyproject.toml`.

```bash
python -m pip install -e ".[eda,transformers,notebooks]"
python -m scripts.verification.prepare_submission --checkpoints
python -m scripts.verification.verify_submission --full
python app.py --device auto
```

`prepare_submission` kiểm SHA-256 rồi khôi phục dữ liệu, scores, metadata và
selection vào `artifacts/final_submission/`. Thêm `--checkpoints` tải chín C và
hai A từ Releases theo [catalog](results/final_submission/checkpoint_assets.json).
Nếu chỉ chạy demo, thêm `--demo-only` để lấy ba checkpoint đại diện và hai A.
BART-MNLI tải từ Hugging Face theo revision đã khóa khi được chọn lần đầu.
Đây là khôi phục kết quả đã huấn luyện, không train lại. Mặc định app và notebook
chọn đúng profile này; biến `GOEMOTIONS_ARTIFACTS_ROOT` hoặc `--artifacts-root`
là tùy chọn chủ động khi cần một bộ khác. Không chép đè metadata của hai bộ lên nhau.

Muốn chỉ tính lại bảng từ scores, chưa cần trọng số lớn:

```bash
python -m scripts.verification.prepare_submission
python -m scripts.verification.verify_submission --export
```

Muốn kiểm notebook, tokenizer, dữ liệu và vẽ lại sau khi tải đủ checkpoint:

```bash
python -m scripts.data.run_notebooks --notebook all --publish
python -m scripts.verification.audit_encoding
python -m scripts.verification.crosscheck_submission
python -m scripts.analysis.plot_submission
python -m scripts.analysis.evaluate_external_probe
```

Bảy notebook đã chạy đủ **72 cell mã**. A và cả chín C có suy luận lại từ
checkpoint; B trong notebook tính metric từ scores NLI đã lưu. Phép thử mười
câu có suy luận mới cho cả A/B/chín C, không chỉnh nhãn, train hoặc tìm ngưỡng.
Các kết quả mới của lệnh thử câu được ghi riêng ở `artifacts/external_probe_replay/`.

Benchmark kiểm chính xác từ scores gốc và hash trọng số. Replay AMP trên môi
trường khác có thể lệch sát ngưỡng: lượt này C3 có max score drift 0,001775 và tối
đa năm quyết định nhãn đổi trong một cấu hình; C1/C2 khớp trên môi trường hiện tại.
Notebook dùng giới hạn score drift 0,005 cho C, báo số quyết định đổi và chênh
metric; đây không phải chứng nhận metric replay bằng hệt benchmark. Scores gốc,
ngưỡng, checkpoint và bảng báo cáo không bị ghi đè. Chẩn đoán train/validation
float32 cũng được lưu riêng với log AMP huấn luyện.

Demo giữ giao diện cũ có sáu lựa chọn. HTTP/API đã kiểm thành công cho cả sáu,
mỗi lựa chọn trả đủ 28 hàng điểm. Phiên cuối **chưa kiểm lại trực quan trình
duyệt** vì công cụ bị dừng; [ảnh cũ](assets/demo.png) chỉ minh họa bố cục đã dùng.
`127.0.0.1:7860` là địa chỉ cục bộ, chưa có hosting công khai.

## Các bộ kết quả giữ riêng để đối chiếu

`results/final_submission/` là nguồn cho Word final. `results/training_20261025/`
và `reports/project_results/` giữ nguyên bộ Huy chạy lại toàn bộ; `results/report_2_1/`
giữ snapshot trước đó. Các nghiên cứu trong `scripts/distilbert/` là hồ sơ lịch sử,
không tự động trở thành C3 của bản final. Checkpoint lớn phát hành ở Releases;
Git chỉ giữ bundle runtime khoảng 23,5 MB, bảng, hình và biên nhận.

## Huấn luyện một thực nghiệm khác

Các hướng dẫn phía dưới dành cho học và tạo lượt chạy mới. Dùng thư mục artifacts
riêng và chỉ định rõ khi chạy công cụ tổng hợp/kiểm chứng. Không dùng pipeline train
để mở lại bộ nộp đã khóa. `--resume` chỉ bỏ qua lượt hoàn thành hợp lệ;
`--overwrite` thay kết quả cũ nên không dùng cho bộ cần giữ.

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
| **D · Demo** | Gradio | Mặc định C chọn bằng validation; cho phép thử A, B và ba C |

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

| Source để chỉnh | Bản đã chạy, có output | Phạm vi của kết quả |
|---|---|---|
| [00 · Chạy thử](notebooks/00_quick_start.ipynb) | [Mở kết quả 00](notebooks/executed/00_quick_start.executed.ipynb) | Fit/predict trên dữ liệu minh họa; không dùng làm benchmark |
| [01 · EDA](notebooks/01_eda.ipynb) | [Mở kết quả 01](notebooks/executed/01_eda.executed.ipynb) | Tính mới thống kê/biểu đồ trên official snapshot |
| [02 · Baseline](notebooks/02_baseline.ipynb) | [Mở kết quả 02](notebooks/executed/02_baseline.executed.ipynb) | Đọc model đã fit, suy luận test mới cả hai A, kiểm sáu cấu hình |
| [03 · Zero-shot](notebooks/03_zero_shot.ipynb) | [Mở kết quả 03](notebooks/executed/03_zero_shot.executed.ipynb) | Tính lại sáu dòng validation/test từ scores NLI đã lưu; không chạy mới BART |
| [04 · Transformers](notebooks/04_transformers.ipynb) | [Mở kết quả 04](notebooks/executed/04_transformers.executed.ipynb) | Hồ sơ C1/C2/C3; BCE forward; suy luận test đủ chín checkpoint của pipeline cục bộ |
| [05 · DistilBERT Huy](notebooks/05_distilbert_huy.ipynb) | [Mở kết quả 05](notebooks/executed/05_distilbert_huy.executed.ipynb) | Ba lượt full C3; suy luận validation/test, kiểm metric, ngưỡng, nhãn hiếm, lỗi và dự đoán |
| [06 · EDA bổ sung](notebooks/06_eda_extra.ipynb) | [Mở kết quả 06](notebooks/executed/06_eda_extra.executed.ipynb) | EDA độc lập; tokenizer ghim revision; đồng xuất hiện dùng int64 tránh tràn uint8 |

Mở Jupyter từ gốc repo: `python -m jupyterlab`, chọn kernel của `.venv`.
Hoặc chạy notebook 02 bằng Python hiện tại, sau bước train/analyze A:

```bash
python -m scripts.data.run_notebooks --notebook 02_baseline
```

Runner mặc định giữ source sạch, ghi output vào `reports/notebooks/`.
Để chạy toàn bộ và cập nhật bản có output dùng nộp/bảo vệ:

```powershell
& $py -m scripts.data.run_notebooks --notebook all --publish --artifacts-root ../goemotions-multilabel-classification
```

`$py` được khai báo ở phần tái lập đầu README. Trên máy khác dùng Python của môi
trường đã cài đủ extra `eda,notebooks,transformers` và trỏ `--artifacts-root` đến
thư mục có `data/` và `reports/` của các run đã báo cáo. Nếu mở Jupyter trực tiếp,
có thể đặt biến môi trường `GOEMOTIONS_ARTIFACTS_ROOT`; notebook tự tìm thư mục
code bằng các thư mục cha, kể cả khi mở bản trong `notebooks/executed/`.

Bản notebook công bố dùng checkpoint đối chiếu với bảng `reports/project_results/` của cùng lượt chạy. Manifest ghi thời điểm thực thi cell riêng với thời điểm huấn luyện.
**Run All không huấn luyện lại các mô hình benchmark từ pretrained.** Code train
và log từng epoch được trình bày; các cell đánh giá A/C gọi model thật trên 5.427
mẫu test và kiểm với bộ số báo cáo. Không tìm ngưỡng hoặc chọn lại model bằng test.

[Manifest thực thi](notebooks/executed/execution_manifest.json) gắn source với bản
có output bằng SHA-256 chuẩn hóa LF, số code cell và thời gian chạy UTC. CI kiểm
mọi source có bản chạy, cell thực thi tuần tự, không có error output và không lệch
source. Clone mới xem được output ngay; muốn Run All lại cần dataset/artifacts,
weights lớn không được nhúng vào Git.

Notebook 04/05 cần bộ model/score/protocol do pipeline chung tạo, gồm C3 tại
`data/processed/transformers/distilbert/seed_*/full/standard/`. Thiếu bộ nào sẽ dừng,
không thay bằng checkpoint C3 riêng. [Bảng số liệu và hình của báo cáo 2.1](results/report_2_1/)
vẫn được giữ làm snapshot lịch sử; trọng số lớn nằm trong bộ bàn giao cục bộ.

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

Ảnh giao diện cập nhật ngày **22/10/2026**, BERT seed 123 được chọn bằng validation.
Đây là minh chứng một lần chạy, không khẳng định app đang mở trên máy bạn.
Giao diện giữ bố cục đơn giản: chọn mô hình, nhập bình luận, bấm nhận diện rồi
xem nhãn và bảng điểm. Sáu lựa chọn dùng ngưỡng 0,5 khi khởi động mặc định.
Sau khi khôi phục bộ nộp và checkpoint như hướng dẫn đầu trang:

```bash
python -m scripts.transformers.verify_demo --device auto
python app.py --device auto
```

Minh chứng mới: [kiểm API sáu lựa chọn](results/final_submission/demo_api_recheck.json).
Mỗi lựa chọn trả đủ 28 nhãn; đầu vào rỗng được xử lý. Biên nhận này không thêm
ngày mới và giữ riêng [biên nhận lịch sử](results/final_submission/demo_verification.json).
Ảnh ngày 22/10/2026 là minh họa giao diện cũ; phiên cuối chưa kiểm lại trực quan.

Mở `http://127.0.0.1:7860/`. App cần checkpoint thật; clone source không có trọng số.
Ngưỡng mặc định 0,5; ngưỡng riêng cần `--thresholds` đúng protocol run đã chọn.

Giao diện có mục **Chọn mô hình** để thử BERT, RoBERTa, DistilBERT, hai baseline
standard/balanced và zero-shot BART-MNLI trên câu nhập. Mặc định vẫn là C được
chọn bằng validation trong `selected_model.json`. Các lựa chọn so sánh dùng
ngưỡng 0,5 theo mặc định. Kết quả hiển thị tên mô hình, seed của checkpoint C,
các nhãn được chọn và bảng điểm đủ 28 nhãn. Thông số huấn luyện, các thí nghiệm
chọn ngưỡng và kết quả benchmark được trình bày trong các mục thực nghiệm.
Nếu truyền `--thresholds`, tệp ngưỡng riêng chỉ áp dụng cho model mặc định.
RoBERTa và DistilBERT dùng seed tốt nhất theo validation trong bộ đã chốt.
Muốn xem riêng C3 của snapshot 2.1, dùng `--c3-source report-2-1` và cung cấp bundle
C3 lịch sử của báo cáo 2.1 (hoặc đường dẫn `GOEMOTIONS_C3_RUN_ROOT`).
Kết quả cũ được xóa khi đổi lựa chọn. Model được nạp khi bấm dự đoán và app chỉ
giữ một model trong bộ nhớ. BART-MNLI tải đúng revision đã ghi trong metadata
vào cache trên ổ chứa repo khi chưa có; lần đầu có thể cần tải trọng số lớn.
Các lựa chọn cần artifacts tương ứng trong `--artifacts-root`; thiếu model sẽ
báo lỗi, không dùng model khác để thay thế. Đổi model không huấn luyện lại hay
thay đổi số liệu benchmark đã công bố.

### Kiểm dữ liệu và chuẩn bị bảo vệ

```powershell
& $py -m scripts.verification.audit_data_boundaries --artifacts-root ../goemotions-multilabel-classification
```

[Biên nhận kiểm dữ liệu](results/presentation_audit/data_boundary_audit.json)
ghi kết quả dựng lại TF-IDF chỉ từ train: 58.338 đặc trưng, vocabulary và IDF
khớp cả hai baseline (sai lệch IDF lớn nhất bằng 0). ID ba tập không giao nhau,
nhưng train–test có 32 chuỗi trùng ảnh hưởng 37 dòng test; validation–test có
10 chuỗi trùng ảnh hưởng 13 dòng test. Không diễn giải kết quả này thành
chứng nhận không rò rỉ: chưa loại trừ gần trùng, cùng hội thoại hoặc pretraining,
và test đã được quan sát trong EDA/báo cáo trước.

Mỗi split được mã hóa riêng nhưng dùng chung mapping 28 nhãn; validation/test
chỉ transform bằng vectorizer đã fit trên train. Tokenizer pretrained được giữ
nhất quán theo từng kiến trúc. Mã hóa không bảo đảm giữ toàn bộ nội dung:
TF-IDF bỏ đặc trưng ngoài vocabulary, các C cắt ở 128 token. Log hiện tại có
train loss và validation Macro-F1, chưa có train F1/validation loss từng epoch.
Cả chín lượt đạt validation F1 cao nhất ở epoch cuối; chưa chứng minh hội tụ
hoặc hoàn toàn không overfit/underfit.

[Biên nhận demo nhiều mô hình](results/presentation_audit/multi_model_demo_verification.json)
kiểm HTTP, sáu mô hình ở ngưỡng 0,5 qua Gradio API và thao tác chọn mô hình,
nhập câu, bấm dự đoán trên trình duyệt độc lập. Ảnh giao diện mới nằm tại assets/.
Biên nhận demo trong snapshot báo cáo cũ chỉ kiểm hàm suy
luận và vẫn giữ nguyên phạm vi lịch sử của lần kiểm đó.

Các nghiên cứu để đối chiếu, không phải các kỹ thuật đều đã được chạy trong repo:

- [GoEmotions (ACL 2020)](https://aclanthology.org/2020.acl-main.372/): dữ liệu, taxonomy và hướng fine-tuning BERT được kế thừa.
- [SemEval-2018 Task 1](https://aclanthology.org/S18-1001/): bối cảnh cảm xúc trên tweet trước GoEmotions, khác bộ dữ liệu/hệ nhãn.
- [BERT và clipped asymmetric loss (IEEE Access 2025)](https://ieeexplore.ieee.org/document/10947681/): liên quan mất cân bằng và nhãn hiếm; chín lượt C của nhóm chưa dùng loss này.
- [Weighted loss trên BRIGHTER (SemEval 2025)](https://aclanthology.org/2025.semeval-1.37/): liên hệ kỹ thuật, không so điểm như cùng benchmark GoEmotions.
- [TACO (ACL 2025)](https://aclanthology.org/2025.acl-long.247/): học tương phản, nhưng thí nghiệm GoEmotions lọc mẫu một nhãn và bỏ neutral nên không so trực tiếp với bài toán 28 nhãn đa nhãn.
- [So sánh LR, BiLSTM và BERT (2026)](https://arxiv.org/abs/2601.18162): preprint gần phạm vi đồ án; repo hiện tại chưa có thí nghiệm BiLSTM.

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

<a id="huy-c3-test"></a>
#### Hồ sơ C3 lịch sử trước bộ nộp ngày 26/10/2026

Ba checkpoint full seed 42, 123, 2026 của Huy đã được đánh giá trên 5.427 mẫu
test từ checkpoint có sẵn. Checkpoint và ngưỡng được
chọn trên validation và khóa trước khi chạy test. Mean ± sample std (`ddof=1`):

| Ngưỡng | Macro-F1 test | Micro-F1 test |
|---|---:|---:|
| Cố định 0,5 | 0.4157 ± 0.0045 | 0.5756 ± 0.0025 |
| Riêng từng nhãn từ validation | 0.4913 ± 0.0091 | 0.5952 ± 0.0037 |

Các số ở mục lịch sử này khác bộ training_20261025 được chọn cho final; không dùng để thay hàng C3 ở đầu README. Chúng thuộc nghiên cứu bổ sung trước đó của Huy; không gộp vào bảng chính của báo cáo nhóm 2.1. BERT seed 123 vẫn được chọn cho demo
bằng validation. Test đã được nhóm quan sát trước trong EDA/báo cáo.

Tái kiểm kết quả và xuất bảng/hình từ checkpoint, scores đã lưu tại máy:

```bash
python -m scripts.distilbert.evaluate_c3_test --device cuda
python -m scripts.distilbert.export_c3_final_report
```

Lệnh đầu kiểm hash, khóa checkpoint/ngưỡng từ validation cho cả ba seed trước
khi đọc test, rồi lưu kết quả riêng tại
`data/processed/c3_distilbert/test_evaluation_20261021/`. Chạy lại kiểm kết quả
đã lưu; không huấn luyện lại và không sửa `run.json` lịch sử.
Lệnh thứ hai tạo `reports/c3_huy_final_20261021/`, thay toàn bộ C3 bằng kết quả
của Huy trong bảng/hình đối chiếu. A/B/C1/C2 được lấy từ các CSV đã lưu tại
commit `3c44309` (cần lịch sử Git này); không gộp thành sáu seed C3.

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
├── notebooks/            # Source sạch 00–06; executed/ giữ bản đã chạy và manifest
├── examples/             # Ví dụ CPU không cần Internet
├── results/report_2_1/   # Bảng/hình và biên nhận bộ số báo cáo 2.1
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

**Nhóm 4 người:** Duy — baseline A, zero-shot B, viết/tổng hợp báo cáo và điều phối;
Quốc Khánh — BERT C1; Đức Trí — RoBERTa C2;
Nhật Huy — DistilBERT C3, nghiên cứu riêng và demo.
Tỷ lệ đóng góp do nhóm xác nhận, không suy từ tác giả commit.

**Tác giả Git bản cập nhật:** Nguyễn Bảo Duy · `dzyuu1612` · `baoduynguyen1612@gmail.com`.
Nhóm chưa công bố giấy phép riêng cho source; dữ liệu/checkpoint/thư viện theo nguồn tương ứng.
Score chưa mặc nhiên là xác suất nghiệp vụ; áp dụng ngoài Reddit tiếng Anh cần kiểm chứng riêng.
