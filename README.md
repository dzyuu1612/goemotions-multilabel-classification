# GoEmotions · Phân loại cảm xúc đa nhãn

**Nhóm 6 · HUFLIT · Báo cáo IEEE v8 rút gọn**

Một bình luận có thể diễn đạt nhiều cảm xúc cùng lúc. Đồ án dự đoán tập nhãn trong 27 cảm xúc và nhãn `neutral` trên GoEmotions, rồi so sánh năm hệ trên cùng bộ test chính thức.

[**Mở notebook đã chạy**](notebooks/GoEmotions_ToanBo_QuyTrinh.ipynb) · [Các bảng CSV](#doi-chieu-bao-cao) · [Chạy demo](#chay-demo)

## Kết quả chính

Macro-F1 dưới đây được tính trên **5.427 câu test**. A và B có một lượt. Ba mô hình C báo trung bình ± độ lệch chuẩn mẫu của ba seed 42, 123 và 2026.

| Hệ | Phương pháp | Ngưỡng 0,5 | Ngưỡng chung | Ngưỡng riêng |
|---|---|---:|---:|---:|
| A | TF-IDF + Logistic Regression cân bằng | 0,4441 | **0,4530** | 0,4493 |
| B | BART-MNLI zero-shot | 0,1035 | 0,1473 | 0,1609 |
| C1 | BERT | 0,4696 ± 0,0055 | 0,4794 ± 0,0038 | 0,4938 ± 0,0075 |
| C2 | RoBERTa | 0,4765 ± 0,0079 | 0,4985 ± 0,0121 | **0,5101 ± 0,0076** |
| C3 | DistilBERT | 0,4729 ± 0,0039 | 0,4883 ± 0,0055 | 0,4972 ± 0,0027 |

RoBERTa đạt Macro-F1 trung bình cao nhất trong các cấu hình đã đo. Với checkpoint RoBERTa seed 2026, chuyển từ 0,5 sang ngưỡng riêng giúp giảm **903 lượt bỏ sót nhãn** và tăng **1.765 lượt gán nhãn thừa**. Bảng ba chế độ có đủ Macro-F1, Micro-F1, Hamming Loss và các chỉ số bổ sung.

[CSV ba chế độ ngưỡng](experiments/threshold_error_20261110/results/modes_mean_std.csv) · [CSV từng checkpoint](experiments/threshold_error_20261110/results/modes_per_checkpoint.csv)

## Chạy notebook

Cần Python **3.12 hoặc 3.13**. Các lệnh dưới dùng PowerShell.

```powershell
git clone https://github.com/dzyuu1612/goemotions-multilabel-classification.git
cd goemotions-multilabel-classification
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[eda,transformers,notebooks]"
.\.venv\Scripts\python -m scripts.data.run_notebooks --notebook all
```

Lệnh chạy toàn bộ cell và lưu bản chạy mới trong `reports/notebooks/`. Nếu dùng JupyterLab, chạy `.\.venv\Scripts\python -m jupyterlab`, mở notebook trong thư mục `notebooks/`, rồi chọn **Restart Kernel → Run All**. Lần đầu cần Internet để lấy dữ liệu, tokenizer và mô hình A đã fit.

Notebook có sẵn bảng, đáp án tính tay và biểu đồ. Phần A suy luận lại từ mô hình đã fit. Phần B và C tính lại chỉ số từ điểm dự đoán đã lưu. Phần cuối đối chiếu độ trễ từ thời gian từng lô đã ghi. Các bước này chạy trên CPU và không cần tải trọng số Transformer.

<a id="doi-chieu-bao-cao"></a>

## Đối chiếu báo cáo IEEE v8 rút gọn

| Nội dung | Tệp kiểm tra |
|---|---|
| Ba seed của BERT, RoBERTa và DistilBERT | [Điểm từng lượt](experiments/best_epochs_20261103/main_results/report_data/all_runs.csv) · [Trung bình và độ lệch chuẩn](experiments/best_epochs_20261103/main_results/report_data/mean_std.csv) |
| Chọn epoch và chẩn đoán overfit | [Lịch sử 72 epoch](experiments/best_epochs_20261103/sweep_log.csv) · [Đối chiếu train/validation](experiments/best_epochs_20261103/overfit_audit/overfit_comparison.csv) |
| Bảng 5.7, Phụ lục F · chọn ngưỡng `approval` | [Đủ 19 mức validation](experiments/report_v8/results/approval_thresholds.csv) |
| Bảng 5.8 · TP, FP, FN của ba checkpoint seed 2026 | [Số đếm test](experiments/report_v8/results/test_fp_fn.csv) |
| Bảng 5.9 và phân tích lỗi | [F1 từng nhãn](experiments/best_epochs_20261103/main_results/report_data/per_label_test.csv) · [Cặp FN → FP](experiments/threshold_error_20261110/results/pair_counts_long.csv) |
| Bảng 5.19 · cơ cấu mẫu khảo sát | [48 mục `neutral → approval` và 2 mục `relief → admiration`](experiments/report_v8/results/annotation_sample.csv) |
| Bảng 5.21, Phụ lục G · đo suy luận | [Kết quả 5 hệ](experiments/report_v8/results/inference_summary.csv) · [Thời gian từng lô](experiments/report_v8/results/inference_timings.csv) · [Cấu hình phép đo](experiments/report_v8/results/inference_setup.json) |
| Bảng D.1–D.2 · ngưỡng đã khóa | [CSV đủ 28 nhãn của A/B và ba seed mỗi C](experiments/threshold_error_20261110/results/thresholds_28.csv) |
| Bảng E.1–E.3 · validation theo epoch | [Macro-F1 tại ngưỡng 0,5](experiments/threshold_error_20261110/results/epoch_table_val_macro_f1.csv) · [BCE](experiments/threshold_error_20261110/results/epoch_table_val_bce.csv) |

Ở RoBERTa seed 2026, `approval` đạt F1 validation cao nhất **0,4054 tại ngưỡng 0,25** trong 19 mức đã thử. Ba C chọn checkpoint bằng Macro-F1 validation tại 0,5, sau đó chọn ngưỡng riêng mỗi nhãn trên lưới từ 0,05 đến 0,95.

Hướng phát triển ở mục 6.3 là thử BCE có trọng số cho nhãn hiếm trên RoBERTa. Đối chiếu với BCE thường bằng F1 của năm nhãn hiếm, số nhãn gán thừa và Macro-F1 của cả 28 nhãn.

<a id="chay-demo"></a>

## Chạy demo

Khôi phục A/B và tải [gói checkpoint C đại diện](https://github.com/dzyuu1612/goemotions-multilabel-classification/releases/tag/goemotions-best-epochs-20261103), sau đó chạy ứng dụng từ thư mục gốc.

```powershell
.\.venv\Scripts\python -c "from src.final_submission import restore_baselines; restore_baselines()"
.\.venv\Scripts\python experiments/best_epochs_20261103/restore_best_models.py
.\.venv\Scripts\python app.py --device auto
```

Mở [demo cục bộ](http://127.0.0.1:7861), nhập câu tiếng Anh và bấm **Nhận diện cảm xúc**. Mặc định là **RoBERTa, seed 2026, ngưỡng riêng**.

Để tái hiện Hình 5.17, nhập `I was worried, but your help made me feel so relieved.`. Điểm `relief` là **0,2304**, đạt ngưỡng riêng **0,05**. Ở chế độ 0,5, câu này cho tập nhãn rỗng.

## Mã nguồn và kiểm tra

- [Huấn luyện A](scripts/baseline/run_baseline.py), [suy luận B](scripts/zero_shot/run_zero_shot.py) và [huấn luyện C](experiments/best_epochs_20261103/train_transformer.py)
- [Cấu hình và lệnh chạy C](experiments/best_epochs_20261103/README.md)
- [Kiểm ngưỡng và phân tích sai](experiments/threshold_error_20261110/analysis.py)
- [Cách tái hiện phép đo suy luận](experiments/report_v8/README.md)
- [Bài báo GoEmotions](https://aclanthology.org/2020.acl-main.372/) và [dữ liệu chính thức](https://github.com/google-research/google-research/tree/master/goemotions)

Sau khi cài phụ thuộc, chạy các lệnh sau để kiểm số và cấu trúc repo.

```powershell
.\.venv\Scripts\python tools/check_repository.py
.\.venv\Scripts\python -m experiments.report_v8.verify_tables
.\.venv\Scripts\python experiments/best_epochs_20261103/verify_report_tables.py
```

`src/` và `scripts/` chứa mã dùng chung. `experiments/` giữ cấu hình, điểm dự đoán và CSV làm căn cứ cho báo cáo. `results/final_submission/` chứa gói dữ liệu nhỏ cần để tính lại kết quả A/B. Trọng số mô hình tải từ GitHub Releases theo lệnh demo.
