# GoEmotions — phân loại cảm xúc đa nhãn

**Nguyễn Bảo Duy — Nhóm 6, HUFLIT.** Mã nguồn và notebook của báo cáo Final v7: A (TF-IDF + Logistic Regression), B (BART-MNLI), C1 (BERT), C2 (RoBERTa), C3 (DistilBERT). Bản v7 bổ sung phân tích ngưỡng và lỗi từ điểm đã lưu, không huấn luyện lại.

## Notebook đã chạy

Mỗi notebook chỉ giữ một bản có đầu ra thật, không có bản `.executed` trùng lặp.

| Nội dung | Notebook |
|---|---|
| Ví dụ baseline nhỏ trên CPU | [00_quick_start](notebooks/00_quick_start.ipynb) |
| Phân tích dữ liệu GoEmotions | [01_eda](notebooks/01_eda.ipynb) |
| A — TF-IDF và Logistic Regression | [02_baseline](notebooks/02_baseline.ipynb) |
| B — BART-MNLI, tính lại từ điểm đã lưu | [03_zero_shot](notebooks/03_zero_shot.ipynb) |
| **C1/C2/C3 — đủ 9 lượt, kết quả Final v6** | [GoEmotions_KetQua_EpochTotNhat](experiments/best_epochs_20261103/GoEmotions_KetQua_EpochTotNhat.ipynb) |
| **Final v7 — ba chế độ ngưỡng, năm loại lỗi, giao nhóm, near-miss** | [GoEmotions_PhanTich_Nguong_Loi_DaChay](experiments/threshold_error_20261110/GoEmotions_PhanTich_Nguong_Loi_DaChay.ipynb) |
| Kiểm dữ liệu và tokenizer | [06_eda_extra](notebooks/06_eda_extra.ipynb) |
| Tính tay attention, sigmoid, BCE, gradient | [07_improvement_workflow](notebooks/07_improvement_workflow.ipynb) |

Notebook C tính lại 18 hàng test × 7 chỉ số từ điểm dự đoán đã lưu; không huấn luyện lại hoặc chọn ngưỡng trên test. Notebook Final v7 kiểm thêm đủ 33 bộ metric, ngưỡng validation và cách đếm lỗi. Các notebook A/B và EDA giữ kết quả đã chạy. `notebooks/execution_manifest.json` kiểm mã băm, thứ tự thực thi và lỗi của toàn bộ tám notebook.

[Hướng dẫn phân tích theo nhận xét của giảng viên](experiments/threshold_error_20261110/README.md) · [Kế hoạch đã ghi](experiments/threshold_error_20261110/DECISION_PHANTICH.md) · [Bảng so ba chế độ](experiments/threshold_error_20261110/results/modes_mean_std.csv). Phần gán nhãn nguyên nhân cần hai người độc lập; các phiếu vẫn để trống, chưa công bố tỷ lệ đồng thuận.

## Cài đặt

Python 3.12–3.13; cài PyTorch phù hợp máy trước nếu cần CUDA khác.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[eda,transformers,notebooks]"
```

Môi trường đã chạy: Python 3.13.9, PyTorch 2.13.0+cu130, Transformers 4.57.6, RTX 5060 Laptop.

## Kết quả dùng trong báo cáo

Bộ chia chính thức: **43.410 train / 5.426 validation / 5.427 test**, 28 nhãn. Cả ba C dùng LR `2e-5`, batch 16, max_length 128, tối đa 8 epoch và ba seed 42/123/2026. Epoch chọn: **BERT 7/6/7; RoBERTa 7/8/5; DistilBERT 7/8/8**. Chọn checkpoint bằng Macro-F1 validation @0,5, hòa chọn epoch sớm; khóa checkpoint và ngưỡng trước khi chấm test.

| Mô hình · ngưỡng riêng | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| BERT | 0,4938 ± 0,0075 | 0,5717 ± 0,0051 | 0,0405 ± 0,0016 |
| RoBERTa | 0,5101 ± 0,0076 | 0,5909 ± 0,0033 | 0,0377 ± 0,0002 |
| DistilBERT | 0,4972 ± 0,0027 | 0,5779 ± 0,0069 | 0,0389 ± 0,0012 |

Trung bình và std mẫu (`ddof=1`) từ đủ ba seed. RoBERTa cao nhất trung bình trong cấu hình đã đo; chưa kiểm định thống kê. C2/C3 tăng Macro-F1 so với lượt ban đầu, nhưng Micro-F1 cả ba giảm và Hamming Loss tăng. BERT còn đổi LR so với lượt cũ. Train BCE giảm trong khi validation BCE tăng ở epoch sau: vẫn có dấu hiệu overfit theo loss. Test cũ đã được xem trước khi đổi lịch; cần holdout mới để kết luận mạnh hơn.

[CSV từng seed](experiments/best_epochs_20261103/main_results/report_data/all_runs.csv) · [Mean/std](experiments/best_epochs_20261103/main_results/report_data/mean_std.csv) · [Lịch sử 72 epoch](experiments/best_epochs_20261103/sweep_log.csv) · [Bảng lịch sử B.3](experiments/best_epochs_20261103/main_results/report_data/original_test_summary.csv)

Kiểm lại số liệu mà không huấn luyện:

```powershell
python tools/check_repository.py
python experiments/best_epochs_20261103/verify_report_tables.py
python -m unittest discover -s tests -v
```

## Demo và huấn luyện

```powershell
python -c "from src.final_submission import restore_baselines; restore_baselines()"
python experiments/best_epochs_20261103/restore_best_models.py
python app.py --device auto
```

Demo nạp checkpoint đã lưu, mặc định **RoBERTa/2026, ngưỡng riêng**, chạy cục bộ tại `http://127.0.0.1:7861`. Chưa có URL demo công khai. [Gói checkpoint tải riêng](https://github.com/dzyuu1612/goemotions-multilabel-classification/releases/tag/goemotions-best-epochs-20261103) được kiểm SHA-256; không đưa weights lớn vào cây mã nguồn.

Huấn luyện C1/C2/C3 dùng [trainer đã chạy](experiments/best_epochs_20261103/train_transformer.py); lệnh và cấu hình trong [README thí nghiệm](experiments/best_epochs_20261103/README.md). Seed không bảo đảm GPU huấn luyện lại trùng từng bit; notebook dùng điểm đã lưu để tái hiện bảng.

## Thư mục

- `src/`, `scripts/`: xử lý dữ liệu, huấn luyện, đánh giá và ứng dụng.
- `notebooks/`, `experiments/best_epochs_20261103/`: notebook có kết quả, trainer và số liệu hiện tại.
- `tests/`, `examples/`, `configs/`, `requirements/`: kiểm thử, ví dụ và cấu hình.
- `results/final_submission/`: bốn tệp cần để khôi phục A/B và đối chiếu số lịch sử. Gói runtime chứa dữ liệu/điểm đã lưu để notebook và demo chạy được; không phải bản Word/PPT hay bản code thứ hai.

Đã bỏ các thư mục báo cáo sinh tự động, ảnh minh họa cũ, triển khai demo cũ, kết quả thử cũ và notebook trùng. Các bản trước còn trong lịch sử Git. Giấy phép/model card của các mô hình nền được giữ trong `experiments/best_epochs_20261103/model_attribution/`.

[Bài báo GoEmotions](https://aclanthology.org/2020.acl-main.372/) · [Dữ liệu chính thức](https://github.com/google-research/google-research/tree/master/goemotions)
