# GoEmotions — phân loại cảm xúc đa nhãn

Mã nguồn đồ án GoEmotions, gồm TF-IDF + Logistic Regression (A), zero-shot BART-MNLI (B), BERT (C1), RoBERTa (C2) và DistilBERT (C3). Dữ liệu dùng bộ chia chính thức: **43.410 train / 5.426 validation / 5.427 test**, 28 nhãn.

## Bộ kết quả hiện tại — báo cáo Final v6

[Bộ mã, số liệu và notebook của báo cáo chính](experiments/best_epochs_20261103/README.md) nằm ở `experiments/best_epochs_20261103/`.

- Cả ba C dùng learning rate `2e-5`, batch 16, max_length 128, ngân sách tám epoch; mỗi kiến trúc có ba seed 42, 123, 2026.
- Chọn checkpoint từng seed bằng Macro-F1 validation @0,5 cao nhất; hòa lấy epoch sớm hơn. Checkpoint và ngưỡng khóa trước đánh giá test.
- Epoch chọn theo thứ tự seed: **BERT 7/6/7; RoBERTa 7/8/5; DistilBERT 7/8/8**.
- A/B giữ nguyên. Bộ số cũ trong `results/final_submission/` dùng để đối chiếu lịch sử, không phải kết quả C của báo cáo hiện tại.

| Mô hình | Macro-F1 test, ngưỡng riêng | Micro-F1 test, ngưỡng riêng | Hamming Loss |
|---|---:|---:|---:|
| BERT | 0,4938 ± 0,0075 | 0,5717 ± 0,0051 | 0,0405 ± 0,0016 |
| RoBERTa | 0,5101 ± 0,0076 | 0,5909 ± 0,0033 | 0,0377 ± 0,0002 |
| DistilBERT | 0,4972 ± 0,0027 | 0,5779 ± 0,0069 | 0,0389 ± 0,0012 |

Mean và sample std (`ddof=1`) từ đủ ba seed, tính trên số chưa làm tròn. So với lượt trước, Macro-F1 C2/C3 tăng, nhưng Micro-F1 cả ba giảm và Hamming Loss tăng. BERT giảm cả Macro-F1 và Micro-F1. Đường BCE validation tăng trong khi train giảm ở các epoch sau: không kết luận hết overfit chỉ vì F1 tăng. BERT trước dùng bốn epoch/LR5e-5; đây không phải đối chứng BERT ba epoch chỉ thay số epoch. Test từng được xem ở báo cáo trước, không gọi là holdout mới.

## Vì sao dùng cấu hình chung

Bảng chính giữ cùng LR `2e-5` và ngân sách tám epoch cho ba kiến trúc; không chọn cấu hình tốt nhất riêng cho từng mô hình. Kế hoạch tám epoch kiểm tra giả thuyết C2/C3 ở ba epoch học chưa đủ và theo dõi cả F1/BCE; không chứng minh tám epoch tối ưu chung. Vì vậy BERT hiện tại có Macro-F1 ngưỡng riêng 0,4938, thấp hơn 0,5038 ở lượt bốn epoch/LR5e-5.

Lịch tám epoch được quyết định sau khi test của lượt ban đầu đã được xem. Checkpoint, đại diện và ngưỡng của bộ hiện tại khóa bằng validation trước test. Vẫn cần holdout mới chưa quan sát. [Bảng lịch sử B.3](experiments/best_epochs_20261103/main_results/report_data/original_test_summary.csv) giữ nguyên sáu dòng test của lượt cũ; [hồ sơ quyết định](experiments/best_epochs_20261103/protocol_history/DECISION_SWEEP.md) được ghi trước sáu lượt C2/C3. Lượt BERT tám epoch đã có trước hồ sơ này.

## Cài đặt

Python 3.12 hoặc 3.13. Môi trường chạy thực tế: Python 3.13.9, PyTorch 2.13.0+cu130, Transformers 4.57.6, RTX 5060 Laptop. Cài PyTorch phù hợp GPU/CPU trước nếu máy cần một bản CUDA khác.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[eda,transformers,notebooks]"
```

## Xem kết quả, không huấn luyện lại

Mở [notebook có sẵn đầu ra](experiments/best_epochs_20261103/GoEmotions_KetQua_EpochTotNhat.ipynb). Notebook đọc điểm dự đoán thật, tính lại 18 hàng test × 7 chỉ số, đối chiếu CSV, tính mean/std và hiện biểu đồ học, kết quả và chẩn đoán overfit.

- [Kết quả từng seed](experiments/best_epochs_20261103/main_results/report_data/all_runs.csv)
- [Mean và std](experiments/best_epochs_20261103/main_results/report_data/mean_std.csv)
- [So sánh trước–sau](experiments/best_epochs_20261103/main_results/report_data/test_before_after.csv)
- [Lịch sử 72 epoch](experiments/best_epochs_20261103/sweep_log.csv)
- [Đối chiếu overfit](experiments/best_epochs_20261103/overfit_audit/overfit_comparison.csv)
- [Biên nhận khóa trước test](experiments/best_epochs_20261103/main_results/LOCK_BEFORE_TEST.json)

Kiểm các bảng bằng một lệnh, không mở Jupyter và không train:

```powershell
python experiments/best_epochs_20261103/verify_report_tables.py
```

Lệnh tính lại 18 hàng test × 7 chỉ số, kiểm mean/std của ba seed và sáu dòng lịch sử (Bảng B.3).

## Demo Gradio bằng checkpoint đã lưu

Từ thư mục gốc repository:

```powershell
python -c "from src.final_submission import prepare_runtime; prepare_runtime()"
python experiments/best_epochs_20261103/restore_best_models.py
python experiments/best_epochs_20261103/app_best_epochs.py --device auto
```

Demo hiện chạy cục bộ, chưa có URL demo công khai để truy cập từ máy khác. Mở `http://127.0.0.1:7861` trên máy chạy ứng dụng. Mặc định **RoBERTa seed 2026 / ngưỡng riêng**, chọn bằng validation. Có hai chế độ ngưỡng và sáu lựa chọn A/B/C. Không có nhãn đạt ngưỡng thì trả tập rỗng, không tự thêm neutral. Điểm chưa được hiệu chuẩn thành độ tin cậy tuyệt đối. CPU có thể cho điểm khác AMP GPU gần ngưỡng.

[Một gói ZIP](https://github.com/dzyuu1612/goemotions-multilabel-classification/releases/tag/goemotions-best-epochs-20261103) chứa ba checkpoint C đại diện, tokenizer và giấy phép nguồn; script khôi phục kiểm SHA-256. Không phải ensemble. ZIP còn giữ checkpoint BERT epoch 3 để bảo toàn biên nhận chẩn đoán của lượt gốc. Cả chín lượt được tái hiện đánh giá bằng điểm dự đoán trong repository.

## Huấn luyện khi cần kiểm quy trình

Xem lệnh và giới hạn tái lập tại [hướng dẫn bộ hiện tại](experiments/best_epochs_20261103/README.md). Trainer công bố khớp byte với trainer đã chạy; các module dùng chung được kiểm mã băm. Seed không bảo đảm huấn luyện GPU trùng từng bit; notebook dùng điểm đã lưu để tái hiện bảng báo cáo. Không chạy huấn luyện khi mở notebook kết quả hoặc demo.

[GoEmotions — bài báo gốc](https://aclanthology.org/2020.acl-main.372/) và [dữ liệu chính thức](https://github.com/google-research/google-research/tree/master/goemotions). Giấy phép và model card đúng revision của ba mô hình nền nằm trong `experiments/best_epochs_20261103/model_attribution/`.
