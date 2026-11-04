# GoEmotions — checkpoint tốt nhất trong ngân sách tám epoch

Bộ kết quả dùng BERT, RoBERTa và DistilBERT, mỗi kiến trúc ba seed 42, 123, 2026. Cả ba C dùng learning rate `2e-5`, batch 16, max_length 128, AdamW, weight_decay 0.01, warmup_ratio 0.1, AMP fp16 và ngân sách tám epoch. Mỗi seed chọn checkpoint có Macro-F1 validation tại 0.5 cao nhất; hòa thì lấy epoch sớm hơn.

Checkpoint, lựa chọn mô hình và các ngưỡng được khóa bằng validation trước khi chấm test. Không chọn lại theo test. A/B giữ kết quả trước đó. Test đã từng được quan sát ở phiên bản trước, nên bộ số này không được mô tả như một holdout mới.

## Xem và tính lại kết quả

Mở `GoEmotions_KetQua_EpochTotNhat.ipynb`. Notebook có sẵn đầu ra và tính lại F1, Precision, Recall, Hamming Loss từ đủ chín bộ điểm test đã lưu. Nó đối chiếu CSV, tính mean và sample std với `ddof=1`, rồi hiện các biểu đồ. Không cần huấn luyện lại để xem hoặc tái hiện đánh giá.

Các số gốc và biên nhận nằm trong `main_results/`; đường học đủ 72 epoch nằm trong `sweep_log.csv`. `LOCK_BEFORE_TEST.json` ràng buộc từng checkpoint, epoch và protocol bằng SHA-256. `selected` trong CSV đánh dấu đúng một epoch cho từng mô hình/seed.

Lịch sử 72 epoch nằm trong `sweep_log.csv`; log gốc của sáu lượt C2/C3 nằm trong `training_logs/`. Biên nhận đủ chín lượt nằm trong `main_results/training_receipts/`.

## Demo bằng mô hình đã lưu

Từ thư mục gốc repository, cài các phụ thuộc theo README chính. Khôi phục runtime A/B đã công bố, sau đó tải gói ba checkpoint C đại diện:

```powershell
python -c "from src.final_submission import restore_baselines; restore_baselines()"
python experiments/best_epochs_20261103/restore_best_models.py
python experiments/best_epochs_20261103/app_best_epochs.py --device auto
```

Giao diện nạp checkpoint và tokenizer khi cần; không train lại. Một gói ZIP chứa ba C đại diện, mỗi kiến trúc một seed được chọn bằng validation. Đây không phải ensemble ba seed. Kết quả đủ chín seed trong báo cáo được tái hiện bằng các điểm dự đoán đã lưu trong notebook.

`model_attribution/` giữ model card ở đúng revision và văn bản giấy phép nguồn của các mô hình nền; thư mục này cũng nằm trong gói ZIP.

## Huấn luyện từ đầu khi cần kiểm quy trình

`train_transformer.py` là bản mã đã thực sự dùng cho lượt quét. Ví dụ chạy một seed từ thư mục gốc repository:

```powershell
python -m experiments.best_epochs_20261103.train_transformer --architecture roberta --seed 42 --revision e2da8e2f811d1448a5b465c236feacd80ffbac7b --epochs 8 --learning-rate 2e-5 --batch-size 16 --max-length 128 --diagnostics --tag sweep8 --no-test --run-dir artifacts/epoch8/roberta/seed42
```

Trainer chỉ nạp train/validation, ghi đầy đủ F1/BCE ở eval mode cho mỗi epoch và lưu checkpoint tốt nhất. Test thuộc bước đánh giá riêng sau khi khóa. Đường học giữ đủ tám epoch, kể cả các epoch không được chọn. Epoch 3 trong lịch 8 không tương đương một lượt chỉ chạy 3 epoch vì lịch learning rate phụ thuộc tổng số bước.

Phép đối chiếu BERT với lượt trước thay cả learning rate và số epoch; không gọi đó là đối chứng chỉ thay epoch. Không hứa kết quả tăng và không bỏ seed có kết quả thấp.

Log C3 ghi nhận cảnh báo attention GPU có phép tính không tất định. Cố định seed không bảo đảm một lượt huấn luyện lại trùng từng bit với lượt đã lưu. Notebook tái hiện đánh giá dùng các điểm dự đoán đã lưu và kiểm chúng với CSV của báo cáo.

## Kết quả trước và sau

`main_results/report_data/test_before_after.csv` giữ mean/std và chênh lệch Macro-F1, Micro-F1, Hamming Loss so với lượt ban đầu. Macro-F1 C2/C3 tăng, nhưng Micro-F1 cả ba giảm và Hamming Loss tăng. BERT giảm cả hai F1; phép đối chiếu BERT còn đổi learning rate. Không dùng bảng test này để chọn lại checkpoint.

## Đồng bộ báo cáo Final v6

Bảng lịch sử ở `main_results/report_data/original_test_summary.csv` khớp CSV của lượt cũ. `report_context.json` giải thích cấu hình chung và việc test cũ đã được xem. `protocol_history/` giữ nguyên hồ sơ trước sáu lượt C2/C3, không áp hồi tố cho lượt BERT8 đã hoàn tất trước đó.

Chạy `python experiments/best_epochs_20261103/verify_report_tables.py` từ gốc repo để kiểm từ điểm dự đoán. Demo là ứng dụng cục bộ; chưa có URL demo công khai. Notebook đã chạy, có đầu ra sẵn.
