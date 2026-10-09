# Kiểm tra kỹ thuật DistilBERT của Nhật Huy

**Đây là pilot, chưa phải kết quả C3 chính thức hoặc mô hình demo cuối.**

- Dữ liệu: 1024 train / 256 validation, lấy mẫu bằng seed 2026.
- Huấn luyện: seed 42, 1 epoch; 64 bước cập nhật.
- Batch vật lý 4, tích lũy 4; batch hiệu dụng 16.
- Checkpoint: `distilbert/distilbert-base-uncased`; revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU; CUDA 12.8.
- Thời gian train và validation: 46.5 giây (không gồm tải model).
- CUDA memory allocated cực đại: 1348.6 MiB; reserved: 1450.0 MiB.
- Chênh lệch score lớn nhất sau lưu/nạp checkpoint: 0.
- Không đọc test. Không chọn ngưỡng, không báo mean/std từ một seed.

| Chỉ số trên validation PILOT, ngưỡng 0,5 | Giá trị |
|---|---:|
| Macro-F1 | 0.000000 |
| Micro-F1 | 0.000000 |
| Hamming Loss | 0.041295 |

Số mẫu không có nhãn dự đoán đạt ngưỡng 0,5: 256/256.
Đây là kết quả quan sát của checkpoint pilot; không tự đổi ngưỡng để tạo dự đoán đẹp hơn.

Mục đích là kiểm luồng dữ liệu → GPU → loss đa nhãn → checkpoint → scores theo ID → metric chung → suy luận.
Không so các số pilot này trực tiếp với baseline full hoặc dùng để kết luận DistilBERT tốt/xấu hơn BERT/RoBERTa.
Một số nhãn hiếm có thể không xuất hiện trong mẫu pilot; bảng vẫn tính đủ 28 nhãn với zero_division=0.
Để hoàn thành C3, cần thống nhất cấu hình với nhóm, chạy đủ full train/validation cho ít nhất 3 seed,
phân tích lỗi và nâng cao theo kế hoạch. Demo cuối phải dùng C thắng theo validation.

Xem `pilot_run.json` (hash, môi trường, config), `pilot_metrics.json`, `pilot_history.csv`, `pilot_per_label.csv`.
Checkpoint và scores lớn nằm ở `data/processed/c3_distilbert/pilot/seed_42/`, không commit Git.
