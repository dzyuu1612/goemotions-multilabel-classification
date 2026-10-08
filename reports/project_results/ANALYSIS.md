# Phân tích kết quả, nhãn hiếm và chi phí

## 1. Năm nhãn hiếm: trước và sau cải tiến

**Bảng tạm: VALIDATION — có calibration.** Test chưa đủ mọi cặp A/C1/C2/C3 nên chưa dùng để đưa ra bảng so sánh cuối. Các cấu hình tuned được đo trên cùng validation đã dùng để chọn ngưỡng; mức tăng có thể lạc quan, không được gọi là kết quả test. Các ô thiếu được ghi rõ.

Nhãn hiếm lấy theo năm support thấp nhất trên train từ metadata A standard; hòa theo ID nhãn. A: standard fixed → balanced tuned. C1/C2/C3: cùng kiến trúc, cùng ít nhất ba seed, fixed → tuned. C báo mean ± sample std (`ddof=1`); Δ tính theo cặp seed. A là một run, không tạo std bằng 0.

| Nhãn | Mô hình | Train + | Validation + | Test + | F1 trước | F1 sau | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| grief | A: TF-IDF + LR | 77 | 13 | Chưa có/không khớp | 0.0000 | 0.4375 | +0.4375 |
| grief | C1: BERT | 77 | 13 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| grief | C2: RoBERTa | 77 | 13 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| grief | C3: DistilBERT | 77 | 13 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| pride | A: TF-IDF + LR | 111 | 15 | Chưa có/không khớp | 0.0000 | 0.6087 | +0.6087 |
| pride | C1: BERT | 111 | 15 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| pride | C2: RoBERTa | 111 | 15 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| pride | C3: DistilBERT | 111 | 15 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| relief | A: TF-IDF + LR | 153 | 18 | Chưa có/không khớp | 0.0000 | 0.1739 | +0.1739 |
| relief | C1: BERT | 153 | 18 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| relief | C2: RoBERTa | 153 | 18 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| relief | C3: DistilBERT | 153 | 18 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| nervousness | A: TF-IDF + LR | 164 | 21 | Chưa có/không khớp | 0.0000 | 0.3125 | +0.3125 |
| nervousness | C1: BERT | 164 | 21 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| nervousness | C2: RoBERTa | 164 | 21 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| nervousness | C3: DistilBERT | 164 | 21 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| embarrassment | A: TF-IDF + LR | 303 | 35 | Chưa có/không khớp | 0.1081 | 0.5507 | +0.4426 |
| embarrassment | C1: BERT | 303 | 35 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| embarrassment | C2: RoBERTa | 303 | 35 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |
| embarrassment | C3: DistilBERT | 303 | 35 | Chưa có/không khớp | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả |

Support là số câu có nhãn thật, không phải số lần model dự đoán nhãn. Δ > 0 là tăng, Δ < 0 là giảm. Bảng làm tròn bốn chữ số; số gốc và mọi mức giảm được giữ trong `rare_before_after.csv`. Các nhóm C thiếu seed/config/revision nhất quán sẽ không có mean/std.

## 2. Chi phí thực nghiệm C

Lấy `elapsed_seconds` từ metadata của từng run full hoàn tất. Đây là thời gian toàn run C (chuẩn bị dữ liệu/model, train, validation và ghi checkpoint), không phải riêng thời gian optimizer. Tổng và trung bình chỉ hiển thị khi đủ tập seed yêu cầu; chi phí phụ thuộc máy và cache.

| Kiến trúc | Full seeds đủ chi phí | Số tham số | Tổng elapsed (giây) | Mean elapsed/run (giây) | Trạng thái |
|---|---|---:|---:|---:|---|
| bert | 1/3 | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả | Thiếu run/metadata |
| roberta | 0/3 | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả | Thiếu run/metadata |
| distilbert | 0/3 | Thiếu kết quả | Thiếu kết quả | Thiếu kết quả | Thiếu run/metadata |

Nguồn chi tiết: `training_costs.csv`. Phép đo chi phí B cần tổng thời gian suy luận có xử lý resume; không dùng thời gian của một lần tiếp tục để đại diện toàn dataset.

## 3. Learning curves trên validation

**Chưa có hình đủ ba kiến trúc × các seed:** xem `learning_curves.csv` và `missing_artifacts.csv`; không dựng đường giả hoặc thay model thiếu bằng điểm 0.

## 4. Hồ sơ đối chiếu

- `per_label.csv`: P/R/F1 và TP/FP/FN/support cho từng nhãn, split, cấu hình và seed.
- `aggregate_metrics.csv`: Macro/Micro-F1 fixed/tuned mean ± sample std của ba C.
- `rare_before_after.csv`: nhãn hiếm, cặp trước/sau và Δ, giữ toàn bộ tăng/giảm.
- `learning_curves.csv`, `training_costs.csv`: epoch và chi phí từ metadata thật.
- `missing_artifacts.csv`, `analysis_manifest.json`: trạng thái thiếu/không hợp lệ và phạm vi dữ liệu.

Script chỉ đọc artifact và kết quả đã lưu; không huấn luyện, chọn lại ngưỡng hoặc mở nhãn test.