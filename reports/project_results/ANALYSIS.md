# Phân tích kết quả, nhãn hiếm và chi phí

## 1. Năm nhãn hiếm: trước và sau cải tiến

**Bảng chính: TEST.** Ngưỡng của các cấu hình được khóa trên validation trước khi đánh giá test. Bảng giữ cả mức tăng, giảm và không đổi.

Nhãn hiếm lấy theo năm support thấp nhất trên train từ metadata A standard; hòa theo ID nhãn. A: standard fixed → balanced tuned. C1/C2/C3: cùng kiến trúc, cùng ít nhất ba seed, fixed → tuned. C báo mean ± sample std (`ddof=1`); Δ tính theo cặp seed. A là một run, không tạo std bằng 0.

| Nhãn | Mô hình | Train + | Validation + | Test + | F1 trước | F1 sau | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| grief | A: TF-IDF + LR | 77 | 13 | 6 | 0.0000 | 0.4615 | +0.4615 |
| grief | C1: BERT | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0444 ± 0.0770 | +0.0444 ± 0.0770 |
| grief | C2: RoBERTa | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| grief | C3: DistilBERT | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| pride | A: TF-IDF + LR | 111 | 15 | 16 | 0.0000 | 0.4167 | +0.4167 |
| pride | C1: BERT | 111 | 15 | 16 | 0.1133 ± 0.1112 | 0.4713 ± 0.0445 | +0.3580 ± 0.0843 |
| pride | C2: RoBERTa | 111 | 15 | 16 | 0.0000 ± 0.0000 | 0.1000 ± 0.1732 | +0.1000 ± 0.1732 |
| pride | C3: DistilBERT | 111 | 15 | 16 | 0.0000 ± 0.0000 | 0.4458 ± 0.0710 | +0.4458 ± 0.0710 |
| relief | A: TF-IDF + LR | 153 | 18 | 11 | 0.0000 | 0.1176 | +0.1176 |
| relief | C1: BERT | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.2356 ± 0.1042 | +0.2356 ± 0.1042 |
| relief | C2: RoBERTa | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.0800 ± 0.1386 | +0.0800 ± 0.1386 |
| relief | C3: DistilBERT | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.0417 ± 0.0722 | +0.0417 ± 0.0722 |
| nervousness | A: TF-IDF + LR | 164 | 21 | 23 | 0.0000 | 0.1714 | +0.1714 |
| nervousness | C1: BERT | 164 | 21 | 23 | 0.3318 ± 0.0503 | 0.3457 ± 0.0408 | +0.0139 ± 0.0773 |
| nervousness | C2: RoBERTa | 164 | 21 | 23 | 0.0000 ± 0.0000 | 0.3078 ± 0.0443 | +0.3078 ± 0.0443 |
| nervousness | C3: DistilBERT | 164 | 21 | 23 | 0.0000 ± 0.0000 | 0.3316 ± 0.1238 | +0.3316 ± 0.1238 |
| embarrassment | A: TF-IDF + LR | 303 | 35 | 37 | 0.0000 | 0.2778 | +0.2778 |
| embarrassment | C1: BERT | 303 | 35 | 37 | 0.5089 ± 0.0175 | 0.4831 ± 0.0536 | -0.0257 ± 0.0370 |
| embarrassment | C2: RoBERTa | 303 | 35 | 37 | 0.1222 ± 0.1347 | 0.4461 ± 0.0268 | +0.3238 ± 0.1149 |
| embarrassment | C3: DistilBERT | 303 | 35 | 37 | 0.1297 ± 0.0258 | 0.3730 ± 0.0700 | +0.2432 ± 0.0795 |

Support là số câu có nhãn thật, không phải số lần model dự đoán nhãn. Δ > 0 là tăng, Δ < 0 là giảm. Bảng làm tròn bốn chữ số; số gốc và mọi mức giảm được giữ trong `rare_before_after.csv`. Các nhóm C thiếu seed/config/revision nhất quán sẽ không có mean/std.

## 2. Chi phí thực nghiệm C

Lấy `elapsed_seconds` từ metadata của từng run full hoàn tất. Đây là thời gian toàn run C (chuẩn bị dữ liệu/model, train, validation và ghi checkpoint), không phải riêng thời gian optimizer. Tổng và trung bình chỉ hiển thị khi đủ tập seed yêu cầu; chi phí phụ thuộc máy và cache.

| Kiến trúc | Full seeds đủ chi phí | Số tham số | Tổng elapsed (giây) | Mean elapsed/run (giây) | Trạng thái |
|---|---|---:|---:|---:|---|
| bert | 3/3 | 108,331,804 | 24688.89 | 8229.63 | Đủ |
| roberta | 3/3 | 124,667,164 | 16329.05 | 5443.02 | Đủ |
| distilbert | 3/3 | 66,975,004 | 6115.15 | 2038.38 | Đủ |

Nguồn chi tiết: `training_costs.csv`. Phép đo chi phí B cần tổng thời gian suy luận có xử lý resume; không dùng thời gian của một lần tiếp tục để đại diện toàn dataset.

## 3. Learning curves trên validation

Mỗi đường là trung bình qua cùng tập seed; dải màu là ±1 sample std. Đây là metric validation ở ngưỡng 0,5 theo epoch, không phải learning curve trên test.

![Validation learning curves của BERT, RoBERTa và DistilBERT](<D:/DoAn/Đồ án NLP/goemotions-multilabel-classification/reports/project_results/figures/learning_curves_validation.png>)

## 4. Hồ sơ đối chiếu

- `per_label.csv`: P/R/F1 và TP/FP/FN/support cho từng nhãn, split, cấu hình và seed.
- `aggregate_metrics.csv`: Macro/Micro-F1 fixed/tuned mean ± sample std của ba C.
- `rare_before_after.csv`: nhãn hiếm, cặp trước/sau và Δ, giữ toàn bộ tăng/giảm.
- `learning_curves.csv`, `training_costs.csv`: epoch và chi phí từ metadata thật.
- `missing_artifacts.csv`, `analysis_manifest.json`: trạng thái thiếu/không hợp lệ và phạm vi dữ liệu.

Script chỉ đọc artifact và kết quả đã lưu; không huấn luyện, chọn lại ngưỡng hoặc mở nhãn test.