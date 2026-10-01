# Kết quả phần A — baseline cổ điển GoEmotions

Bản này sinh tự động từ các file validation đã chạy. Dữ liệu: GoEmotions `simplified`, revision `add492243ff905527e67aeb8b80c082af02207c3`; train 43.410, validation 5.426, 28 nhãn. **Chưa dùng nhãn test.**

## Thiết lập

- TF-IDF unigram và bigram, `min_df=2`, tối đa 100.000 đặc trưng; fit trên train.
- One-vs-Rest Logistic Regression: 28 bộ phân loại nhị phân, `C=1`, `solver=liblinear`, `max_iter=1000`. Standard không có trọng số lớp; balanced dùng `class_weight='balanced'` tính từ train cho từng bộ phân loại.
- Dùng cùng split và thứ tự nhãn, ngưỡng gốc 0,5. Macro-F1 tính trung bình F1 của đủ 28 nhãn; `zero_division=0`.

## Bảng validation

| Cấu hình | Ngưỡng | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Hamming Loss |
|---|---|---:|---:|---:|---:|---:|
| standard | Cố định 0,5 | 0.2025 | 0.3760 | 0.7254 | 0.2538 | 0.0354 |
| standard | Chọn theo từng nhãn trên val | 0.4391 | 0.5427 | 0.4876 | 0.6119 | 0.0433 |
| balanced | Cố định 0,5 | 0.4562 | 0.5099 | 0.4158 | 0.6592 | 0.0532 |
| balanced | Chọn theo từng nhãn trên val | 0.4901 | 0.5467 | 0.4796 | 0.6356 | 0.0443 |

**Cách hiểu:** các hàng chọn ngưỡng được đo trên chính validation đã dùng để chọn ngưỡng; mức tăng ở đó có thể lạc quan. Chỉ dùng test một lần sau khi cả nhóm khóa cấu hình để xác nhận kết luận.

## Môi trường và thời gian đo

| Biến thể | Python | scikit-learn | Số đặc trưng TF-IDF | Fit (giây) | Dự đoán val (giây) |
|---|---|---|---:|---:|---:|
| standard | 3.13.9 | 1.7.2 | 58338 | 19.52 | 0.21 |
| balanced | 3.13.9 | 1.7.2 | 58338 | 29.60 | 0.33 |

Thời gian phụ thuộc máy và cache; xem JSON để có đủ phiên bản thư viện.

## Năm nhãn hiếm nhất theo số mẫu dương trên train

Danh sách được chọn bằng train trước khi nhìn kết quả validation.

| Nhãn | Train + | Val + | standard F1@0,5 | standard F1 tuned-val | balanced F1@0,5 | balanced F1 tuned-val |
|---|---:|---:|---:|---:|---:|---:|
| grief | 77 | 13 | 0.0000 | 0.0000 | 0.4118 | 0.4375 |
| pride | 111 | 15 | 0.0000 | 0.3333 | 0.5161 | 0.6087 |
| relief | 153 | 18 | 0.0000 | 0.1053 | 0.1277 | 0.1739 |
| nervousness | 164 | 21 | 0.0000 | 0.0909 | 0.3077 | 0.3125 |
| embarrassment | 303 | 35 | 0.1081 | 0.3636 | 0.5135 | 0.5507 |

## Phân tích lỗi và khả năng giải thích

Mỗi biến thể lưu `error_examples_validation.csv` với ID, văn bản, nhãn thật, nhãn dự đoán và điểm số. Bốn nhóm: bỏ sót nhãn hiếm, dự đoán nhãn thừa, đúng một phần ở mẫu đa nhãn, và không dự đoán nhãn nào. Cần đọc lại từng ví dụ trước khi trích vào báo cáo, vì nhãn gốc cũng có thể thiếu.

- **rare_false_negative**, ID `eczwil0`: `I am so proud of this community.` — thật: pride; đoán: (không nhãn). Điểm pride = 0.3776.
- **false_positive**, ID `ed832y6`: `"Homeopaths love it!"` — thật: neutral; đoán: love. Điểm love = 1.0000.
- **partial_multi_label**, ID `eczdvun`: `Thank you. I really appreciate your response` — thật: admiration, gratitude; đoán: gratitude. Điểm admiration = 0.4989.
- **no_label_predicted**, ID `eeoh5vh`: `And he even dared to reference fairly odd parents in there, wtf lol` — thật: amusement; đoán: (không nhãn). Điểm amusement = 0.4999.

`top_features.csv` ghi từ/cặp từ có hệ số LR cao và thấp cho từng nhãn. Đây là liên hệ thống kê trong mô hình, không chứng minh nguyên nhân cảm xúc.

## Bằng chứng chạy lại và bàn giao

- `standard`: `data/processed/baseline/full/validation_metrics.json`, `validation_scores.npz`, `per_label_validation.csv`, `thresholds_validation.json`, `error_examples_validation.csv`, `top_features.csv`, `model.joblib`.
- `balanced`: `data/processed/baseline/balanced/full/validation_metrics.json`, `validation_scores.npz`, `per_label_validation.csv`, `thresholds_validation.json`, `error_examples_validation.csv`, `top_features.csv`, `model.joblib`.
- `validation_scores.npz` gồm `ids`, `scores` N×28, `label_names`; ghép theo ID, không ghép theo thứ tự dòng. Các file lớn nằm trong `data/processed/` và được Git bỏ qua.
- Môi trường, thời gian fit, số đặc trưng và cấu hình nằm trong `validation_metrics.json` của từng biến thể.
- Báo cáo này chỉ mô tả phần A. Nhóm vẫn cần B zero-shot, ba kiến trúc C mỗi kiến trúc ba seed, demo từ C tốt nhất và so sánh lỗi giữa C1/C2/C3.

## Nguồn phương pháp

- [Google Research: GoEmotions](https://github.com/google-research/google-research/blob/master/goemotions/README.md)
- [scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: Precision, Recall, F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
