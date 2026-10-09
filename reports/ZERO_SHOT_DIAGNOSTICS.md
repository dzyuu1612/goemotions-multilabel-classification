# Kiểm tra nhánh B: BART-MNLI zero-shot

**Ngày kiểm tra:** 08/10/2026. Bằng chứng chi tiết và checksum nằm trong [zero_shot_audit.json](execution/zero_shot_audit.json).

## 1. Kết luận và phạm vi

Chưa phát hiện lỗi triển khai cụ thể trong luồng B đã kiểm tra. Điểm thấp thuộc thiết lập đã khóa: checkpoint BART-MNLI, 28 tên nhãn GoEmotions, template `This text expresses {}.`, `multi_label=True` và các ngưỡng chọn trên validation. Kết quả này không đủ để kết luận nguyên nhân gây lỗi hoặc tự quyết định phải chạy lại toàn bộ B.

Lượt kiểm tra chỉ đọc code, cấu hình trong cache, phần header của safetensors và artifacts đã lưu. Không đọc tensor trọng số, không tải model, không chạy suy luận mới, không dùng GPU và không sửa inference. **Chưa đối chiếu raw logits bằng một lượt chạy mới.**

## 2. Checkpoint vẫn dùng đầu MNLI ba lớp

- Checkpoint: `facebook/bart-large-mnli`, revision `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`; validation và test ghi cùng revision.
- Cached `config.json` khai báo `BartForSequenceClassification`, `_num_labels=3`, contradiction=0, neutral=1, entailment=2.
- Header safetensors ghi `classification_head.out_proj.weight` có shape `[3, 1024]`, bias `[3]`. Chỉ header đã được đọc; chưa kiểm lại checksum toàn bộ file trọng số.
- `scripts/run_zero_shot.py::build_pipeline` nạp checkpoint bằng pipeline zero-shot, không truyền `num_labels=28`, không thay head và không có optimizer/bước train; model được chuyển sang `eval()`.

**28 là số candidate labels được lần lượt kiểm bằng NLI**, không phải số lớp của head MNLI. Model card chính thức xác nhận checkpoint đã được huấn luyện trên MNLI trước khi dùng cho zero-shot. Nhóm không fine-tune trọng số trên GoEmotions. [Nguồn BART-MNLI](https://huggingface.co/facebook/bart-large-mnli).

## 3. Điểm độc lập và thứ tự nhãn

Code pipeline Transformers 4.57.6 đang cài được kiểm tra trực tiếp. Với `multi_label=True`, nó tính riêng mỗi candidate:

```text
score = exp(logit_entailment) /
        (exp(logit_contradiction) + exp(logit_entailment))
```

Các chỉ số được dùng là entailment=2, contradiction=0; NLI-neutral=1 không nằm trong phép chuẩn hóa này. Đây là cách model card hướng dẫn. Tổng 28 điểm của một câu không bắt buộc bằng 1. Nhãn `neutral` của GoEmotions là một candidate riêng, khác lớp neutral của head NLI. [Cách tính điểm chính thức](https://huggingface.co/facebook/bart-large-mnli).

HF trả candidate labels theo score giảm dần. `src/zero_shot.py::map_pipeline_scores` tạo ánh xạ tên→điểm, rồi dựng lại từng hàng theo `data/labels.json`; không lấy vị trí của danh sách đã sắp xếp làm thứ tự chuẩn. Code cũng từ chối nhãn thiếu/trùng/lạ và điểm không hữu hạn hoặc ngoài [0,1].

Manifest từng batch khóa dữ liệu/ID, revision, template, mapping, device/dtype và batch size; code resume kiểm thứ tự, phạm vi và checksum chunk. Audit đã kiểm checksum của hai manifest toàn tập; không đọc lại từng chunk trong lượt này.

## 4. Bằng chứng trong artifacts thực tế

Validation/test đều là full, `smoke=false`, CUDA float16; môi trường ghi PyTorch 2.13.0+cu130 và Transformers 4.57.6. Score được lưu dưới dạng float64 do bước tạo mảng của code; không có nghĩa model đã suy luận ở float64.

| Kiểm tra | Validation | Test |
|---|---:|---:|
| Shape scores | 5.426×28 | 5.427×28 |
| Điểm hữu hạn/trong [0,1] | Đạt | Đạt |
| ID không trùng, mapping canonical | Đạt | Đạt |
| SHA scores khớp metadata | Đạt | Đạt |
| SHA manifest batch khớp metadata | Đạt | Đạt |
| Số dự đoán từng cột khớp TP+FP của metrics fixed | Đạt | Đạt |
| Trung bình nhãn dự đoán/câu ở ngưỡng 0,5 | 15,3826 | 15,3818 |
| Trung bình nhãn thật/câu theo support đã lưu | 1,1758 | 1,1662 |

Smoke 8 câu có cùng ID/nhãn với tám hàng đầu full-validation; chênh lệch score lớn nhất bằng 0. Đây là so sánh artifacts đã lưu, không thay thế một phép đối chiếu raw logits độc lập.

| Thiết lập B trên test | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| Fixed 0,5 | 0,1035 | 0,1008 | 0,5315 |
| Ngưỡng riêng đã khóa trên validation | 0,1609 | 0,1752 | 0,2941 |

Ở fixed 0,5, Micro-Precision khoảng 0,0542 và Micro-Recall khoảng 0,7148. Số nhãn dự đoán lớn hơn rất nhiều số nhãn thật phù hợp với thống kê FP cao và precision thấp. Đây là mô tả số đo, không phải kết luận nhân quả.

## 5. Đánh đổi khi không dùng NLI-neutral

Phép so sánh hai lớp chỉ hỏi entailment tương đối với contradiction; nó không xét độ lớn của logit neutral. Do đó không thể mặc định điểm này bằng xác suất entailment của softmax ba lớp hoặc xác suất cảm xúc đã được hiệu chuẩn trên GoEmotions.

Về toán học, nếu neutral có xác suất lớn trong softmax ba lớp, việc bỏ neutral khỏi mẫu số có thể tạo score hai lớp cao hơn xác suất entailment ba lớp. **Audit chưa đo các raw logits đó**, chưa kiểm calibration và chưa so các template khác; không khẳng định hiện tượng này là nguyên nhân của toàn bộ FP đã quan sát. Tên nhãn, cách diễn đạt hypothesis và khác biệt giữa MNLI với GoEmotions cũng chưa được tách bằng thí nghiệm đối chứng.

## 6. Đối chiếu nhỏ nếu cần chốt rủi ro

Nếu phát sinh bằng chứng lỗi mới, có thể dùng đúng 3 câu validation để so pipeline với 28 scores đã lưu, đồng thời xem logits contradiction/neutral/entailment. Giữ nguyên checkpoint/revision, template, candidate labels, dtype và `multi_label=True`; ghi sai số nếu batching tạo khác biệt số học. Đối chiếu hiện có đã đủ xử lý rủi ro triển khai được nêu; không chạy thêm GPU hoặc chạy lại toàn bộ B trong lượt này.

Không đổi ngưỡng/template/chuẩn hóa vì đã thấy kết quả test. Một lượt kiểm nhỏ nhằm xác nhận thực thi và giải thích cách tính điểm, không phải phép lựa chọn thiết lập mới. Chỉ xem xét chạy lại toàn B khi tìm được lỗi cụ thể làm thay đổi kết quả.

**Giới hạn nguồn:** URL cấu hình tại revision cố định chưa truy cập được qua công cụ duyệt web của lượt audit. Cấu hình và head được xác nhận từ cache đúng revision tại máy chạy; model card chính thức được đọc để đối chiếu ngữ nghĩa pipeline.
