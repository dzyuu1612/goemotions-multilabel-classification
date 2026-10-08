# Phần phương pháp C3 — Nhật Huy

## Bản thảo có thể đưa vào báo cáo nhóm

Nhánh C3 sử dụng checkpoint `distilbert/distilbert-base-uncased` và tokenizer tương ứng
để biểu diễn bình luận tiếng Anh. Nhóm giữ split simplified và thứ tự 28 nhãn GoEmotions
thống nhất với các nhánh khác. Mỗi mẫu được biểu diễn bằng một vector multi-hot 28 chiều;
nhiều phần tử có thể đồng thời bằng 1. Văn bản được tokenize, cắt tối đa 128 token và
padding động theo batch. Không ghép train với validation.

Mô hình có đầu phân loại 28 đầu ra và fine-tune toàn bộ tham số. Loss là
`BCEWithLogitsLoss` trên logits và target float32. Khi suy luận, sigmoid được áp dụng
độc lập cho từng đầu ra; một nhãn được dự đoán nếu score của nhãn đạt ngưỡng 0,5 trong
thiết lập cơ sở. Không dùng softmax hoặc argmax để ép mỗi câu chỉ có một cảm xúc.

Mã thực hiện dùng AdamW, learning rate `2e-5`, weight decay 0,01 (trừ bias và LayerNorm),
linear warmup 10% tổng số bước cập nhật rồi giảm tuyến tính, gradient clipping 1,0.
Cấu hình full dùng batch vật lý/hiệu dụng 16, không tích lũy thêm. AMP và gradient
checkpointing được bật, CPU threads=4. Khả năng chạy batch thật ở độ dài 128 trên GPU
4 GB được kiểm trước khi chốt full. Cấu hình giữ nguyên giữa ba seed 42, 123, 2026.
GradScaler tự giảm loss scale và bỏ bước cập nhật khi overflow; scheduler chỉ tiến khi
optimizer thực sự cập nhật. Số bước bị bỏ được lưu cùng log/run metadata.

Sau mỗi epoch, mô hình được đánh giá trên validation bằng hàm metric chung của repo.
Checkpoint được chọn theo Macro-F1 validation tại ngưỡng 0,5 trên đủ 28 nhãn; trường hợp
hòa giữ epoch sớm hơn. Các phép chia không xác định dùng `zero_division=0`.
Micro-F1, Precision/Recall micro và macro, Hamming Loss, P/R/F1/support từng nhãn cũng
được xuất. Test không tham gia huấn luyện, chọn checkpoint hoặc chọn cấu hình.

Để hỗ trợ tái lập, run lưu revision dữ liệu/model, seed, môi trường phần mềm, phần cứng,
thời gian, bộ nhớ GPU, ID mẫu, scores sigmoid theo đúng ID và thứ tự nhãn, checkpoint,
tokenizer và hash artifact. Sau khi lưu, chương trình nạp lại checkpoint và đối chiếu
scores với kết quả trước khi lưu. Demo và CLI dùng chung hàm suy luận để tránh khác biệt
về tiền xử lý, mapping nhãn và ngưỡng.

## Phân biệt phần đã thực hiện và phần chưa có kết quả

**Đã thực hiện ở mức pilot:** 1.024 mẫu train và 256 mẫu validation lấy mẫu bằng seed
2026, huấn luyện seed 42 trong một epoch. Các số đo thực tế và giới hạn diễn giải nằm
trong [PILOT_RESULTS.md](c3_distilbert/PILOT_RESULTS.md). Pilot kiểm khả năng chạy và
tính nhất quán của pipeline; không dùng để báo cáo thứ hạng C3 so với baseline hoặc C1/C2.

**Thực nghiệm C3 full:** train 43.410 mẫu và validation 5.426 mẫu, ba seed
42/123/2026, mỗi seed ba epoch; báo từng seed và mean ± sample std (`ddof=1`).
Số liệu, trạng thái nghiệm thu và hạn chế nằm trong [C3_RESULTS.md](c3_distilbert/full/C3_RESULTS.md).
Chỉ run hoàn thành và kiểm đủ artifact mới được đưa vào báo cáo; lượt bị ngắt được lưu
riêng, không trộn vào kết quả. Chưa đánh giá test hoặc chọn kiến trúc thắng của cả nhóm.

**Nâng cao thuộc C3:** giữ checkpoint và thay ngưỡng quyết định từng nhãn. Với mỗi seed,
chọn ngưỡng trên validation của đúng checkpoint, lưới 0,05..0,95, bước 0,05; hòa chọn
gần 0,5 nhất rồi ngưỡng lớn hơn. Bảng cơ sở và tuned được giữ riêng, báo đủ ba seed và
năm nhãn hiếm xác định từ train. Tuning và đánh giá trên cùng validation có thể lạc quan;
không coi đây là mức cải thiện trên test. Không tự bổ sung weighting hoặc contrastive.

**Phân tích lỗi:** dùng ID/text/true/pred/scores thật, thống kê FN/FP, câu đa nhãn bị bỏ
sót một phần và lỗi nhãn hiếm/neutral. Nhóm lỗi có thể giao nhau. Huy đọc lỗi C3 riêng;
đối chiếu ba kiến trúc còn cần scores C1/C2 cùng ID từ các bạn.

**Demo:** checkpoint đại diện C3 được chọn theo Macro-F1 validation @0,5 trong ba seed;
không mặc định C3 là kiến trúc thắng. App/CLI kiểm hash ngưỡng và dùng cùng hàm suy luận.
Demo cuối của cả nhóm cần thay bằng checkpoint C thắng khi đã có đủ kết quả so sánh.

DistilBERT được tác giả tiền huấn luyện bằng distillation; trong phần này Huy fine-tune
checkpoint công bố sẵn, không tự thực hiện distillation hoặc tạo thêm dữ liệu huấn luyện.
Nguồn: [Sanh và cộng sự (2019)](https://arxiv.org/abs/1910.01108),
[model card](https://huggingface.co/distilbert/distilbert-base-uncased) (đọc 07/10/2026).

## Nguồn và liên hệ mã

- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).
- [DistilBERT trong Transformers](https://huggingface.co/docs/transformers/model_doc/distilbert).
- [PyTorch — AMP, unscale và gradient clipping](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
- Mã dùng chung: `src/c3.py`, `src/data.py`, `src/metrics.py`.
- Vòng lặp train: `scripts/train_distilbert.py`; notebook: `notebooks/distilbert_huy.ipynb`.
- Kế hoạch: bản phân công nhóm ngày 04/10/2026 do người dùng cung cấp trong file ZIP.
