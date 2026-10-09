# B — Zero-shot BART-MNLI trên 28 nhãn GoEmotions

## 1. Vai trò trong đồ án

B trả lời: một mô hình đã học suy luận ngôn ngữ (NLI), chưa được nhóm fine-tune
trên GoEmotions, có thể nhận ra các cảm xúc tốt tới mức nào? So sánh B với A
(TF-IDF + Logistic Regression được fit trên train) và C (Transformer fine-tune).
Không gọi B là mô hình C, không dùng B thay demo từ mô hình C tốt nhất.

Checkpoint: `facebook/bart-large-mnli`. Nó đã được tác giả huấn luyện trên MNLI;
"zero-shot" ở đây nghĩa là nhóm không cập nhật trọng số trên GoEmotions. Không
khẳng định checkpoint chưa từng thấy mọi câu trong Reddit khi được pretrain.

## 2. B hoạt động như thế nào?

Ví dụ văn bản tiếng Anh: `Thank you! This made me happy.` Mỗi nhãn tạo một câu
giả thuyết: `This text expresses gratitude.` hoặc `This text expresses joy.`
Mô hình NLI so sánh văn bản với từng giả thuyết. Pipeline biến điểm entailment
và contradiction thành một điểm riêng cho từng nhãn khi `multi_label=True`.
Điểm đó là tín hiệu NLI, không phải một xác suất cảm xúc đã được hiệu chuẩn.

Một câu tạo **28 cặp văn bản–giả thuyết**; điểm các nhãn không cần cộng thành 1.
Ngưỡng 0,5 ban đầu tạo tập nhãn. Ví dụ này giải thích cơ chế, không phải kết quả
đã đo. `neutral` vẫn là nhãn thứ 28; không tự ép neutral khi không có nhãn đạt
ngưỡng, không lấy duy nhất nhãn cao nhất để làm mất tính đa nhãn.

HF trả kết quả theo điểm giảm dần. `map_pipeline_scores()` đưa từng nhãn về
đúng thứ tự trong `data/labels.json` trước khi lưu hoặc tính metric.

## 3. Chạy từng bước

Chạy từ gốc repo. Cần Python 3.12/3.13 và môi trường PyTorch phù hợp thiết bị.
`requirements-zero-shot.txt` thêm thư viện B vào môi trường baseline. BART-large
cần tải model khá lớn lần đầu và CPU có thể chạy lâu; kiểm tra nhỏ trước.

```powershell
python -m pip install -r requirements-zero-shot.txt
python -m scripts.run_zero_shot --smoke --device cpu --batch-size 2
python -m scripts.run_zero_shot --device auto --batch-size 8
```

Lần chạy GPU của đồ án dùng `--device cuda --batch-size 16 --dtype float16`.
Float16 giảm bộ nhớ cho GPU 8 GB; dtype được lưu trong metadata, giữ nguyên khi
resume và test. CPU dùng float32. Đây là thay đổi độ chính xác số khi suy luận,
không phải fine-tune trọng số.

`--smoke` dùng 8 câu đầu validation. `--limit 100` dùng 100 câu và **luôn đánh
dấu smoke**, kể cả khi limit đủ lớn để lấy hết split. Kết quả smoke chỉ xác nhận
pipeline chạy, không dùng trong bảng kết quả chính thức.

Validation mặc định không đọc train/test, không fit mô hình, không chọn prompt
trên test. Lần full lưu:

```text
data/processed/zero_shot/full/
  validation_scores.npz             # ids, scores [5426,28], label_names
  run_metadata.json                 # checkpoint SHA, data hash, môi trường, metric
  validation_metrics.json           # chỉ số shared ở ngưỡng 0,5
  validation_checkpoints/
    checkpoint_manifest.json        # batch đã xong và SHA-256 từng chunk
    chunk_000000_000008.npz          # điểm của batch; tiếp tục cùng cấu hình
```

Chạy lại cùng lệnh sẽ bỏ qua batch đã hoàn tất. Thay model revision, thứ tự dữ
liệu, batch-size, môi trường hoặc thiết bị sẽ bị từ chối khi resume. Đọc
`model_revision` trong metadata rồi truyền `--revision <SHA>` để tái lập đúng
checkpoint. Khi run đã hoàn tất, giữ nguyên metadata và file điểm; tránh thay
đổi hash sau khi đã khóa protocol. Không xóa chung thư mục giữa hai thí nghiệm.

## 4. Ngưỡng và đánh giá test

**B mặc định:** trọng số cố định, template `This text expresses {}.`, 28 nhãn,
`multi_label=True`, ngưỡng 0,5. Macro/Micro-F1, Precision/Recall, Hamming Loss dùng
chung `src.metrics.evaluate_multilabel` với A/C.

Nếu chọn ngưỡng dựa vào nhãn validation, gọi rõ là **zero-shot về trọng số có
hiệu chỉnh ngưỡng trên validation**. Đây là một biến thể có bước dùng nhãn;
không gọi toàn bộ quy trình đó là zero-shot không dùng nhãn.

Trước test, dùng `scripts.freeze_experiment` do nhóm quản lý để khóa cấu hình
và ngưỡng từ full validation. CLI khóa nhận `--run-dir
data/processed/zero_shot/full`. Xem `python -m scripts.freeze_experiment --help`
để chọn mode fixed/global/tuned; không dùng test để quyết định mode thắng.

```powershell
python -m scripts.run_zero_shot --split test --device auto --batch-size 8 --protocol data/processed/zero_shot/full/final_protocol.json
```

Nếu validation đã dùng cấu hình GPU trên, lệnh test cũng truyền
`--device cuda --batch-size 16 --dtype float16`. Dùng `scripts.complete_project`
để lấy dtype từ metadata và giữ cùng protocol.

Test bị chặn nếu thiếu protocol, dùng smoke, nhãn/revision khác, metadata chưa
hoàn tất hoặc hash validation bị thay đổi. Protocol v1 chứa method, checkpoint,
model_revision, hypothesis_template, label_names, data_revision,
validation_scores_sha256, run_metadata_sha256, thresholds, threshold_mode,
frozen_at_utc. `test_run_metadata.json` riêng không ghi đè metadata validation.
Validator chỉ kiểm identity/hashes/ngưỡng, không đọc nhãn để tune lại ngưỡng.

## 5. Giải thích code khi bảo vệ

1. `run_zero_shot.py`: đọc tham số, kiểm protocol nếu test, đọc đúng split.
2. `resolve_model_revision`: xác định commit SHA model để lưu phiên bản thật.
3. `build_pipeline`: tải BART-MNLI; dùng chế độ eval, không có optimizer/loss fit.
4. `predict_batch`: truyền 28 candidate labels và template, bật đa nhãn.
5. `predict_in_batches`: gọi pipeline, đưa nhãn về đúng cột, ghi từng batch.
6. `write_scores`: xuất ma trận theo ID cho module so sánh dùng chung.
7. `evaluate_multilabel`: so nhãn thật và nhãn dự đoán, không chọn ngưỡng ở test.

Batch-size trong HF là số cặp NLI xử lý cùng lượt; không mặc định mỗi câu chỉ
tốn một lượt tính. Ngoài F1, báo thời gian, thiết bị và dung lượng model nếu có
đo, vì chi phí BART-MNLI có thể cao hơn baseline TF-IDF.

## 6. Kiểm chứng và trạng thái

```powershell
python -m unittest tests.test_zero_shot -v
```

Tests dùng dữ liệu giả để kiểm: thứ tự nhãn, nhãn thiếu/trùng/lạ, điểm NaN hoặc
ngoài [0,1], batch chưa đủ, resume không tính lại batch đã xong, đổi cấu hình,
chunk bị sửa và test thiếu protocol. Tests không tải model và không chứng minh
chất lượng trên GoEmotions. Có code hoặc notebook chưa có nghĩa đã có kết quả;
chỉ điền số thực từ run metadata hoàn tất vào báo cáo.

## 7. Nguồn chính thức

- [Model card BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli):
  nguồn checkpoint và ví dụ NLI zero-shot.
- [Hugging Face zero-shot pipeline](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline):
  `candidate_labels`, `hypothesis_template`, `multi_label` và cấu trúc đầu ra.
- [GoEmotions ACL 2020](https://aclanthology.org/2020.acl-main.372/): bài nền tảng
  của đề tài và bài toán cảm xúc đa nhãn. BART-MNLI là phương pháp so sánh B của
  đồ án, không được trình bày như baseline BERT gốc của bài báo này.
