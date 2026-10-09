# Rà soát phần Nhật Huy sau tích hợp — 09/10/2026

Phạm vi theo phân công: kiểm C3 của Huy, đọc phần C3 trong so sánh/lỗi chung và xác định
đầu vào còn thiếu để Huy nghiệm thu demo D trên máy mình. Repo nền: `f4330df`.
Không huấn luyện, không chọn ngưỡng mới, không đánh giá test mới và không thay các báo cáo
ngày 08/10. Kiểm suy luận C3 dùng đúng checkpoint có sẵn và các câu train/validation thật.

## 1. Kiểm lại demo C3 sau khi đổi tên và tích hợp

Chạy bằng `.venv-huy` trên máy Huy, PyTorch 2.11.0+cu128, Transformers 4.57.6;
suy luận CPU. App được kiểm là **`app_distilbert_huy.py`**, checkpoint full seed 123.

| Nội dung | Kết quả |
|---|---|
| Hash artifact, mapping 28 nhãn, ID/scores | Đạt |
| Tính lại metrics validation từ scores C3 đã lưu | Khớp |
| CLI so với hàm suy luận chung | Sai lệch lớn nhất 2,3841858e-7; nhỏ hơn atol 1e-6 |
| Widget Streamlit so với hàm suy luận chung | Sai lệch lớn nhất 0 |
| Ngưỡng riêng đúng checkpoint, seed và mapping | Đạt; app/CLI khớp |
| Input rỗng | Xử lý đúng, không dự đoán |
| Input dài | Mẫu train `edwy61c`, 316 token; có thông báo cắt token |
| Nhiều nhãn | Đạt trên mẫu validation `eczdvun` |
| Không nhãn vượt ngưỡng | Trả tập rỗng trên mẫu thật; không ép neutral |
| Khởi động → dừng → khởi động lại server | Cả hai lượt HTTP `/` và health đều 200 |

Kiểm widget dùng Streamlit AppTest. Kiểm HTTP dùng server thật ở cổng 8511 và đã dừng
sau kiểm để không giữ tiến trình thừa. Hai phép kiểm này riêng biệt; không tuyên bố
đã thao tác trình duyệt hoặc có ảnh/video mới. Mẫu parity đầu tiên là validation `edgurhb`.

Bằng chứng: [verification.json](verification.json), [server_restart.json](server_restart.json),
log `server_first_start.log`, `server_restart.log` và các file stderr tương ứng.

## 2. Đối chiếu C3 trong hồ sơ thực nghiệm chung

Đã kiểm 74 JSON thuộc phần Transformer xuất theo manifest: hash bản copy khớp hash
nguồn được ghi. Đối chiếu **18 dòng C3** (3 seed × 2 split × 3 chế độ ngưỡng), mỗi dòng
7 metrics, với JSON kết quả/protocol gốc đã xuất; tính lại **6 dòng mean/std**, `ddof=1`.
Kết quả đều khớp. Liên kết hash giữa metadata, protocol và kết quả test cũng khớp.

Hai bộ C3 được giữ riêng:

| Hồ sơ | Validation Macro-F1 @0,5, mean ± sample std | Phạm vi |
|---|---:|---|
| C3 Huy chạy riêng | 0,406084 ± 0,004544 | 3 seed; chưa đánh giá test |
| C3 trong bộ chung | 0,406409 ± 0,006018 | 3 seed; có validation/test trong hồ sơ chung |

Không cộng thành sáu seed hoặc chuyển kết quả test của bộ chung thành kết quả của Huy.
Tính lại quy tắc chọn bằng metrics validation đã xuất của chín run cho cùng lựa chọn
**BERT seed 123**, đúng `selected_model.json`. Không dùng test để chọn lại.

**Giới hạn:** máy Huy chưa có trọng số/scores NPZ của bộ chung. Phép kiểm trên xác nhận
tính nhất quán của các bảng và JSON được bàn giao; chưa tính lại metrics chung từ NPZ,
chưa tái chạy mô hình chung và chưa xác nhận toàn bộ trọng số từ hash thực tế.

## 3. Rà phần C3 trong ba case lỗi chung đã có

Đã kiểm **45 dòng ví dụ** trong CSV: đủ 28 scores, nhãn dự đoán theo ngưỡng 0,5, FN/FP,
cờ nhóm lỗi; mỗi mẫu cùng văn bản/nhãn thật giữa ba kiến trúc. Seed/hash run và hash
scores trong manifest khớp metadata/test JSON xuất kèm.

Ba case có nhận xét sẵn ở [error_case_studies.md](../../error_case_studies.md) được đọc lại:

| ID và nhóm đã chọn | Đối chiếu phần C3 và ý nghĩa |
|---|---|
| `eczj48j` — bỏ sót một phần nhãn | C3 nhận admiration, bỏ excitement và neutral; giống tập nhãn dự đoán của C2. C1 dự đoán caring nên không thuộc cờ partial, nhưng vẫn sai. Không diễn giải cờ False thành đúng. |
| `ed0jr9i` — bỏ sót nhãn hiếm | C3 bỏ embarrassment (score khoảng 0,1956 < 0,5); C2 cũng bỏ, C1 nhận đúng. Nhận xét về các từ “shy”, “awkward” phù hợp nội dung được bàn giao, nhưng không chứng minh cơ chế hay nguyên nhân mô hình. |
| `eczcvgx` — FN và FP cùng câu | C3 thêm approval, bỏ neutral; C1/C2 nhận đúng neutral. Không đủ bằng chứng khẳng định câu châm biếm. |

Đây là **DistilBERT seed 123 của bộ chung**, không phải checkpoint seed 123 riêng của Huy.
Không thay text/gold/scores hoặc bổ sung câu minh họa. Đã đọc các ví dụ test có sẵn trong
báo cáo; không mở lại tập test gốc hay tạo dự đoán mới. Do thiếu NPZ gốc trên máy này,
chỉ kiểm được tính nhất quán trong CSV/metadata, không xác nhận lại scores bằng inference.

Bằng chứng và hash đầu vào: [shared_c3_review.json](shared_c3_review.json).
Lệnh kiểm lại hồ sơ: `python -m scripts.review_c3_integration` (không train/inference).

## 4. Đầu vào còn thiếu cho demo D trên máy Huy

Repo đã có demo chung `app.py` và bằng chứng chạy trên máy đã tạo bộ thực nghiệm chung.
Trên máy Huy, `data/processed/transformers/selected_model.json` và checkpoint BERT chưa
có; `.venv-huy` chưa cài Gradio. Vì vậy chưa xác nhận D chạy trên máy Huy.

Huy cần nhận **nguyên bundle đã được chọn**, không chỉ một file trọng số:

- `data/processed/transformers/selected_model.json`.
- `data/processed/transformers/bert/seed_123/full/standard/`, gồm `run_metadata.json`,
  `label_mapping.json`, `validation_scores.npz`, `validation_metrics.json` và toàn bộ
  `checkpoint/` (model, config, tokenizer/vocabulary). Loader kiểm các file đã ghi hash.
- Nếu dùng ngưỡng riêng, file ngưỡng của đúng run đã khóa. Demo chung mặc định dùng 0,5;
  không thay bằng ngưỡng C3 của Huy.

Checkpoint được chọn là `google-bert/bert-base-cased`, seed 123, epoch 4; không tải
checkpoint pretrained rồi xem đó là model đã fine-tune. SHA-256 của model đã ghi:
`7ce895dc6d6ffbe35097cf65c58898eda4c2015185af3a841b76176ca3b030b2`.

Khi có bundle: kiểm hash theo metadata → chuẩn bị môi trường demo riêng phù hợp →
kiểm app/CLI và input biên → khởi động lại, lưu minh chứng trên máy Huy. Không cài đè
môi trường C3 đã có và không huấn luyện thay một checkpoint còn thiếu trong bàn giao.

Hiện đã hoàn thành các việc độc lập nêu trên; còn chờ bundle BERT để kiểm D tại chỗ.
Không cần đợi thêm bảng C1/C2, không cần chạy thêm C3 trong phạm vi lần cập nhật này.
