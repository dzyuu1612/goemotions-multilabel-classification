# Kế hoạch và phân công nhóm — GoEmotions

Cập nhật **03/10/2026** theo trao đổi nhóm bạn gửi. Kế hoạch chia theo công việc,
người phụ trách và đầu ra; không chia theo tuần, ngày hoặc thời lượng.
Kho chung: [goemotions-multilabel-classification](https://github.com/trangkhanh-ai/goemotions-multilabel-classification).

## 1. Ghi nhận phân công từ trao đổi nhóm

- **Duy:** baseline A.
- **Đức Trí chính là “Thợ Săn Thập Cẩm”** trong đoạn chat; Đức Trí đã nhận **zero-shot B**.
- **Quốc Khánh** đã nhận **phần đầu báo cáo và fine-tune một mô hình**.
- Kiến trúc cụ thể được phân bổ trong kế hoạch này: Khánh = BERT, Hoàng Phúc = RoBERTa, Nhật Huy = DistilBERT.
- Phần RoBERTa/DistilBERT cùng việc hỗ trợ của Phúc/Huy là phân công bổ sung theo yêu cầu của Duy.

## 2. Bảng phân công chính

| Người | Phần chính | Việc hỗ trợ | Sản phẩm cần bàn giao |
|---|---|---|---|
| **Duy — Bảo Duy Nguyễn** | **A: TF-IDF + One-vs-Rest Logistic Regression** | Dữ liệu/metrics dùng chung, bảng A và phối hợp tổng hợp A/B/C | Code A, scores theo ID, cấu hình, metric và phần báo cáo A; chi tiết nằm trong mục riêng của Duy |
| **Đức Trí — Thợ Săn Thập Cẩm** | **B: zero-shot BART-large-MNLI, không fine-tune** | Phần phương pháp/kết quả B, lỗi zero-shot và mapping nhãn | Pipeline B, prompt/checkpoint revision, scores val/test, metrics, ví dụ lỗi B |
| **Quốc Khánh** | **C1: BERT-base + phần đầu báo cáo** | Khởi tạo script fine-tune chung, thống nhất cấu trúc báo cáo | BERT đủ ≥3 seed, checkpoint/log/scores, mean±std; mục tiêu, paper, dữ liệu/EDA và bố cục báo cáo |
| **Hoàng Phúc** | **C2: RoBERTa-base** | Điều phối bảng nhãn hiếm, nâng cao và các cặp cảm xúc dễ nhầm | RoBERTa đủ ≥3 seed; phần báo cáo C2; bảng nâng cao trước/sau từ số liệu cả nhóm |
| **Nhật Huy** | **C3: DistilBERT-base** | Tích hợp demo dùng mô hình C tốt nhất và hướng dẫn chạy demo | DistilBERT đủ ≥3 seed; phần báo cáo C3; app Gradio/Streamlit và minh chứng chạy được |
| **Cả nhóm** | So sánh A/B/C và ≥3 nhóm lỗi giữa C1/C2/C3 | Đọc paper, kiểm số liệu, review code, giải thích độ ổn định và phản biện | Bảng so sánh chung, báo cáo hoàn chỉnh, nguồn, đóng góp thực tế và link demo |

Đức Trí sở hữu B; ba người sở hữu ba C lần lượt là Quốc Khánh, Hoàng Phúc, Nhật Huy.
Mỗi người làm C thực hiện trọn kiến trúc của mình từ pilot đến đủ seed và phân tích kết quả.
Nhật Huy tích hợp **mô hình C thắng theo validation**, kể cả model thắng do Khánh hoặc Phúc huấn luyện.

## 3. Mục tiêu và điều kiện hoàn thành

Phân loại cảm xúc **đa nhãn** trên bình luận Reddit tiếng Anh: một câu có thể có
nhiều cảm xúc, với **27 cảm xúc + neutral**, tổng 28 đầu ra.

- A: baseline cổ điển TF-IDF + Logistic Regression.
- B: pretrained dùng trực tiếp qua HF pipeline, **không fine-tune** trên GoEmotions.
- C: **ba kiến trúc pretrained khác nhau**, mỗi kiến trúc **ít nhất ba seed**, có kết quả từng seed và **mean ± std**.
- D: demo Gradio/Streamlit từ **mô hình tốt nhất trong ba C**.
- Đánh giá: Macro/Micro-F1, Precision/Recall, Hamming Loss; phân tích cặp cảm xúc dễ nhầm.
- Ít nhất ba nhóm lỗi cụ thể, có so sánh C1/C2/C3 và ví dụ thật theo ID.
- Nâng cao: class weighting **hoặc** ngưỡng riêng từng nhãn **hoặc** contrastive; báo F1 nhãn hiếm trước/sau.
- Báo cáo đủ phương pháp, tiền xử lý, cấu hình, số liệu, độ ổn định, hạn chế và đóng góp.

Yêu cầu kỹ thuật lấy từ nội dung cô đã cung cấp. Các checkpoint, seed cụ thể và
phân công hỗ trợ bên dưới là cách triển khai của nhóm. Phần nâng cao không bị
giới hạn riêng ở C; mở rộng tuning sang C là lựa chọn của nhóm để đối chiếu.

## 4. Dữ liệu và giao diện dùng chung

| Hạng mục | Quy ước |
|---|---|
| Dataset | `google-research-datasets/go_emotions`, config `simplified` |
| Revision | `add492243ff905527e67aeb8b80c082af02207c3` |
| Split chính thức | Train 43.410 / validation 5.426 / test 5.427; tổng 54.263 |
| Mapping | Giữ đúng 28 nhãn trong `data/labels.json`; nhãn thật multi-hot N×28 |
| Text | Giữ văn bản nguồn; mỗi phương pháp dùng bộ biểu diễn/tokenizer phù hợp |
| Module dữ liệu | Tái sử dụng `src/data.py`; kiểm ID, SHA-256 và label mapping |
| File dự đoán | NPZ có `ids`, `scores` N×28, `label_names`; ghép bằng ID |
| Scores | Hữu hạn trong 0–1, lưu trước ngưỡng; không chỉ lưu hard labels |
| Metrics | Dùng chung `src/metrics.py`; đủ 28 nhãn, `zero_division=0` |
| Train | Học A/C, fit TF-IDF và tính weighting |
| Validation | Chọn config/checkpoint/prompt nếu có/ngưỡng; ghi rõ cách chọn |
| Test | Chỉ đánh giá sau khi khóa các lựa chọn; không chọn lại bằng test |

Nguồn thô có 58.009 bình luận; simplified đang dùng có 54.263. Không gộp nhãn
thành positive/negative/neutral ở bảng chính; không ép neutral loại trừ cảm xúc khác.
Model/scores lớn chia sẻ riêng hoặc tái chạy; code và bảng nhỏ lưu trên GitHub.

## 5. Đức Trí — zero-shot B

1. Pilot một tập nhỏ validation, kiểm pipeline/mapping và khả năng chạy theo batch.
2. Dùng `facebook/bart-large-mnli`, đủ 28 candidate labels, `multi_label=True`.
   Template khởi đầu: `This text expresses {}.`.
3. Mapping scores về thứ tự `data/labels.json`: pipeline trả nhãn theo score giảm dần.
4. Chạy full validation, lưu từng phần kèm ID để tiếp tục khi phiên chạy gián đoạn.
5. Báo kết quả gốc @0.5. Nếu dùng nhãn validation chỉnh template/ngưỡng, giữ hàng
   riêng và ghi rõ **không cập nhật trọng số nhưng có hiệu chỉnh bằng nhãn đích**.
6. Ghi checkpoint revision, template, batch, thiết bị, thư viện và cấu hình suy luận.
7. Khi protocol chung đã khóa, chạy test với đúng model/prompt/ngưỡng đã chọn.
8. Viết phần B và đọc ví dụ lỗi cụ thể; chuyển scores/metrics cho bảng so sánh chung.

Validation gồm 5.426 × 28 = 151.928 cặp text/giả thuyết, có thể xử lý theo batch.
Neutral của MNLI khác neutral cảm xúc; giải thích kết quả B theo ground truth và ngữ cảnh.

**Nghiệm thu B:** scores N×28 đúng ID/mapping, đủ metrics, không fine-tune, prompt/revision
truy được, kết quả gốc và hiệu chỉnh được phân biệt, có phần viết và ví dụ lỗi.

## 6. Quốc Khánh — BERT và phần đầu báo cáo

### C1 BERT

1. Checkpoint `google-bert/bert-base-uncased`; khóa revision.
2. Tạo script fine-tune tham số hóa checkpoint, seed và cấu hình để Phúc/Huy dùng.
3. Pilot tokenizer, nhãn multi-hot, loss và luồng lưu checkpoint/scores.
4. Chạy các seed đề xuất **42, 123, 2026**; giữ log và cấu hình riêng từng run.
5. Chọn checkpoint bằng Macro-F1 validation @0.5; xuất sigmoid scores theo ID.
6. Tính số từng seed và mean ± sample std, `ddof=1`, n=3; phân tích lỗi BERT.
7. Bàn giao checkpoint, cấu hình, scores và phần phương pháp/kết quả C1.

### Phần đầu báo cáo — ghi nhận theo tin nhắn của Khánh

- Thông tin đề tài/nhóm, bố cục và mục tiêu.
- Tóm tắt bài báo GoEmotions bằng lời nhóm: dữ liệu, đóng góp, bài toán đa nhãn.
- Giới thiệu nguồn dữ liệu, split, nhãn và EDA; lấy bảng/biểu đồ đã kiểm của nhóm.
- Mô tả thiết kế so sánh A/B/C/D và quy trình đánh giá chung.
- Đặt mẫu bảng/cách trích nguồn để các bạn ghép phần riêng vào thống nhất.
- Nhận phần A từ Duy, B từ Đức Trí, C2 từ Phúc, C3/demo từ Huy để tổng hợp.
  Mỗi người vẫn tự viết phương pháp, kết quả và hạn chế của phần mình.

**Nghiệm thu Khánh:** BERT ≥3 seed có log/scores và mean±std; phần đầu báo cáo rõ,
có nguồn, không gán kết quả chưa chạy thành số thực nghiệm.

## 7. Hoàng Phúc — RoBERTa và tổng hợp nâng cao

1. Checkpoint `FacebookAI/roberta-base`; dùng tokenizer tương ứng và khóa revision.
2. Dùng script chung, kiểm riêng tokenization/truncation; chạy seed **42, 123, 2026**.
3. Xuất scores/checkpoint/log riêng từng seed; tính metrics và mean±std cho C2.
4. Viết phương pháp, kết quả và lỗi RoBERTa; đối chiếu với BERT/DistilBERT theo cùng ID.
5. Điều phối tiêu chí nhãn hiếm từ train, nhận bảng F1/support của A/B/C.
6. Tổng hợp trước/sau nâng cao, giữ cả nhãn tăng, không tăng hoặc giảm.
7. Tổng hợp các cặp FN nhãn A + FP nhãn B nổi bật từ dự đoán; đọc văn bản có ID.

**Phúc điều phối bảng nâng cao; người sở hữu từng model tự tạo scores/ngưỡng đúng model.**
Tuning riêng C2 không bắt Phúc huấn luyện thay C1/C3.

**Nghiệm thu Phúc:** RoBERTa ≥3 seed với mean±std; phần C2; bảng nhãn hiếm/nâng cao
và cặp lỗi có nguồn artifact, tách kết quả đã chạy với cấu hình dự kiến.

## 8. Nhật Huy — DistilBERT và demo

1. Checkpoint `distilbert/distilbert-base-uncased`; khóa revision/tokenizer.
2. Pilot bằng script chung; chạy seed **42, 123, 2026**, lưu checkpoint/log/scores.
3. Tính metrics, mean±std; viết phương pháp, kết quả và lỗi DistilBERT.
4. Nhận checkpoint và bộ ngưỡng của kiến trúc C thắng theo validation để dựng demo.
5. Dùng Gradio/Streamlit: nhập tiếng Anh, hiện nhiều nhãn cùng score; xử lý rỗng,
   text dài/truncation và trường hợp không nhãn nào vượt ngưỡng.
6. Kiểm một input cho cùng score/nhãn giữa demo và script suy luận tương ứng.
7. Viết hướng dẫn cài/chạy, lưu revision/cấu hình, bàn giao link hoặc ảnh/video minh chứng.

**Nghiệm thu Huy:** DistilBERT ≥3 seed và mean±std; demo best C chạy được,
khởi động lại được, cùng mapping/ngưỡng và có hướng dẫn.

## 9. Quy tắc fine-tune và chọn mô hình chung

- Tokenizer riêng → encoder → head **28 logits**.
- Huấn luyện dùng `BCEWithLogitsLoss` với multi-hot float; đưa logits trực tiếp vào
  loss. Sigmoid dùng khi xuất scores; không softmax chung 28 nhãn.
- Pilot đề xuất: max_length 128, batch 16, learning rate 2e-5, epoch 3.
  Điều chỉnh theo thiết bị/validation, chốt cấu hình và ghi effective batch nếu tích lũy gradient.
- Ba người mỗi người làm trọn ba seed của kiến trúc mình; giữ số từng seed trước bảng mean±std.
- Bảng chính dùng mốc @0.5; các hàng tuning/weighting phải ghi rõ.
- Chọn kiến trúc C bằng **mean Macro-F1 validation** của ba seed. Nhóm chốt cách xử
  lý khi gần bằng/hòa, cân nhắc std và chi phí.
- Đề xuất checkpoint demo: checkpoint có Macro-F1 val tốt nhất trong kiến trúc thắng;
  nói rõ đó là checkpoint demo, khác điểm trung bình kiến trúc.
- Không chọn kiến trúc, seed hoặc ngưỡng bằng test.

## 10. Nâng cao và phân tích lỗi

### Nhãn hiếm và ngưỡng

Năm nhãn hiếm theo train: grief 77, pride 111, relief 153, nervousness 164,
embarrassment 303. Ghi F1/P/R/support trước/sau, kể cả nhãn giảm.

Chọn ngưỡng từ validation của **đúng model/seed**; giữ cấu hình gốc @0.5.
Không chuyển ngưỡng A sang B/C. Nếu dùng `pos_weight`, tính từ train rồi train
lại; ngưỡng của model cũ không thuộc model weighted mới. Tuning không cần train lại.
Chốt lưới và luật hòa trước khi so sánh; điểm tuned trên cùng val có thể lạc quan.
Mở rộng weighted C/contrastive khi đã có đầy đủ các cấu hình chính.

### Ít nhất ba nhóm lỗi giữa C1/C2/C3

| Nhóm lỗi | Cách xác định | Bằng chứng bàn giao |
|---|---|---|
| Cảm xúc gần nghĩa | FN nhãn thật A cùng FP nhãn B trong một câu | Cặp nhãn, số đếm, ID/text và prediction ba C |
| Bỏ sót cảm xúc thứ hai | True có ≥2 nhãn, model chỉ tìm một phần | True/predicted labels, scores/ngưỡng và so sánh ba C |
| Hàm ý, phủ định hoặc thiếu ngữ cảnh | Chọn FN/FP rồi đọc thủ công trước khi phân loại nguyên nhân | Ví dụ cụ thể và nhận xét model nào sai giống/khác nhau |
| Nhãn hiếm hoặc neutral | Support thấp, FN/FP cụ thể | F1/P/R/support và độ ổn định qua seed |

Khánh/Phúc/Huy đưa lỗi của C mình; Phúc tổng hợp; Trí bổ sung lỗi B và Duy bổ sung
lỗi A khi cần đối chiếu. Chọn ít nhất ba nhóm có minh chứng thật.
Đồng xuất hiện nhãn thật trong EDA khác lỗi prediction; không cộng cặp FN/FP
để suy ra tổng câu lỗi vì một câu có thể góp nhiều cặp.

## 11. Trình tự triển khai theo đầu việc

1. Thống nhất split, mapping, metric, file scores, seed và cấu hình; chia sẻ giao diện data/metrics.
2. Đức Trí dựng B; Khánh/Phúc/Huy pilot C riêng. Khánh chia sẻ script chung cho hai bạn.
3. Mỗi người xuất đầy đủ validation scores, log/checkpoint và phần báo cáo mình.
4. Ghép A/B/C, đủ chín run C và mean±std; phân tích lỗi, chọn best C.
5. Huy tích hợp demo; Phúc tổng hợp nâng cao/nhãn hiếm từ artifacts của mỗi người.
6. Khóa protocol, model/prompt/ngưỡng và danh sách configs; đánh giá test sau đó.
7. Ghép báo cáo, kiểm nguồn/số liệu/đóng góp, chuẩn bị demo và phản biện.

Mỗi bước có thể phối hợp song song khi đủ đầu vào. Kế hoạch không đặt thời lượng.

## 12. Các báo cáo và bộ sản phẩm cuối

| Sản phẩm | Nội dung phải có | Người phối hợp |
|---|---|---|
| Báo cáo tiến độ lần 1 | Tóm tắt paper/EDA, A và B có số cụ thể, tiến độ seed của ba C, phân công thực tế | Khánh tổng hợp; Duy/Trí/Phúc/Huy cung cấp phần mình |
| Báo cáo tiến độ lần 2 | Đủ ba C × ≥3 seed, mean±std so với A/B, ≥3 nhóm lỗi, minh chứng demo và nâng cao | Khánh tổng hợp; Phúc bảng nâng cao/lỗi; Huy demo |
| Báo cáo cuối | Phương pháp/cấu hình/kết quả, nâng cao/nhãn hiếm, giải thích std, hạn chế, nguồn, đóng góp thực tế, code và demo | Cả nhóm |
| Artifacts | Checkpoint/log/config, scores theo ID, metric JSON/CSV, threshold đúng model/seed | Từng người sở hữu model |

Bảng cần ghép: A/B/C gốc, chín run C, mean±std từng kiến trúc, nâng cao trước/sau,
F1/support nhãn hiếm và bảng lỗi đối chiếu. Chưa có số thì ghi chưa đo, không điền số giả định.
Tỷ lệ đóng góp lấy từ công việc thực tế, không tự chia đều.

## 13. Tài liệu và nguồn dùng chung

- [Bài báo GoEmotions — ACL 2020](https://aclanthology.org/2020.acl-main.372/).
- [Dữ liệu Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).
- [BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli).
- [BERT](https://huggingface.co/google-bert/bert-base-uncased), [RoBERTa](https://huggingface.co/FacebookAI/roberta-base), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased).
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
- [Chọn ngưỡng trên validation](https://scikit-learn.org/1.7/modules/classification_threshold.html).
- [Notebook EDA](../notebooks/eda.ipynb), [EDA bổ sung](../notebooks/eda_extra.ipynb), [báo cáo dữ liệu](../reports/THONG_KE_DU_LIEU.md).
- [Trang kế hoạch Notion](https://app.notion.com/p/3ed7c27769028185af2dfbaac4c4586b).

## Mục riêng của Duy

[Phần của Duy — baseline, tài liệu và báo cáo](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d).
Notebook, hướng dẫn, bảng kết quả, hồ sơ đối chiếu và báo cáo baseline nằm trong mục này.
