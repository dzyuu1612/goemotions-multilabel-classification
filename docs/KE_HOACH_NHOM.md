# Kế hoạch nhóm — Đề tài 1: GoEmotions

Cập nhật **02/10/2026**. Kho chung: `trangkhanh-ai/goemotions-multilabel-classification`.
Người làm baseline: **Bảo Duy Nguyễn — dzyuu1612**. Các tên TV2–TV4 chưa được điền
trong bản phân công nguồn; nhóm bổ sung tên thật trước khi nộp.

## 1. Mục tiêu và cơ sở lập kế hoạch

Phân loại cảm xúc đa nhãn trong bình luận Reddit tiếng Anh, so sánh A cổ điển,
B zero-shot và C fine-tune; làm demo và giải thích kết quả bằng số liệu/lỗi cụ thể.

Nguồn đối chiếu là ảnh đề tài 1 và nội dung yêu cầu chung của cô do bạn cung cấp,
bản phân công ngày 28/09 và kế hoạch cập nhật ngày 30/09. Bản này cập nhật tiến độ A
và sửa điểm phân công chưa khớp yêu cầu. Tài liệu kỹ thuật tham khảo ở mục 14.

| Mức | Nội dung |
|---|---|
| Yêu cầu cô | A: baseline cổ điển, một SV; B: pretrained dùng trực tiếp không fine-tune, làm chung hoặc cùng người A |
| Yêu cầu cô | C: ba kiến trúc khác nhau, ba SV còn lại mỗi người sở hữu trọn một kiến trúc; mỗi kiến trúc ít nhất ba seed, mean ± std |
| Yêu cầu cô | D: Gradio/Streamlit từ model tốt nhất trong ba kiến trúc C; ít nhất ba nhóm lỗi và so sánh C1/C2/C3 |
| Yêu cầu đề tài 1 | Split chính thức; khảo sát phân bố nhãn, tiền xử lý, TF-IDF + LR, fine-tune; xử lý ngưỡng đa nhãn; Macro/Micro-F1, P/R, Hamming |
| Yêu cầu nâng cao | Class weighting hoặc ngưỡng riêng từng nhãn hoặc contrastive; có số F1 nhãn hiếm trước/sau |
| Đề xuất nhóm | BART-MNLI cho B; BERT/RoBERTa/DistilBERT cho C; seed 42/123/2026; ưu tiên tuning để tiết kiệm GPU |

Ảnh đề 1 ghi nâng cao để đạt mức 8–10; rubric chung ghi nâng cao trong mức 9–10.
Nhóm nên hoàn thành nâng cao cùng toàn bộ A/B/C/D. Không coi đây là cam kết điểm số.
PDF không bắt nâng cao chỉ ở C: kết quả nâng cao A có thể đóng góp bằng chứng;
áp dụng thêm ngưỡng cho C là đề xuất có ích, cần báo kết quả thật.

## 2. Phân công bốn người

**Bảng dưới là phương án theo kế hoạch cập nhật, cần nhóm điền tên TV2–TV4 và xác
nhận người chủ trì các việc phối hợp. Trạng thái chưa có bằng chứng được ghi là chưa kiểm chứng.**

| Thành viên | Sở hữu chính | Việc phối hợp đề xuất | Sản phẩm riêng | Trạng thái kiểm chứng |
|---|---|---|---|---|
| TV1 — Bảo Duy Nguyễn (`dzyuu1612`) | A: TF-IDF + OvR LR gốc và nâng cao, metrics/analysis A | Phối hợp B với TV2; chuẩn đầu ra; tổng hợp bảng khi B/C có số | Notebook A, model/scores local, metric, bảng nhãn hiếm/cặp lỗi, báo cáo A | A validation đã chạy và rà soát |
| TV2 — chưa điền tên | C1: BERT-base, trọn ba seed | Chủ trì B zero-shot cùng TV1; khởi tạo script fine-tune chung | Ba checkpoint/log/scores; phần BERT; B val/test nếu nhận chủ trì | Chưa có run trong repo đã kiểm |
| TV3 — chưa điền tên | C2: RoBERTa-base, trọn ba seed | Tích hợp demo best C, phối hợp QA demo | Ba checkpoint/log/scores; phần RoBERTa; app/demo và hướng dẫn | Chưa có run/app trong repo đã kiểm |
| TV4 — chưa điền tên | C3: DistilBERT-base, trọn ba seed | Điều phối bảng nhãn hiếm/nâng cao; tổng hợp các nhóm lỗi | Ba checkpoint/log/scores; phần DistilBERT; bảng trước/sau | Chưa có run trong repo đã kiểm |
| Cả nhóm | Đọc paper, bàn protocol, phân tích ≥3 nhóm lỗi C1/C2/C3, báo cáo/phản biện | Review chéo, lịch GPU, bảng đóng góp | Báo cáo tuần 4/7/9 và demo cuối | Cần cập nhật theo run thật |

Mỗi người viết phương pháp, cấu hình, kết quả và hạn chế cho phần mình; người tổng
hợp ghép theo mẫu chung. Phần EDA đã tồn tại từ các commit khác của nhóm; không gán
toàn bộ EDA cho TV1. Bản phân công cũ giao TV4 cả C2+C3 và cho phép một seed cần
được thay bằng một người/một C và ít nhất ba seed cho mỗi C.

## 3. Dữ liệu và hợp đồng dùng chung

| Hạng mục | Quy ước |
|---|---|
| Dataset | `google-research-datasets/go_emotions`, config `simplified` |
| Revision cố định | `add492243ff905527e67aeb8b80c082af02207c3` |
| Split | Train 43.410 / validation 5.426 / test 5.427; tổng 54.263 |
| Nhãn | 27 cảm xúc + neutral; thứ tự chính xác trong `data/labels.json`, N×28 multi-hot |
| Văn bản | Giữ văn bản nguồn; A dùng TF-IDF, B/C dùng tokenizer của checkpoint |
| Nạp dữ liệu | Tái sử dụng `src/data.py`; giữ ID, kiểm SHA-256 và mapping |
| Prediction file | NPZ gồm `ids`, `scores` N×28 và `label_names`; join theo ID |
| Scores | Hữu hạn, 0–1; lưu trước khi threshold, không chỉ lưu hard labels |
| Metrics | Dùng chung `src/metrics.py`; đủ 28 nhãn, `zero_division=0` |
| Train | Fit TF-IDF/weighting/model A/C; không fit trên validation/test |
| Validation | Chọn config/checkpoint/prompt nếu có/ngưỡng; công bố việc dùng nhãn |
| Test | Suy luận và tính metric sau freeze; không điều chỉnh lựa chọn theo test |

58.009 là số bình luận nguồn thô, 54.263 là simplified sử dụng trong đồ án.
Không gộp 28 nhãn thành positive/negative/neutral trong bảng chính; không tự ép
neutral loại trừ cảm xúc khác. EDA đã xem thống kê test mô tả, nhưng lựa chọn mô
hình/ngưỡng và phân tích lỗi phát triển phải dựa trên train/validation.

## 4. A — kế hoạch và sản phẩm baseline của TV1

| Mã | Công việc | Đầu ra | Trạng thái |
|---|---|---|---|
| A1 | Kiểm dữ liệu và multi-hot; tái sử dụng module có sẵn | Mapping 28 nhãn, train/val đúng số, checks ID/hash | Đã có |
| A2 | Metric chung micro/macro P/R/F1, Hamming, per-label | `src/metrics.py`, kiểm bằng ví dụ tính tay | Đã có |
| A3 | TF-IDF + 28 LR nhị phân, bản standard | Full fit/train, validation scores/metrics, model | Đã chạy |
| A4 | Bản balanced; ngưỡng 0.5/chung/riêng | Hai model × ba luật quyết định, bảng sáu hàng | Đã chạy validation |
| A5 | Phân tích năm nhãn hiếm, FN/FP và ví dụ lỗi | CSV/JSON/report, ID kiểm chứng, top features | Đã có |
| A6 | Notebook đơn giản và hướng dẫn học/chạy | 22 cell, 11 cell mã có output thật; thứ tự đọc | Đã có |
| A7 | Xuất bảng nhỏ và bàn giao GitHub | JSON/CSV đã kiểm hash; branch/PR và README | Đã push fork của TV1, PR #3 vào repo chung đang chờ merge |
| A8 | Phối hợp B và ghép bảng A/B/C | B benchmark và bảng toàn nhóm từ số thật | Chờ B/C; chưa hoàn thành |
| A9 | Freeze rồi test cuối | Protocol sáu cấu hình, test metrics và rare-label changes | Mã đã chuẩn bị, chưa chạy test thật |

Cấu hình đã chạy: unigram+bigram, `min_df=2`, `max_features=100000`, thực tế
58.338 đặc trưng; `OneVsRestClassifier(LogisticRegression(C=1, solver='liblinear',
max_iter=1000, random_state=42))`. Standard `class_weight=None`, balanced
`class_weight='balanced'` cho từng bài toán nhị phân. Không đổi TF-IDF giữa hai bản.

Lưới ngưỡng A 0.05–0.95 bước 0.05, chọn bằng Macro-F1 cho ngưỡng chung và F1 từng
nhãn cho ngưỡng riêng; hòa chọn gần 0.5, rồi mức lớn hơn. Lưới là lựa chọn thực
nghiệm của nhóm, không phải yêu cầu cô. Scores/ngưỡng liên kết hash đúng model.

## 5. B — zero-shot không fine-tune

1. Người chủ trì B + TV1 pilot khoảng 100 câu validation để đo tốc độ và RAM/VRAM.
2. Dùng HF pipeline trực tiếp với `facebook/bart-large-mnli`, đủ 28 candidate labels,
   `multi_label=True`; hypothesis template khởi đầu `This text expresses {}.`.
3. Mapping score về đúng `data/labels.json` vì pipeline trả nhãn theo score giảm dần.
4. Chạy đủ validation theo batch, lưu từng phần và ID để tiếp tục khi Colab ngắt.
5. Báo cấu hình cố định 0.5 trước. Nếu dùng nhãn val chọn template/ngưỡng, giữ hàng
   riêng, ghi rõ zero-shot về trọng số nhưng đã hiệu chỉnh bằng nhãn dữ liệu đích.
6. Lưu checkpoint revision, template, batch, thiết bị, thư viện, thời gian, scores.
7. Khóa các lựa chọn trước test; sau đó chạy cùng split test như A/C.

5.426 × 28 = 151.928 cặp văn bản/giả thuyết trên validation, không phải 151.928
API call. Có thể xử lý nhiều cặp trong một batch; không suy ra thời gian nếu chưa
đo pilot. Neutral của MNLI không đồng nhất neutral cảm xúc; đọc lỗi B theo ngữ cảnh.

## 6. C — ba kiến trúc, mỗi kiến trúc ba seed

| Kiến trúc | Checkpoint dự kiến | Chủ sở hữu | Seed chính |
|---|---|---|---|
| C1 BERT | `google-bert/bert-base-uncased` | TV2 | 42, 123, 2026 |
| C2 RoBERTa | `FacebookAI/roberta-base` | TV3 | 42, 123, 2026 |
| C3 DistilBERT | `distilbert/distilbert-base-uncased` | TV4 | 42, 123, 2026 |

Ba seed là đề xuất số cụ thể; số lượng ít nhất ba/kiến trúc là yêu cầu cô.
Thiết lập pilot đề xuất: max_length 128, batch 16, learning_rate 2e-5, epoch 3.
Điều chỉnh theo GPU và validation, chốt trước các run chính, ghi effective batch
nếu dùng gradient accumulation. Khóa revision checkpoint trước khi benchmark.

Luồng học: tokenizer riêng → encoder → head 28 logits → BCEWithLogitsLoss với
multi-hot float. Loss nhận logits trực tiếp; sigmoid dùng lúc xuất scores.
Không softmax chung 28 nhãn, không chuyển thành bài toán một lớp.

Mỗi người thực hiện đầy đủ: môi trường → pilot → ba seed → chọn checkpoint bằng
Macro-F1 validation @0.5 → lưu log/config/checkpoint/val scores → báo mean ± sample
std (`ddof=1`, n=3). Ghi số từng seed trước bảng trung bình. Không chọn seed đẹp
nhất để đại diện độ ổn định cả kiến trúc. Bảng baseline A không có yêu cầu ba seed.

## 7. Nâng cao và nhãn hiếm

Nhãn hiếm xác định từ support **train**: grief 77, pride 111, relief 153,
nervousness 164, embarrassment 303. Ghi F1, P/R và support cho cả năm nhãn,
bao gồm nhãn không tăng hoặc giảm. Số dương val 13–35 nên kết luận cần thận trọng.

- A đã có weighting và threshold tuning, đủ cơ sở báo thực nghiệm A trước/sau.
- Đề xuất C: chọn ngưỡng riêng từ val scores của từng model/seed; không cần train
  lại cho tuning. Giữ mốc 0.5 để tách ảnh hưởng tuning.
- Nếu thử `pos_weight`: tính `(N-positive)/positive` chỉ từ train, chốt cách cap
  bằng pilot val; huấn luyện lại và tạo bộ ngưỡng mới đúng weighted model.
- Ưu tiên hoàn tất chín run C trước khi mở rộng weighted C; contrastive là một
  hướng thay thế, không bắt phải làm đồng thời cả ba phương pháp.
- Bảng cuối ghi cấu hình gốc/weighting/tuning/kết hợp khi đã chạy thật; không ghi
  các biến thể chưa chạy như kết quả.

## 8. Chọn best C và demo

Chọn **kiến trúc C** theo mean Macro-F1 validation của ba seed ở mốc 0.5; nếu hòa
xét độ ổn định/chi phí theo quy tắc nhóm chốt. Không cho A/B thắng để thay best C
trong demo cô yêu cầu. Đề xuất chọn checkpoint có val Macro-F1 cao nhất trong
kiến trúc thắng để triển khai một app; nói rõ đây là checkpoint demo, không phải
điểm trung bình kiến trúc. Bộ ngưỡng phải thuộc checkpoint đó.

Demo Gradio/Streamlit: nhập tiếng Anh, hiện nhiều nhãn và score, thông báo nếu
không nhãn nào vượt ngưỡng; xử lý rỗng, text dài/truncation, khởi động lại được.
Kiểm cùng input với script evaluate cho cùng mapping/ngưỡng. Bàn giao hướng dẫn
cài, model revision, lệnh chạy, link hoặc video/ảnh minh chứng. Không cần giao diện cầu kỳ.

## 9. Phân tích ít nhất ba nhóm lỗi

| Nhóm | Cách nhận diện và minh chứng | So sánh cần làm |
|---|---|---|
| Cảm xúc gần nghĩa | FN nhãn A cùng FP nhãn B; đếm cả split, đọc ví dụ có ID | C1/C2/C3 có nhầm cùng cặp, cùng câu không? |
| Thiếu cảm xúc thứ hai | Ground truth ≥2 nhãn, model chỉ tìm một phần; đọc scores gần ngưỡng | Tuning giúp từng kiến trúc thế nào? |
| Hàm ý/mỉa mai/phủ định/ngữ cảnh | Lọc FN/FP rồi đọc thủ công, không tự coi mọi lỗi là mỉa mai | Ba model có cùng khó khăn ngữ cảnh không? |
| Nhãn hiếm/neutral | F1/support và lỗi cụ thể; neutral không bị loại trừ cưỡng bức | Recall ổn định qua seed hay thay đổi nhiều? |

Chọn ít nhất ba nhóm cụ thể sau khi đọc dữ liệu. Bảng gồm ID, text, true labels,
prediction C1/C2/C3, scores/ngưỡng và nhận xét. Đồng xuất hiện nhãn thật trong EDA
khác nhầm lẫn từ prediction. Một câu có thể góp nhiều cặp FN/FP; không cộng bảng
cặp để tính tổng câu lỗi. Không dùng số ví dụ chọn lọc làm tỷ lệ lỗi toàn split.

## 10. Lịch 9 tuần và báo cáo

Tuần tương đối theo ngày bắt đầu/hạn cô; ngày cụ thể chưa có trong nguồn. Nhóm
điền sau khi xác nhận lịch học, không suy ra tuần hiện tại từ ngày 02/10.

| Tuần | Công việc | Cổng hoàn thành |
|---|---|---|
| 1 | Đọc paper/yêu cầu, EDA, chia một A + ba C; chốt label/revision/metrics | Bảng tên thật, protocol, lịch GPU, mỗi người đọc dữ liệu |
| 2 | A + B full validation; mỗi người C setup và mini-run | A/B có số; C1/C2/C3 chạy pilot được |
| 3 | C chạy seed 42, 123; kiểm mapping/log; A/B kiểm lỗi | Sáu run C hoặc status/vướng mắc rõ |
| 4 | C chạy seed 2026; nộp tiến độ 1 | Tóm tắt paper/EDA, A+B có số cụ thể, tiến độ seed từng C, phân công và mức hoàn thành |
| 5 | Tổng hợp đủ chín run, mean ± std; chọn best C; bắt đầu demo | Chín dòng riêng + ba dòng tổng hợp; app bản đầu |
| 6 | ≥3 nhóm lỗi đối chiếu C1/C2/C3; hoàn thiện demo | Bảng lỗi có ID và app chạy local |
| 7 | Nâng cao, F1 nhãn hiếm; nộp tiến độ 2 | Đủ 3×3, mean±std với A/B, lỗi, minh chứng demo, kế hoạch/kết quả nâng cao |
| 8 | Freeze; chấm test theo protocol; viết báo cáo/slide | Test configs đã khóa, số truy được về artifacts, bản nháp đầy đủ |
| 9 | Kiểm số, tổng duyệt và nộp/phản biện | DOCX/PDF, nâng cao, giải thích std, % đóng góp thật, code và link demo, hạn chế/hướng phát triển |

## 11. Quy trình freeze và nghiệm thu

Trước test: chốt dataset/label/metrics, checkpoint từng seed, prompt B, thresholds,
bảng configs gốc+nâng cao, rare labels, quy tắc chọn best C, commit và nơi artifacts.
Các hệ thống phải áp dụng cùng protocol, không chỉ khóa riêng A.

Phần A đã có `scripts.freeze_baseline` khóa sáu configs và hash; tiếp theo
`scripts.evaluate_baseline_test` kiểm hash, suy luận một lần/model, lưu metric
các configs đã khóa, rare-label changes. Chạy lại cùng protocol đọc kết quả đã có.
**Chưa chạy hai lệnh này với test thật.**

Tiêu chí bàn giao từng C: ≥3 seed đã chạy; config/log, checkpoint, val/test scores
theo ID; P/R/F1/Hamming; mean±std; phần viết phương pháp/lỗi/hạn chế. Demo dùng
best C, bản báo cáo cuối phải có đóng góp thực tế, không tự chia 25% khi chưa đo.

## 12. Tiến độ đã kiểm chứng ngày 02/10

| Hạng mục | Đã có | Còn lại |
|---|---|---|
| EDA | Notebook/statistics từ đồng đội; đã nhập cập nhật main vào baseline branch | Nhóm tự kiểm và viết tóm tắt paper bằng lời mình |
| A | Hai model full train, sáu hàng validation, rare labels/cặp lỗi, notebook, 16 tests mã | Test cuối sau freeze; phối hợp B và bảng toàn nhóm |
| B | Mô tả/ ví dụ pipeline trong README | Chưa có benchmark val/test trong repo kiểm được |
| C1/C2/C3 | Kế hoạch/checkpoint | Chưa có chín run/checkpoint/scores trong repo kiểm được |
| Demo | Kế hoạch best C | Chưa có app kiểm chứng |
| Nâng cao | A weighting/tuning validation thực | Xác nhận test cuối; C nâng cao nếu chọn |

Các việc chưa có trong repo có thể đang ở Colab của đồng đội; trạng thái ở đây là
bằng chứng đã kiểm, không suy đoán người khác chưa làm. Khi nhận artifact cập nhật trạng thái.

## 13. Việc tiếp theo và phối hợp Git

1. TV1 đọc [thứ tự tài liệu](THU_TU_DOC_BASELINE.md), tập giải thích A và dùng
   [báo cáo cá nhân](../reports/BAO_CAO_BASELINE_BAO_DUY.md) làm bản nháp tiến độ.
2. Cả nhóm điền tên TV2–TV4, người B/demo/nâng cao, ngày bắt đầu/hạn báo cáo,
   nơi chia sẻ model/scores lớn và lịch GPU.
3. TV2 + TV1 triển khai B full val, đủ số cho cuối tuần 4.
4. TV2/TV3/TV4 pilot đúng C được giao rồi chạy ba seed; cập nhật run tracker:
   architecture/seed/config/val score/runtime/artifact/status.
5. Gộp code qua PR; xem diff từ main mới nhất. GitHub author của TV1 là `dzyuu1612`
   với email Git đã cấu hình. Giữ authorship EDA của đồng đội.
6. Model/raw/scores ở `data/processed/` không đưa vào Git; chia sẻ riêng. Các bảng
   validation nhỏ đã xuất tại `reports/baseline_validation/` để đọc trực tiếp.
7. Nghiệm thu các phần bắt buộc, freeze chung, test cuối, báo cáo/phản biện.

## 14. Nguồn phương pháp để học và trích dẫn

- [Demszky và cộng sự, GoEmotions — ACL 2020](https://aclanthology.org/2020.acl-main.372/): nguồn bài toán/dataset/BERT gốc.
- [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions): nhãn và split nguồn.
- [TF-IDF, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
- [OneVsRestClassifier, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html).
- [LogisticRegression, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html): weighting và solver.
- [Threshold tuning, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/classification_threshold.html): validation tách train, tránh dùng test chọn ngưỡng.
- [BART-large-MNLI model card](https://huggingface.co/facebook/bart-large-mnli): pipeline zero-shot/multi_label.
- [BERT](https://huggingface.co/google-bert/bert-base-uncased), [RoBERTa](https://huggingface.co/FacebookAI/roberta-base), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased): checkpoint dự kiến, cần khóa revision khi chạy.
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): loss đa nhãn dùng logits.

Nguồn cô cung cấp là căn cứ yêu cầu môn học. Các nguồn kỹ thuật giải thích công cụ,
không thay thế rubric của cô hoặc chứng minh nhóm đã chạy B/C.

## 15. Nơi lưu kế hoạch và bàn giao

- [Notion mới của đồ án](https://app.notion.com/p/3ed7c27769028185af2dfbaac4c4586b): toàn bộ kế hoạch, phân công, thứ tự đọc và báo cáo cá nhân.
- [Nhánh baseline đã push](https://github.com/dzyuu1612/goemotions-multilabel-classification/tree/codex/baseline-starter): code/notebook/tables của TV1.
- [PR #3 vào repo chung](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/3): chưa merge ngày 02/10; đang chờ nhóm review.

Tài khoản GitHub `dzyuu1612` hiện chưa có quyền ghi repo chung, nên bàn giao qua
fork và PR. Tên/email Git của các commit TV1 đã kiểm: `dzyuu1612` /
`baoduynguyen1612@gmail.com`. Không thay author các commit EDA của đồng đội.
