# GoEmotions: Phân loại cảm xúc đa nhãn trên văn bản mạng xã hội

## Thống kê dữ liệu đã khám phá — đã hoàn thành

Phần EDA đã chạy trên **54.263 mẫu GoEmotions simplified**, giữ nguyên train/validation/test
43.410/5.426/5.427 mẫu. Các sản phẩm đáp ứng ba nội dung yêu cầu:

| Nội dung yêu cầu | Kết quả và tệp xem trực tiếp |
|---|---|
| Số mẫu | [Số mẫu và tỷ lệ từng split](reports/tables/split_sizes.csv) |
| Phân bố nhãn/lớp | [Số mẫu dương và tỷ lệ của đủ 28 nhãn](reports/tables/label_distribution.csv), gồm từng split và tổng |
| Ví dụ mẫu dữ liệu | [30 ví dụ nguyên văn, có ID, nhãn và nhận xét](reports/tables/examples_30.csv), phủ đủ 28 nhãn |

- [Notebook giải thích từng bước bằng tiếng Việt, có output](notebooks/eda.ipynb).
- [Báo cáo thống kê chi tiết và biểu đồ](reports/THONG_KE_DU_LIEU.md).
- [Hướng dẫn cài đặt và chạy lại EDA](docs/EDA.md).
- [Bản HTML đã chạy](reports/eda.html): tải về và mở bằng trình duyệt.

Đã đối chiếu số mẫu, support/tỷ lệ từng nhãn và 30 ví dụ với dữ liệu nguồn.
Với bài toán đa nhãn, một bình luận được tính vào nhiều nhãn nên tổng support có thể
lớn hơn số mẫu. Phần triển khai hiện tại là EDA; các phương pháp mô hình trong phần
giới thiệu dưới đây là kế hoạch của đồ án.

## Bài toán và bài báo nền tảng

GoEmotions được giới thiệu trong bài báo của Demszky và cộng sự tại ACL 2020. Bài báo xây dựng dữ liệu bình luận Reddit với 27 cảm xúc và neutral, đồng thời nghiên cứu mô hình dựa trên BERT.

Hai đặc điểm của bài toán cần phân biệt:

- **Fine-grained:** phân biệt nhiều loại cảm xúc chi tiết, thay vì chỉ tích cực/tiêu cực.
- **Multi-label:** một bình luận có thể mang nhiều nhãn cùng lúc.

Đây là hai khái niệm khác nhau. Đề tài này có cả hai đặc điểm. Các thí nghiệm TF-IDF + Logistic Regression và zero-shot BART-MNLI là thiết kế của đồ án; không mô tả chúng như toàn bộ phương pháp của bài báo gốc.

## Dữ liệu sử dụng

Nguồn GoEmotions ban đầu có 58.009 bình luận. Đồ án sử dụng phiên bản `simplified` đã lọc theo mức đồng thuận của người gán nhãn, với 54.263 mẫu:

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 43.410 | Fit TF-IDF; huấn luyện mô hình có giám sát; tính trọng số nhãn |
| Validation | 5.426 | Chọn cấu hình, checkpoint và ngưỡng theo thiết kế đã công bố |
| Test | 5.427 | Đánh giá cuối sau khi đã khóa các lựa chọn |

Giữ split chính thức và thứ tự 28 nhãn từ metadata. Biểu diễn nhãn bằng vector multi-hot: mỗi vị trí là 0 hoặc 1; một hàng có thể có nhiều số 1. Không thay nhãn gốc và không tự ép neutral loại trừ các nhãn khác.

Dataset revision trong mã EDA hiện tại: `add492243ff905527e67aeb8b80c082af02207c3`.

## Trạng thái triển khai

Repo hiện có mã và sản phẩm EDA. Nhóm đã thống nhất các hệ thống dự kiến dưới đây nhưng chưa công bố kết quả benchmark mô hình trong repo.

- [Notebook EDA](notebooks/eda.ipynb)
- [Báo cáo EDA](reports/THONG_KE_DU_LIEU.md)
- [Số mẫu](reports/tables/split_sizes.csv)
- [Phân bố nhãn](reports/tables/label_distribution.csv)
- [Ví dụ dữ liệu](reports/tables/examples_30.csv)
- [Cách chạy EDA](docs/EDA.md)

## Ba hướng tiếp cận đã thống nhất

| Hướng | Hệ thống dự kiến | Học thêm từ nhãn GoEmotions? |
|---|---|---|
| A — Baseline | TF-IDF + One-vs-Rest Logistic Regression | Có, trên train |
| B — Zero-shot | BART-large-MNLI | Không cập nhật trọng số |
| C — Fine-tuning | BERT-base, RoBERTa-base, DistilBERT-base | Có, mỗi mô hình trên train |

TF-IDF là cách biểu diễn văn bản cho A, không phải một mô hình phân loại riêng. Zero-shot nghĩa là không fine-tune mô hình trên GoEmotions; BART-MNLI đã được huấn luyện trước trên MNLI.


Ba hướng tiếp cận cùng giải quyết một bài toán. Mỗi hướng tự tạo dự đoán từ văn bản; đầu ra của A không phải đầu vào bắt buộc của B hoặc các mô hình fine-tune. Nhóm dự kiến so sánh tổng cộng **năm hệ thống**: một baseline, một zero-shot và ba mô hình fine-tune.

```text
GoEmotions: split gốc + 28 nhãn + quy tắc đánh giá chung
    ├── A: TF-IDF + One-vs-Rest Logistic Regression
    ├── B: zero-shot facebook/bart-large-mnli
    └── C: fine-tune BERT-base / RoBERTa-base / DistilBERT-base
                          ↓
            So sánh trên cùng tập test và phân tích lỗi
```

Danh sách checkpoint dự kiến: `google-bert/bert-base-uncased`, `FacebookAI/roberta-base`, `distilbert/distilbert-base-uncased`. Ba mô hình này thuộc **cùng hướng fine-tuning**, nhưng được huấn luyện và đánh giá riêng.

### A — TF-IDF + One-vs-Rest Logistic Regression

TF-IDF chuyển văn bản thành vector đặc trưng từ/cặp từ. One-vs-Rest huấn luyện một Logistic Regression nhị phân cho mỗi nhãn: tổng cộng 28 bộ phân loại.

```text
Huấn luyện: text train → fit TF-IDF → X_train
                                      + Y_train multi-hot
                                      → fit 28 Logistic Regression

Dự đoán: text → transform bằng TF-IDF đã fit
             → 28 điểm dự đoán → ngưỡng từng nhãn → tập nhãn
```

Chỉ fit từ vựng/IDF và mô hình trên train. Khởi đầu có thể dùng ngưỡng 0,5; nếu điều chỉnh, chọn bằng validation và ghi rõ. TF-IDF là bộ biểu diễn đặc trưng, không phải mô hình tự phân loại cảm xúc.

### B — Zero-shot BART-large-MNLI

Checkpoint `facebook/bart-large-mnli` đã được huấn luyện trên tác vụ suy luận ngôn ngữ tự nhiên MNLI. Đồ án sử dụng checkpoint đó mà không huấn luyện thêm trọng số trên GoEmotions.

```text
Text + 28 nhãn ứng viên + hypothesis template
    → tạo cặp text/giả thuyết cho từng nhãn
    → BART-MNLI → 28 score → ngưỡng → tập nhãn
```

Nhãn ứng viên là đầu vào do nhóm cung cấp, không phải kết quả mô hình sinh ra. Bật `multi_label=True` để đánh giá độc lập từng nhãn. Score NLI không phải xác suất cảm xúc đã được hiệu chuẩn. Nhãn neutral của tác vụ MNLI và nhãn neutral của GoEmotions có ý nghĩa khác nhau.

Ví dụ suy luận một câu — không phải mã đánh giá benchmark:

```python
import json
from pathlib import Path
from transformers import pipeline

# Chạy từ thư mục gốc repo; file này đã có trong repo.
labels = json.loads(Path("data/labels.json").read_text(encoding="utf-8"))
assert len(labels) == 28

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)
result = classifier(
    "I am really grateful for everything you have done.",
    candidate_labels=labels,
    hypothesis_template="This text expresses {}.",
    multi_label=True,
)
# 0,5 là ngưỡng khởi đầu cho ví dụ, chưa phải ngưỡng đã tối ưu.
predicted = [label for label, score in zip(result["labels"], result["scores"])
             if score >= 0.5]
print(predicted)
```

Khi đánh giá, ánh xạ score về thứ tự nhãn chuẩn vì pipeline trả nhãn theo score giảm dần. Báo cáo B ban đầu có thể dùng ngưỡng cố định. Nếu dùng nhãn validation để chỉnh template/ngưỡng, phải mô tả là zero-shot về trọng số, có hiệu chỉnh bằng dữ liệu đích, và tách khỏi thiết lập không sử dụng nhãn đích.

### C — Fine-tune BERT-base, RoBERTa-base và DistilBERT-base

Ba checkpoint dự kiến được nêu ở trên. Với **mỗi** mô hình, thêm đầu phân loại có 28 đầu ra và huấn luyện riêng trên GoEmotions train; trong thiết lập fine-tune toàn bộ, cập nhật cả encoder và đầu phân loại. Sơ đồ dưới đây dùng BERT để minh họa; RoBERTa và DistilBERT có luồng tương tự nhưng dùng tokenizer/encoder tương ứng.

```text
Text → tokenizer BERT → encoder BERT → đầu phân loại → 28 logits
Huấn luyện: logits + nhãn multi-hot → BCEWithLogitsLoss → cập nhật trọng số
Suy luận: logits → sigmoid từng nhãn → ngưỡng từng nhãn → tập nhãn
```

Không dùng softmax chung 28 nhãn để ép chọn duy nhất một cảm xúc. `BCEWithLogitsLoss` nhận logits trực tiếp; không áp sigmoid trước loss này. Sigmoid được dùng khi chuyển logits thành điểm để dự đoán.

## Quy trình đánh giá chung

1. Thống nhất split, thứ tự nhãn và ID mẫu; giữ cùng văn bản nguồn, mỗi phương pháp có bộ biểu diễn/tokenizer phù hợp.
2. Huấn luyện A/C trên train; B giữ nguyên trọng số.
3. Với nhánh sử dụng validation, chọn cấu hình/checkpoint/ngưỡng trên validation; công bố rõ việc dùng nhãn validation cho B nếu có.
4. Khóa cấu hình rồi đo trên cùng test. Không dùng test để chọn mô hình hoặc ngưỡng.
5. Báo cáo Micro-F1, Macro-F1, Precision/Recall có ghi cách lấy trung bình, Hamming Loss và P/R/F1/support từng nhãn. Macro-F1 tính trên đủ 28 nhãn, thống nhất cách xử lý mẫu số bằng 0.
6. Ghi checkpoint/revision, seed, phần cứng, thư viện, thời gian fit và suy luận. Với nhiều seed fine-tune, báo cáo kết quả từng seed và trung bình/độ lệch chuẩn.

Không giả định trước zero-shot tốt hơn baseline hoặc fine-tuning tốt nhất. Thứ hạng phải dựa trên số liệu thực nghiệm.

## Phần nâng cao và nhãn hiếm

Hướng đề xuất là khảo sát class weighting và ngưỡng riêng từng nhãn:

- A: thử trọng số lớp trong từng Logistic Regression nhị phân.
- C: thử `pos_weight` trong BCEWithLogitsLoss; tính từ train, chọn cách xử lý trọng số quá lớn bằng validation nếu cần.
- Chọn ngưỡng từng nhãn từ validation của chính mô hình đó. Không chuyển nguyên ngưỡng A sang B/C.
- Xác định tập nhãn hiếm từ train và công bố tiêu chí trước khi so sánh.
- So sánh thiết lập gốc, chỉ weighting, chỉ tuning và kết hợp; báo cáo F1/support của tất cả nhãn hiếm đã xác định, kể cả nhãn giảm điểm.

Đây là hướng thử nghiệm nhằm cải thiện nhãn hiếm, chưa phải tuyên bố rằng đã cải thiện. Contrastive representation là hướng khác được đề bài nêu; không coi mọi phương án là đồng thời bắt buộc khi chưa làm rõ cách hiểu yêu cầu.

## Phân tích lỗi

Phân biệt **đồng xuất hiện** (nhãn thật cùng có trong một mẫu) với **nhầm lẫn** (nhãn dự đoán thừa/bỏ sót so với ground truth). Các cặp như anger/annoyance chỉ nên ghi là ví dụ cần khảo sát trước khi có minh chứng.

Dùng dự đoán theo ID, thống kê lỗi và ví dụ nguyên văn để chứng minh; tránh gán một cặp là thường xuyên nhầm chỉ dựa trên trực giác. Trong bài toán đa nhãn, confusion matrix dạng single-label không đủ mô tả toàn bộ lỗi.

## Kết quả thực nghiệm

| Hệ thống | Thiết lập dự kiến | Trạng thái |
|---|---|---|
| A | TF-IDF + One-vs-Rest Logistic Regression | Dự kiến, chưa có số đo trong repo |
| B | BART-large-MNLI, multi_label=True | Dự kiến, chưa có số đo trong repo |
| C1 | BERT-base-uncased, 28 đầu ra | Dự kiến, chưa có số đo trong repo |
| C2 | RoBERTa-base, 28 đầu ra | Dự kiến, chưa có số đo trong repo |
| C3 | DistilBERT-base-uncased, 28 đầu ra | Dự kiến, chưa có số đo trong repo |

Khi có kết quả, thay bảng trạng thái bằng bảng số liệu và dẫn đến notebook/log chạy thật.

## Môi trường và cài đặt

Mã EDA được repo ghi nhận đã kiểm tra trên Python 3.12.6. Xem `docs/EDA.md` để chạy lại. `requirements.txt` hiện phục vụ EDA, chưa gồm scikit-learn, torch hoặc transformers; cài file đó chưa đủ để chạy A/B/C. Môi trường mô hình cần cài bổ sung và khóa phiên bản sau khi chạy thành công.

Nhóm cần bổ sung môi trường mô hình riêng, ghi phiên bản đã kiểm tra và lựa chọn PyTorch phù hợp thiết bị. Không dùng lệnh placeholder `python=3.x` như một lệnh cài đặt hoàn chỉnh. Ví dụ ở phần B cần môi trường đã cài transformers và PyTorch.

## Tài liệu tham khảo

- [Bài báo GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/)
- [GoEmotions của Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- [OneVsRestClassifier và nhãn multi-hot](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
- [TF-IDF trong scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [BART-large-MNLI model card](https://huggingface.co/facebook/bart-large-mnli)
- [Zero-shot pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline)
- [BERT-base-uncased model card](https://huggingface.co/google-bert/bert-base-uncased)
- [RoBERTa-base model card](https://huggingface.co/FacebookAI/roberta-base)
- [DistilBERT-base-uncased model card](https://huggingface.co/distilbert/distilbert-base-uncased)
- [BCEWithLogitsLoss trong mã nguồn PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py)

Trích dẫn bài báo nền tảng: Demszky, D., Movshovitz-Attias, D., Ko, J.,
Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained
Emotions*. Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, 4040–4054. https://doi.org/10.18653/v1/2020.acl-main.372

Giấy phép cho mã của repo chưa được nhóm công bố; xem giấy phép tại nguồn cho dữ liệu
GoEmotions và các checkpoint sử dụng.
