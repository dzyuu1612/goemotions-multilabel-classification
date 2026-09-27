
# GoEmotions: Phân loại cảm xúc đa nhãn chi tiết

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/Lĩnh_vực-NLP-orange.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Task](<https://img.shields.io/badge/Bài_toán-Multi--Label%20Classification-purple.svg>)](https://en.wikipedia.org/wiki/Multi-label_classification)
[![Dataset](https://img.shields.io/badge/Dataset-GoEmotions-green.svg)](https://github.com/google-research/google-research/tree/master/goemotions)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#giấy-phép)

> Dự án nghiên cứu bài toán phân loại cảm xúc đa nhãn chi tiết trên các bình luận Reddit sử dụng tập dữ liệu GoEmotions, tập trung so sánh ba hướng tiếp cận: mô hình học máy truyền thống, Zero-shot và Fine-tuning các mô hình Transformer.

---

## 📑 Mục lục

* [📌 Tổng quan](#-tổng-quan)
* [🎯 Bối cảnh nghiên cứu](#-bối-cảnh-nghiên-cứu)
* [📊 Tập dữ liệu GoEmotions](#-tập-dữ-liệu-goemotions)
* [🔬 Mục tiêu nghiên cứu](#-mục-tiêu-nghiên-cứu)
* [🧠 Các hướng tiếp cận](#-các-hướng-tiếp-cận)

  * [Baseline](#1-baseline)
  * [Zero-shot](#2-zero-shot)
  * [Fine-tuning](#3-fine-tuning)
* [⚖️ So sánh các hướng tiếp cận](#️-so-sánh-các-hướng-tiếp-cận)
* [🔎 Phân tích lỗi và cảm xúc](#-phân-tích-lỗi-và-cảm-xúc)
* [🚀 Cài đặt](#-cài-đặt)
* [▶️ Bắt đầu nhanh](#️-bắt-đầu-nhanh)
* [📚 Tài liệu](#-tài-liệu)
* [📈 Đánh giá hiệu năng](#-đánh-giá-hiệu-năng)
* [🧩 Framework và mô hình hỗ trợ](#-framework-và-mô-hình-hỗ-trợ)
* [🤝 Đóng góp](#-đóng-góp)
* [📄 Giấy phép](#-giấy-phép)
* [🙏 Ghi nhận và trích dẫn](#-ghi-nhận-và-trích-dẫn)
* [💬 Cộng đồng và hỗ trợ](#-cộng-đồng-và-hỗ-trợ)

---

## 📌 Tổng quan

Dự án nghiên cứu bài toán **phân loại cảm xúc đa nhãn chi tiết (fine-grained multi-label emotion classification)** sử dụng tập dữ liệu [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions).

Các hệ thống phân tích cảm xúc truyền thống thường phân loại văn bản thành một số lượng nhỏ các nhóm như **Tích cực (Positive)**, **Tiêu cực (Negative)** và **Trung tính (Neutral)**, hoặc dựa trên một số cảm xúc cơ bản.

Tuy nhiên, cảm xúc trong giao tiếp thực tế đa dạng và phức tạp hơn nhiều. Đặc biệt trên mạng xã hội, một bình luận có thể thể hiện nhiều trạng thái cảm xúc cùng lúc như ngưỡng mộ, biết ơn, tò mò, bối rối hoặc nhẹ nhõm.

Vì vậy, dự án tập trung vào bài toán **fine-grained emotion classification**, trong đó một văn bản có thể được gán đồng thời nhiều nhãn cảm xúc.

Dự án triển khai và so sánh ba hướng tiếp cận:

**Baseline → Zero-shot → Fine-tune**

Việc so sánh nhằm phân tích sự khác biệt giữa các phương pháp về hiệu quả phân loại, cách tiếp cận, yêu cầu dữ liệu huấn luyện, chi phí tính toán và khả năng thích ứng với bài toán phân loại cảm xúc đa nhãn.

---

## 🎯 Bối cảnh nghiên cứu

### Vấn đề

Hầu hết các hệ thống phân tích cảm xúc trước đây phân loại văn bản ở mức tương đối thô.

Các cách tiếp cận phổ biến bao gồm:

* Positive / Negative
* Positive / Negative / Neutral
* Một số cảm xúc cơ bản như:

  * Vui (Joy)
  * Buồn (Sadness)
  * Sợ hãi (Fear)
  * Tức giận (Anger)
  * Bất ngờ (Surprise)
  * Ghê tởm (Disgust)

Các nhóm cảm xúc này có thể chưa đủ để mô tả đầy đủ sự đa dạng trong cách con người thể hiện cảm xúc bằng ngôn ngữ tự nhiên.

Ví dụ, một bình luận trên Reddit có thể đồng thời thể hiện sự ngưỡng mộ và biết ơn. Nếu chỉ cho phép một nhãn duy nhất, thông tin cảm xúc trong văn bản có thể không được biểu diễn đầy đủ.

### Hướng nghiên cứu

Nghiên cứu GoEmotions giải quyết vấn đề này bằng cách xây dựng một tập dữ liệu lớn với hệ thống nhãn cảm xúc chi tiết, đồng thời nghiên cứu khả năng sử dụng các mô hình Transformer như BERT cho bài toán **multi-label emotion classification**.

Dự án này sử dụng GoEmotions làm tập dữ liệu chính để thực hiện một nghiên cứu so sánh giữa phương pháp học máy truyền thống và các mô hình ngôn ngữ tiền huấn luyện.

---

## 📊 Tập dữ liệu GoEmotions

**GoEmotions** là tập dữ liệu được xây dựng cho bài toán phân loại cảm xúc chi tiết.

Tập dữ liệu bao gồm khoảng:

* **58.000 bình luận tiếng Anh**
* **27 nhãn cảm xúc chi tiết**
* **1 nhãn Neutral**
* Tổng cộng **28 nhóm cảm xúc**

Dữ liệu được thu thập từ các cuộc thảo luận thực tế trên Reddit, giúp phản ánh cách cảm xúc được thể hiện trong giao tiếp trên mạng xã hội.

### Phân loại cảm xúc

Khác với bài toán **single-label classification**, GoEmotions hỗ trợ **multi-label classification**.

Điều này có nghĩa là một bình luận có thể nhận nhiều nhãn cảm xúc nếu trong nội dung xuất hiện nhiều trạng thái cảm xúc khác nhau.

Ví dụ, một bình luận có thể đồng thời thể hiện **admiration** và **gratitude** thay vì bị bắt buộc phải thuộc về một cảm xúc duy nhất.

### Nghiên cứu gốc

Bài báo gốc nghiên cứu việc sử dụng các mô hình dựa trên BERT cho bài toán này, đồng thời phân tích mối quan hệ giữa các nhóm cảm xúc.

Nghiên cứu cũng chỉ ra những cảm xúc thường xuất hiện cùng nhau hoặc những cảm xúc có khả năng gây nhầm lẫn trong quá trình phân loại.

### Tài liệu tham khảo

**GoEmotions: A Dataset of Fine-Grained Emotions**

* ACL Anthology: https://aclanthology.org/2020.acl-main.372/
* GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions

---

## 🔬 Mục tiêu nghiên cứu

Dựa vào bài toán được cho và tập dữ liệu GoEmotions, nhóm có định hướng và triển khai như sau.

### 1. Thực hiện và đánh giá bài toán phân loại cảm xúc đa nhãn

Thực hiện thực nghiệm và đánh giá bài toán **phân loại cảm xúc đa nhãn trên mạng xã hội** dựa trên bộ dữ liệu GoEmotions.

Nhóm tập trung vào khả năng của các mô hình trong việc nhận diện những cảm xúc chi tiết được thể hiện trong các bình luận Reddit.

### 2. So sánh các hướng tiếp cận

So sánh độ hiệu quả giữa các giải pháp và đặt giải pháp truyền thống làm mốc điểm chuẩn.

Ba hướng tiếp cận chính bao gồm:

* **Baseline:** TF-IDF + Logistic Regression
* **Zero-shot:** BART-large-MNLI
* **Fine-tune:** BERT, RoBERTa, DistilBERT

Thông qua ba hướng tiếp cận **Baseline → Zero-shot → Fine-tune**, nhóm hướng đến việc phân tích không chỉ mô hình nào cho kết quả tốt hơn về mặt chỉ số, mà còn làm rõ sự khác biệt về:

* Cách tiếp cận
* Yêu cầu dữ liệu huấn luyện
* Chi phí tính toán
* Khả năng thích ứng với bài toán phân loại cảm xúc đa nhãn

Cuối cùng, nhóm sẽ phân tích các trường hợp mô hình dự đoán đúng, dự đoán sai hoặc nhầm lẫn giữa các cảm xúc có ý nghĩa gần nhau, từ đó đánh giá những khó khăn chính của bài toán và khả năng ứng dụng thực tế của từng phương pháp.

---

## 🧠 Các hướng tiếp cận

### 1. Baseline

**TF-IDF + Logistic Regression**

Baseline sử dụng phương pháp NLP truyền thống gồm:

```text
Văn bản
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Nhãn cảm xúc
```

TF-IDF được sử dụng để biểu diễn văn bản dưới dạng các đặc trưng số, sau đó Logistic Regression được sử dụng làm mô hình phân loại.

Mục đích của Baseline là tạo ra một **mốc tham chiếu truyền thống** để so sánh với các phương pháp dựa trên mô hình ngôn ngữ tiền huấn luyện.

---

### 2. Zero-shot

**BART-large-MNLI và các mô hình ngôn ngữ tiền huấn luyện**

Phương pháp Zero-shot sử dụng một mô hình ngôn ngữ đã được huấn luyện sẵn để dự đoán trực tiếp các nhãn cảm xúc từ văn bản mà **không cần học thêm hoặc cập nhật bất kỳ trọng số nào trên tập dữ liệu GoEmotions**.

Quy trình khái quát:

```text
Văn bản
   ↓
Mô hình ngôn ngữ tiền huấn luyện
   ↓
Các nhãn cảm xúc ứng viên
   ↓
Nhãn cảm xúc được dự đoán
```

Các nhãn ứng viên được lấy từ hệ thống nhãn cảm xúc của GoEmotions.

Phương pháp này cho phép đánh giá khả năng của một mô hình đã được huấn luyện sẵn trong việc nhận diện cảm xúc chi tiết mà không cần Fine-tuning trên tập dữ liệu mục tiêu.

---

### 3. Fine-tuning

**BERT / RoBERTa / DistilBERT**

Phương pháp Fine-tuning sử dụng các kiến trúc Transformer đã được tiền huấn luyện và tiếp tục huấn luyện trên tập dữ liệu GoEmotions để thích ứng với bài toán phân loại cảm xúc đa nhãn.

Các mô hình dự kiến sử dụng:

* BERT
* RoBERTa
* DistilBERT

Quy trình khái quát:

```text
Mô hình Transformer tiền huấn luyện
              ↓
      Fine-tuning trên
         GoEmotions
              ↓
   Phân loại cảm xúc đa nhãn
```

Khác với Zero-shot, trong phương pháp Fine-tuning, trọng số của mô hình được cập nhật dựa trên tập dữ liệu GoEmotions.

Mục tiêu là giúp mô hình thích ứng với các nhóm cảm xúc chi tiết được sử dụng trong tập dữ liệu.

---

## ⚖️ So sánh các hướng tiếp cận

Dự án so sánh ba phương pháp từ nhiều khía cạnh khác nhau.

| Phương pháp | Mô hình / Kỹ thuật       | Huấn luyện trên GoEmotions | Cập nhật trọng số | Mục đích                                                      |
| -------------- | ---------------------------- | ----------------------------: | --------------------: | ---------------------------------------------------------------- |
| Baseline       | TF-IDF + Logistic Regression |                           Có |                   Có | Tạo mốc tham chiếu truyền thống                             |
| Zero-shot      | BART-large-MNLI              |                        Không |                Không | Dự đoán trực tiếp bằng kiến thức đã tiền huấn luyện |
| Fine-tune      | BERT                         |                           Có |                   Có | Thích ứng Transformer với GoEmotions                          |
| Fine-tune      | RoBERTa                      |                           Có |                   Có | Thích ứng Transformer với GoEmotions                          |
| Fine-tune      | DistilBERT                   |                           Có |                   Có | Thích ứng Transformer với GoEmotions                          |

Việc so sánh tập trung vào khả năng của từng phương pháp trong việc xử lý bài toán **phân loại cảm xúc đa nhãn chi tiết**.

---

## 🔎 Phân tích lỗi và cảm xúc

Bên cạnh việc so sánh kết quả của các mô hình, nhóm sẽ phân tích các trường hợp dự đoán cụ thể.

Các nội dung phân tích bao gồm:

* Các trường hợp dự đoán đúng
* Các trường hợp dự đoán sai
* Các cảm xúc thường xuyên bị nhầm lẫn
* Các cảm xúc có ý nghĩa gần nhau
* Các trường hợp một bình luận chứa nhiều cảm xúc cùng lúc

Nghiên cứu GoEmotions cũng phân tích mối quan hệ giữa các nhóm cảm xúc, trong đó có những cảm xúc thường xuất hiện cùng nhau hoặc khó phân biệt.

Một số ví dụ:

* **Annoyance ↔ Anger**
* **Admiration ↔ Gratitude**

Việc phân tích những trường hợp này giúp làm rõ các khó khăn trong bài toán phân loại cảm xúc chi tiết.

---

## 🚀 Cài đặt

### Yêu cầu

Dự án được xây dựng trên môi trường Python phục vụ các bài toán NLP.

Môi trường đề xuất:

```text
Python 3.x
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Sử dụng Conda

Tạo môi trường riêng cho dự án:

```bash
conda create -n goemotions python=3.x
conda activate goemotions
```

Sau đó cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

### Cài đặt từ mã nguồn

Clone repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

> `<REPOSITORY_URL>` và `<REPOSITORY_DIRECTORY>` sẽ được thay thế bằng thông tin repository thực tế của nhóm.

---

## ▶️ Bắt đầu nhanh

Ví dụ tối thiểu dưới đây minh họa cách tiếp cận **Zero-shot** sử dụng BART-large-MNLI:

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "I am really grateful for everything you have done."

labels = [
    "gratitude",
    "admiration",
    "joy",
    "sadness",
    "anger",
    "neutral"
]

result = classifier(
    text,
    candidate_labels=labels
)

print(result)
```

Ví dụ trên minh họa ý tưởng của phương pháp Zero-shot: mô hình nhận một văn bản cùng danh sách các nhãn cảm xúc ứng viên và đưa ra dự đoán mà không cần Fine-tuning trên GoEmotions.

---

## 📚 Tài liệu

Tài liệu của dự án tập trung vào ba hướng tiếp cận chính.

### Baseline

Nội dung bao gồm:

* TF-IDF
* Logistic Regression
* Phân loại cảm xúc đa nhãn

### Zero-shot

Nội dung bao gồm:

* Mô hình ngôn ngữ tiền huấn luyện
* BART-large-MNLI
* Các nhãn cảm xúc ứng viên
* Zero-shot prediction

### Fine-tuning

Nội dung bao gồm:

* BERT
* RoBERTa
* DistilBERT
* Fine-tuning trên GoEmotions
* Phân loại cảm xúc đa nhãn

---

## 📈 Đánh giá hiệu năng

Nhóm sẽ tiến hành so sánh kết quả thực nghiệm giữa ba hướng tiếp cận chính.

| Phương pháp | Mô hình                    | Huấn luyện trên tập mục tiêu | Kết quả |
| -------------- | ---------------------------- | ---------------------------------: | --------: |
| Baseline       | TF-IDF + Logistic Regression |                                Có |       TBD |
| Zero-shot      | BART-large-MNLI              |                             Không |       TBD |
| Fine-tune      | BERT                         |                                Có |       TBD |
| Fine-tune      | RoBERTa                      |                                Có |       TBD |
| Fine-tune      | DistilBERT                   |                                Có |       TBD |

> Các giá trị benchmark sẽ được cập nhật sau khi nhóm hoàn thành thực nghiệm.

Phần benchmark được sử dụng để so sánh trực tiếp giữa phương pháp học máy truyền thống, Zero-shot và các mô hình Transformer được Fine-tune.

---

## 🧩 Framework và mô hình hỗ trợ

### Machine Learning / NLP

* TF-IDF
* Logistic Regression

### Transformer Models

* BERT
* RoBERTa
* DistilBERT
* BART-large-MNLI

### Dataset

* GoEmotions
* Bình luận tiếng Anh trên Reddit
* 27 nhãn cảm xúc chi tiết
* 1 nhãn Neutral

---

## 🤝 Đóng góp

Các đóng góp cho dự án được hoan nghênh.

Nếu muốn đóng góp:

1. Fork repository.
2. Tạo một branch mới.

```bash
git checkout -b feature/your-feature
```

3. Thực hiện thay đổi.
4. Kiểm tra thay đổi.
5. Commit:

```bash
git commit -m "Add your feature"
```

6. Push branch:

```bash
git push origin feature/your-feature
```

7. Tạo Pull Request.

Đối với báo cáo lỗi, nên cung cấp:

* Mô tả rõ ràng về lỗi
* Các bước để tái hiện lỗi
* Thông báo lỗi liên quan
* Thông tin môi trường
* Hành vi mong đợi và hành vi thực tế

---

## 📄 Giấy phép

Giấy phép của repository dự án sẽ được nhóm xác định và cập nhật sau.

> **License:** TBD

Đối với tập dữ liệu GoEmotions và bài báo gốc, người sử dụng cần tham khảo các điều khoản và giấy phép tương ứng được cung cấp bởi nguồn gốc của dataset và publication.

---

## 🙏 Ghi nhận và trích dẫn

Dự án được xây dựng dựa trên **GoEmotions dataset** và nghiên cứu tương ứng.

### GoEmotions

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

**Bài báo:**

https://aclanthology.org/2020.acl-main.372/

### BibTeX

```bibtex
@inproceedings{demszky-etal-2020-goemotions,
    title = "{G}o{E}motions: A Dataset of Fine-Grained Emotions",
    author = "Demszky, Dorottya
              and Movshovitz-Attias, Daniel
              and Ko, Jeongwoo
              and Cowen, Alan
              and Nemade, Gaurav
              and Ravi, Sujith",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

---

## 💬 Cộng đồng và hỗ trợ

Đối với các vấn đề liên quan đến dataset và nghiên cứu GoEmotions:

* **ACL Anthology:** https://aclanthology.org/2020.acl-main.372/
* **GoEmotions Repository:** https://github.com/google-research/google-research/tree/master/goemotions

Đối với các vấn đề liên quan trực tiếp đến project, sử dụng:

* GitHub Issues
* GitHub Discussions
* Tài liệu của project

---

## 📖 Tài liệu tham khảo

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL 2020.
2. GoEmotions Dataset — Google Research.
3. BERT — Bidirectional Encoder Representations from Transformers.
4. BART — Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.

---

## ⭐ Định hướng dự án

Dự án được triển khai theo cấu trúc so sánh:

```text
                         GoEmotions
                              │
                              ▼
                  Phân loại cảm xúc đa nhãn
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          Baseline         Zero-shot       Fine-tune
             │                │                │
       TF-IDF +          BART-large-       BERT /
       Logistic          MNLI              RoBERTa /
       Regression                         DistilBERT
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    So sánh các phương pháp
                              │
                              ▼
                  Phân tích lỗi và cảm xúc
```
