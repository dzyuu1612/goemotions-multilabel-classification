# BÁO CÁO TIẾN ĐỘ LẦN 1

## Phân loại cảm xúc đa nhãn với GoEmotions

**Ngày cập nhật:** 09/10/2026 06:58 (Asia/Bangkok). Đây là bản chuẩn bị báo cáo theo hiện trạng, không xác nhận đã nộp giảng viên.

**Giảng viên / mã lớp / năm học, học kỳ:** ____________________

**MSSV bốn thành viên:** ____________________

## 1. Bài nền tảng và mục tiêu

Demszky và cs. công bố GoEmotions tại ACL 2020: bình luận Reddit tiếng Anh, 27 cảm xúc cùng neutral, có thể nhiều nhãn trong một câu; baseline chính của bài dùng BERT [1]. Đề tài giữ đủ taxonomy và mở rộng so sánh A cổ điển, B zero-shot, ba C fine-tune theo yêu cầu môn học. TF-IDF + Logistic Regression là phần cổ điển cô giao, không phải baseline BERT trong paper.

Mục tiêu là khảo sát dữ liệu, tạo scores đa nhãn, đánh giá công bằng và xây demo từ C được chọn bằng validation. Nâng cao bằng weighting/ngưỡng riêng phải có số trước/sau và nhãn hiếm. Dữ liệu tiếng Anh Reddit không tự chứng minh chất lượng cho tiếng Việt hoặc doanh nghiệp.

## 2. Dữ liệu đã khám phá

Official split gồm 43.410 train, 5.426 validation, 5.427 test; tổng 54.263 ở bản filtered/simplified. Bản thô có 58.009 bình luận [2]. Revision nhóm: `add492243ff905527e67aeb8b80c082af02207c3`.

EDA từ `reports/summary.json`: 63812 lượt nhãn, 8817 câu đa nhãn; tỷ lệ đa nhãn khoảng 16.25%. Tỷ số support nhãn phổ biến/hiếm khoảng 184.66. Năm nhãn hiếm được chọn bằng train: grief 77, pride 111, relief 153, nervousness 164, embarrassment 303.

Ví dụ dữ liệu validation: ID `eczdvun`, “Thank you. I really appreciate your response”, nhãn admiration và gratitude; đây là annotation thật được dùng trong hồ sơ lỗi A. Label mapping/multi-hot có 28 cột; neutral được giữ theo nguồn. ID giữa split không trùng nhưng có văn bản trùng; benchmark chính giữ official split và ghi giới hạn này.

## 3. Kết quả baseline A

A dùng unigram/bigram TF-IDF fit train và 28 bộ LR OvR [3]. Module metric giữ đủ nhãn, `zero_division=0`; báo F1 micro/macro cùng P/R/Hamming [4]. Bảng sau là full validation, không phải test.

| Cấu hình A | Val N | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Hamming |
| --- | --- | --- | --- | --- | --- | --- |
| standard_fixed | 5426 | 0.2025 | 0.3760 | 0.7254 | 0.2538 | 0.0354 |
| standard_global | 5426 | 0.4094 | 0.5100 | 0.4102 | 0.6741 | 0.0544 |
| standard_tuned | 5426 | 0.4391 | 0.5427 | 0.4876 | 0.6119 | 0.0433 |
| balanced_fixed | 5426 | 0.4562 | 0.5099 | 0.4158 | 0.6592 | 0.0532 |
| balanced_global | 5426 | 0.4660 | 0.5176 | 0.4524 | 0.6047 | 0.0473 |
| balanced_tuned | 5426 | 0.4901 | 0.5467 | 0.4796 | 0.6356 | 0.0443 |

Weighting thường tăng recall cùng nhãn thừa; tuned-val được đo trên chính validation chọn ngưỡng nên có thể lạc quan. Không dùng Macro-F1 0,4901 tuned-val để tuyên bố vượt F1 test của paper. Tệp nguồn: `reports/baseline_validation/summary.json` và artifacts model/scores/thresholds của từng biến thể.

## 4. Zero-shot B: kết quả và trạng thái thực tế

Checkpoint BART-large-MNLI được dùng qua pipeline, `multi_label=True`, đủ 28 candidate labels; model đã học MNLI trước đó nhưng nhóm không fine-tune trên GoEmotions [5].

| Phạm vi B | N | Ngưỡng | Macro-F1 | Micro-F1 | Hamming |
| --- | --- | --- | --- | --- | --- |
| FULL validation | 5426 | 0.5 | 0.1060 | 0.1027 | 0.5307 |

Nguồn: `data/processed/zero_shot/full/run_metadata.json` và `validation_metrics.json`. Template: `This text expresses {}.`. B không cập nhật trọng số trên GoEmotions; kết quả threshold hiệu chỉnh nếu có phải tách khỏi hàng gốc. Đã có số full; kiểm frozen protocol trước test.

## 5. Tiến độ ba kiến trúc và môi trường

Tại thời điểm cập nhật có 9/9 seed full được kiểm bằng metadata và validation metrics. Seed đã có metadata nhưng chưa hoàn thành không được tính là run full; smoke/pilot được tách khỏi bảng.

| Kiến trúc | Seed | Trạng thái | Macro-F1 val | Micro-F1 val | Hamming val |
| --- | --- | --- | --- | --- | --- |
| C1 BERT-base-cased | 42 | Full validation hoàn thành | 0.4641 | 0.5760 | 0.0328 |
| C1 BERT-base-cased | 123 | Full validation hoàn thành | 0.4773 | 0.5839 | 0.0318 |
| C1 BERT-base-cased | 2026 | Full validation hoàn thành | 0.4723 | 0.5822 | 0.0321 |
| C2 RoBERTa-base | 42 | Full validation hoàn thành | 0.4286 | 0.5780 | 0.0300 |
| C2 RoBERTa-base | 123 | Full validation hoàn thành | 0.4111 | 0.5730 | 0.0300 |
| C2 RoBERTa-base | 2026 | Full validation hoàn thành | 0.4306 | 0.5792 | 0.0300 |
| C3 DistilBERT-base-uncased | 42 | Full validation hoàn thành | 0.4018 | 0.5668 | 0.0304 |
| C3 DistilBERT-base-uncased | 123 | Full validation hoàn thành | 0.4132 | 0.5739 | 0.0300 |
| C3 DistilBERT-base-uncased | 2026 | Full validation hoàn thành | 0.4042 | 0.5728 | 0.0300 |

| Kiến trúc | Phụ trách | Môi trường thực tế | Config từ metadata |
| --- | --- | --- | --- |
| C1 BERT-base-cased | Quốc Khánh | NVIDIA GeForce RTX 5060 Laptop GPU; Python 3.13.9; torch 2.13.0+cu130; Transformers 4.57.6 | lr=5e-05; epochs=4; batch=16; accum=1; max_length=128; dynamic_batch_trim |
| C2 RoBERTa-base | Đức Trí | NVIDIA GeForce RTX 5060 Laptop GPU; Python 3.13.9; torch 2.13.0+cu130; Transformers 4.57.6 | lr=2e-05; epochs=3; batch=16; accum=1; max_length=128; dynamic_batch_trim |
| C3 DistilBERT-base-uncased | Nhật Huy | NVIDIA GeForce RTX 5060 Laptop GPU; Python 3.13.9; torch 2.13.0+cu130; Transformers 4.57.6 | lr=2e-05; epochs=3; batch=16; accum=1; max_length=128; dynamic_batch_trim |

BERT cased theo lựa chọn kho GoEmotions; RoBERTa và DistilBERT có tokenizer/head riêng. Mỗi kiến trúc cần ba seed cùng cấu hình. Mốc mean±std chỉ tổng hợp khi đủ run; không tự báo std=0 với một seed. Tài liệu kiến trúc giải thích nền tảng C1/C2/C3 [6], [7], [8].

## 6. Phân công và mức hoàn thành theo artifacts

| Thành viên | Vai trò | Bằng chứng hiện có | % công sức |
| --- | --- | --- | --- |
| Bảo Duy Nguyễn | A; điều phối B; data/metrics và bảng nâng cao | A full validation; B full đã có | ____________________ |
| Quốc Khánh | C1 BERT; phần đầu/tổng hợp báo cáo | C1 3/3 seed full; xem bảng run | ____________________ |
| Đức Trí | C2 RoBERTa; hỗ trợ/bàn giao B | C2 3/3 seed full; xem bảng run | ____________________ |
| Nhật Huy | C3 DistilBERT; tích hợp demo best C | C3 3/3 seed full; demo theo hồ sơ | ____________________ |

Đức Trí là “Thợ Săn Thập Cẩm”. B làm chung dưới điều phối của Duy; Trí vẫn phụ trách trọn C2. Huy tích hợp C thắng, không mặc định DistilBERT. Các tỷ lệ công sức để nhóm xác nhận, không tự chia đều.

## 7. Vướng mắc và đầu việc tiếp theo

- Đã đủ B full và 9/9 run C; duy trì logs/checkpoints/scores/config khi bàn giao.
- Đã khóa lựa chọn/ngưỡng trên validation và có test; không chọn lại bằng test.
- Đã có ba case đối chiếu và hồ sơ demo; nhóm cần tự đọc, diễn giải và kiểm khi chuyển máy.
- Thiết bị từng bị ngủ trong quá trình C; thời gian elapsed có gián đoạn, không coi là benchmark tốc độ được kiểm soát.
- Cần đọc và giải thích code, xác nhận metadata hành chính và đóng góp trước khi nộp theo kênh cô chỉ định.

## 8. Hồ sơ kiểm chứng

Kho chung: https://github.com/trangkhanh-ai/goemotions-multilabel-classification

Summary tổng: `reports/project_results/summary.json`; SHA-256 tại thời điểm đọc: `2c73aa31fad9df0a8b0b46bedbb3c69c1316fa4e225f32529811e80ccd2e09b6`. Script này chỉ đọc artifacts, không tự chạy GPU, không xác nhận đã nộp cô. Chạy lại sau khi cập nhật summary/runs để làm mới bản tiến độ.

## Tài liệu tham khảo

[1] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, “GoEmotions: A Dataset of Fine-Grained Emotions,” in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics, 2020, pp. 4040–4054, doi: 10.18653/v1/2020.acl-main.372. https://aclanthology.org/2020.acl-main.372/

[2] Google Research, “GoEmotions: README and official data,” Google Research GitHub repository. Accessed: Oct. 8, 2026. [Online]. Available: https://github.com/google-research/google-research/blob/master/goemotions/README.md

[3] scikit-learn developers, “TfidfVectorizer,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[4] scikit-learn developers, “precision_recall_fscore_support,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

[5] Facebook AI, “bart-large-mnli model card,” Hugging Face. Accessed: Oct. 8, 2026. [Online]. Available: https://huggingface.co/facebook/bart-large-mnli

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proc. NAACL-HLT, vol. 1, 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423. https://aclanthology.org/N19-1423/

[7] Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” 2019, arXiv:1907.11692, doi: 10.48550/arXiv.1907.11692. https://arxiv.org/abs/1907.11692

[8] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” 2019, arXiv:1910.01108, doi: 10.48550/arXiv.1910.01108. https://arxiv.org/abs/1910.01108
