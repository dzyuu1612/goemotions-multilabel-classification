# BÁO CÁO TIẾN ĐỘ LẦN 2

## Phân loại cảm xúc đa nhãn với GoEmotions

**Ngày cập nhật:** 08/10/2026 13:20 (Asia/Bangkok). Đây là bản chuẩn bị báo cáo hiện trạng, không xác nhận đã nộp giảng viên.

**Giảng viên / mã lớp / năm học, học kỳ:** ____________________

**MSSV bốn thành viên:** ____________________

## 1. Mục tiêu và phạm vi kiểm tra

Đề tài tiếp tục bài GoEmotions ACL 2020 [1], giữ 28 nhãn, split/mapping/metric dùng chung. Lần tiến độ 2 cần kết quả đủ ba kiến trúc × ba seed, mean±std so với A/B, ít nhất ba nhóm lỗi đối chiếu C, minh chứng demo và nâng cao. Các bảng được cập nhật bằng artifacts hiện có; nếu còn thiếu, báo trạng thái thay vì tạo số.

## 2. Kết quả A/B/C

| Hệ thống @0,5 | Split | Run | Macro-F1 | Micro-F1 | Hamming |
| --- | --- | --- | --- | --- | --- |
| A_standard | validation | 1 | 0.2025 | 0.3760 | 0.0354 |
| A_balanced | validation | 1 | 0.4562 | 0.5099 | 0.0532 |

Macro/Micro-F1 và Hamming được đọc cùng Precision/Recall và support; Macro-F1 không phải tỷ lệ câu có cả tập nhãn đúng [2]. Summary có ghi split và luật ngưỡng; tuned-val dùng dữ liệu chọn ngưỡng không phải phép đánh giá độc lập.

### 2.1. Zero-shot B

Model BART-MNLI dùng trực tiếp, không cập nhật trọng số GoEmotions [3].

| Phạm vi B | N | Ngưỡng | Macro-F1 | Micro-F1 | Hamming |
| --- | --- | --- | --- | --- | --- |
| SMOKE/PILOT; không thay benchmark full | 8 | 0.5 | 0.0721 | 0.0781 | 0.5268 |

Nguồn: `data/processed/zero_shot/smoke/run_metadata.json` và `validation_metrics.json`. Template: `This text expresses {}.`. B không cập nhật trọng số trên GoEmotions; kết quả threshold hiệu chỉnh nếu có phải tách khỏi hàng gốc. Chưa có B full; điểm pilot này không được dùng kết luận B tốt/kém hơn A hoặc C.

### 2.2. Từng seed C

| Kiến trúc | Seed | Trạng thái | Macro-F1 val | Micro-F1 val | Hamming val |
| --- | --- | --- | --- | --- | --- |
| C1 BERT-base-cased | 42 | Full validation hoàn thành | 0.4641 | 0.5760 | 0.0328 |
| C1 BERT-base-cased | 123 | Có metadata; chưa hoàn thành full | — | — | — |
| C1 BERT-base-cased | 2026 | Chưa có artifacts full | — | — | — |
| C2 RoBERTa-base | 42 | Chưa có artifacts full | — | — | — |
| C2 RoBERTa-base | 123 | Chưa có artifacts full | — | — | — |
| C2 RoBERTa-base | 2026 | Chưa có artifacts full | — | — | — |
| C3 DistilBERT-base-uncased | 42 | Chưa có artifacts full | — | — | — |
| C3 DistilBERT-base-uncased | 123 | Chưa có artifacts full | — | — | — |
| C3 DistilBERT-base-uncased | 2026 | Chưa có artifacts full | — | — | — |

Hiện có 1/9 seed full theo metadata/scores. Bảng này chỉ tính full validation hợp lệ với 5.426 mẫu và 28 nhãn. Mỗi kiến trúc có ít nhất ba seed mới đủ mức nghiệm thu môn học.

### 2.3. Mean ± sample std

| Kiến trúc | Full seeds | Macro-F1 mean±std val | Micro-F1 mean±std val | Hamming mean±std val |
| --- | --- | --- | --- | --- |
| C1 BERT-base-cased | 1 | Chưa đủ 3 seed | Chưa đủ 3 seed | Chưa xếp hạng |
| C2 RoBERTa-base | 0 | Chưa đủ 3 seed | Chưa đủ 3 seed | Chưa xếp hạng |
| C3 DistilBERT-base-uncased | 0 | Chưa đủ 3 seed | Chưa đủ 3 seed | Chưa xếp hạng |

Sample std dùng mẫu số K−1. Một seed không cho sample std; chưa đủ ba seed thì chưa xếp hạng kiến trúc. Bảng từng seed và bảng tổng hợp phải được giữ cùng nhau; điểm checkpoint demo khác trung bình kiến trúc.

Ngưỡng riêng được chọn bằng validation của đúng model; không dùng test để tìm ngưỡng [4].

## 3. Phân tích lỗi có hệ thống

Chưa có hồ sơ lỗi đối chiếu ba C trên test; phần này còn cần hoàn thành theo yêu cầu báo cáo tiến độ 2. Các ví dụ A dưới đây chỉ là bằng chứng đã có, không thay yêu cầu ba C.

Ba nhóm cần đọc thủ công: cảm xúc gần nghĩa; bỏ sót một phần nhãn của câu đa nhãn; phủ định/hàm ý/slang hoặc thiếu ngữ cảnh. Mỗi ví dụ đối chiếu phải cùng ID và true labels, kèm predictions/scores/threshold của C1/C2/C3. Một câu có thể đóng góp nhiều cặp FN/FP; không cộng mọi cặp để tính số câu sai.

Ví dụ A standard @0,5 đã kiểm: `eczwil0` bỏ sót pride; `ed832y6` dự đoán love thừa; `eczdvun` tìm gratitude nhưng thiếu admiration. Đây là ví dụ A thực tế, chưa phải lỗi của các C. Chỉ suy ra nguyên nhân sau khi đọc ngữ cảnh/annotation.

## 4. Demo D

Chưa có selection manifest best C. Không dùng A/B hoặc checkpoint smoke để thay demo môn học.

Chưa có hồ sơ kiểm demo tại thời điểm render. Khi hoàn tất cần URL chạy tại buổi demo hoặc ảnh/video, checkpoint/ngưỡng và so score với script suy luận.

Gradio nối hàm suy luận với giao diện nhập văn bản [5]. Demo cần đúng checkpoint/tokenizer/threshold đã chọn, hỗ trợ nhiều nhãn, xử lý input rỗng và không ép neutral khi mọi score dưới ngưỡng. URL đang chạy/ảnh/video được điền sau khi xác minh, không tạo link giả.

**Link demo / minh chứng buổi trình bày:** ____________________

## 5. Nâng cao và nhãn hiếm

Đã có A standard/balanced và ngưỡng chung/riêng trên validation:

| Cấu hình A | Val N | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Hamming |
| --- | --- | --- | --- | --- | --- | --- |
| standard_fixed | 5426 | 0.2025 | 0.3760 | 0.7254 | 0.2538 | 0.0354 |
| standard_global | 5426 | 0.4094 | 0.5100 | 0.4102 | 0.6741 | 0.0544 |
| standard_tuned | 5426 | 0.4391 | 0.5427 | 0.4876 | 0.6119 | 0.0433 |
| balanced_fixed | 5426 | 0.4562 | 0.5099 | 0.4158 | 0.6592 | 0.0532 |
| balanced_global | 5426 | 0.4660 | 0.5176 | 0.4524 | 0.6047 | 0.0473 |
| balanced_tuned | 5426 | 0.4901 | 0.5467 | 0.4796 | 0.6356 | 0.0443 |

Ngưỡng phải được chọn trên validation của đúng model, tránh dùng test để điều chỉnh [4]. Năm nhãn hiếm chọn bằng train: grief, pride, relief, nervousness, embarrassment. Có bảng F1 trước/sau A tại `reports/BASELINE_RESULTS.md` và bảng per-label. Cải tiến C nếu có phải báo cùng seed/config và mean±std đủ run; pilot một seed được ghi đúng pilot.

Đầu việc nâng cao tiếp theo: khóa threshold từng run, tổng hợp rare-label P/R/F1/support, giữ cả nhãn không cải thiện và đánh giá cuối trên test. Nếu weighted C được bổ sung, phải train lại; threshold tuning riêng không làm encoder học thêm.

## 6. Vai trò, đóng góp và phần chưa hoàn thành

| Thành viên | Vai trò | Bằng chứng hiện có | % công sức |
| --- | --- | --- | --- |
| Bảo Duy Nguyễn | A; điều phối B; data/metrics và bảng nâng cao | A full validation; B chưa đủ full | ____________________ |
| Quốc Khánh | C1 BERT; phần đầu/tổng hợp báo cáo | C1 1/3 seed full; xem bảng run | ____________________ |
| Đức Trí | C2 RoBERTa; hỗ trợ/bàn giao B | C2 0/3 seed full; xem bảng run | ____________________ |
| Nhật Huy | C3 DistilBERT; tích hợp demo best C | C3 0/3 seed full; demo theo hồ sơ | ____________________ |

Đức Trí là “Thợ Săn Thập Cẩm”; nhóm có bốn người. Tỷ lệ công sức để trống cho nhóm thống nhất bằng artifacts, không tự đánh đồng phân công và việc đã làm.

Theo summary hiện đọc: `complete=false`. Những phần cần bổ sung:

- A test
- B full validation + frozen protocol
- C bert seed 42 frozen thresholds
- C bert seed 42 test
- C bert seed 123 full
- C bert seed 2026 full
- C roberta seed 42 full
- C roberta seed 123 full
- C roberta seed 2026 full
- C distilbert seed 42 full
- C distilbert seed 123 full
- C distilbert seed 2026 full
- D selected_model from 3 architectures × 3 seeds

Chưa đủ C/mean±std/lỗi/demo thì bản hiện trạng chưa đáp ứng toàn bộ tiêu chí tiến độ 2; không xác nhận đạt điểm hoặc đã nộp. Nhóm cần kiểm và cập nhật sau khi artifacts hoàn tất.

## 7. Hồ sơ và bước hoàn tất

Summary: `reports/project_results/summary.json`; SHA-256: `29dfd39ae78ef014097a62c93598b9c15d2131b0f4bd7c912422bdfa59397a4b`. Cấu hình, môi trường và thời gian thực tế nằm trong `run_metadata.json`. Thời gian có gián đoạn do ngủ máy không được dùng như benchmark tốc độ được kiểm soát.

Kho chung: https://github.com/trangkhanh-ai/goemotions-multilabel-classification

Thứ tự tiếp tục: đủ seed → tổng hợp validation/lỗi/nâng cao → chọn C/demo → khóa protocol → test → hoàn thiện báo cáo cuối/slide/đóng góp. Script tạo báo cáo chỉ đọc artifacts, không chạy GPU, không thay summary và không tự gửi/nộp.

## Tài liệu tham khảo

[1] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, “GoEmotions: A Dataset of Fine-Grained Emotions,” in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics, 2020, pp. 4040–4054, doi: 10.18653/v1/2020.acl-main.372. https://aclanthology.org/2020.acl-main.372/

[2] scikit-learn developers, “precision_recall_fscore_support,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

[3] Facebook AI, “bart-large-mnli model card,” Hugging Face. Accessed: Oct. 8, 2026. [Online]. Available: https://huggingface.co/facebook/bart-large-mnli

[4] scikit-learn developers, “Tuning the decision threshold for class prediction,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/classification_threshold.html

[5] Gradio, “Quickstart,” Gradio documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://www.gradio.app/guides/quickstart
