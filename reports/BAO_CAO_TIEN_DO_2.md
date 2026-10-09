# BÁO CÁO TIẾN ĐỘ LẦN 2

## Phân loại cảm xúc đa nhãn với GoEmotions

**Ngày cập nhật:** 09/10/2026 06:58 (Asia/Bangkok). Đây là bản chuẩn bị báo cáo hiện trạng, không xác nhận đã nộp giảng viên.

**Giảng viên / mã lớp / năm học, học kỳ:** ____________________

**MSSV bốn thành viên:** ____________________

## 1. Mục tiêu và phạm vi kiểm tra

Đề tài tiếp tục bài GoEmotions ACL 2020 [1], giữ 28 nhãn, split/mapping/metric dùng chung. Lần tiến độ 2 cần kết quả đủ ba kiến trúc × ba seed, mean±std so với A/B, ít nhất ba nhóm lỗi đối chiếu C, minh chứng demo và nâng cao. Các bảng được cập nhật bằng artifacts hiện có; nếu còn thiếu, báo trạng thái thay vì tạo số.

## 2. Kết quả A/B/C

| Hệ thống @0,5 | Split | Run | Macro-F1 | Micro-F1 | Hamming |
| --- | --- | --- | --- | --- | --- |
| A_standard | validation | 1 | 0.2025 | 0.3760 | 0.0354 |
| A_balanced | validation | 1 | 0.4562 | 0.5099 | 0.0532 |
| A_standard | test | 1 | 0.1963 | 0.3800 | 0.0348 |
| A_balanced | test | 1 | 0.4441 | 0.5024 | 0.0547 |
| B_bart_mnli | validation | 1 | 0.1060 | 0.1027 | 0.5307 |
| B_bart_mnli | test | 1 | 0.1035 | 0.1008 | 0.5315 |
| C_bert | validation | 3 | 0.4713 ± 0.0066 | 0.5807 ± 0.0042 | 0.0322 ± 0.0005 |
| C_bert | test | 3 | 0.4720 ± 0.0045 | 0.5820 ± 0.0040 | 0.0318 ± 0.0003 |
| C_roberta | validation | 3 | 0.4234 ± 0.0108 | 0.5767 ± 0.0033 | 0.0300 ± 0.0000 |
| C_roberta | test | 3 | 0.4219 ± 0.0088 | 0.5803 ± 0.0009 | 0.0295 ± 0.0001 |
| C_distilbert | validation | 3 | 0.4064 ± 0.0060 | 0.5712 ± 0.0038 | 0.0302 ± 0.0003 |
| C_distilbert | test | 3 | 0.4116 ± 0.0035 | 0.5731 ± 0.0017 | 0.0297 ± 0.0001 |

Macro/Micro-F1 và Hamming được đọc cùng Precision/Recall và support; Macro-F1 không phải tỷ lệ câu có cả tập nhãn đúng [2]. Summary có ghi split và luật ngưỡng; tuned-val dùng dữ liệu chọn ngưỡng không phải phép đánh giá độc lập.

### 2.1. Zero-shot B

Model BART-MNLI dùng trực tiếp, không cập nhật trọng số GoEmotions [3].

| Phạm vi B | N | Ngưỡng | Macro-F1 | Micro-F1 | Hamming |
| --- | --- | --- | --- | --- | --- |
| FULL validation | 5426 | 0.5 | 0.1060 | 0.1027 | 0.5307 |

Nguồn: `data/processed/zero_shot/full/run_metadata.json` và `validation_metrics.json`. Template: `This text expresses {}.`. B không cập nhật trọng số trên GoEmotions; kết quả threshold hiệu chỉnh nếu có phải tách khỏi hàng gốc. Đã có số full; kiểm frozen protocol trước test.

### 2.2. Từng seed C

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

Hiện có 9/9 seed full theo metadata/scores. Bảng này chỉ tính full validation hợp lệ với 5.426 mẫu và 28 nhãn. Mỗi kiến trúc có ít nhất ba seed mới đủ mức nghiệm thu môn học.

### 2.3. Mean ± sample std

| Kiến trúc | Full seeds | Macro-F1 mean±std val | Micro-F1 mean±std val | Hamming mean±std val |
| --- | --- | --- | --- | --- |
| C1 BERT-base-cased | 3 | 0.4713 ± 0.0066 | 0.5807 ± 0.0042 | 0.0322 ± 0.0005 |
| C2 RoBERTa-base | 3 | 0.4234 ± 0.0108 | 0.5767 ± 0.0033 | 0.0300 ± 0.0000 |
| C3 DistilBERT-base-uncased | 3 | 0.4064 ± 0.0060 | 0.5712 ± 0.0038 | 0.0302 ± 0.0003 |

Sample std dùng mẫu số K−1. Một seed không cho sample std; chưa đủ ba seed thì chưa xếp hạng kiến trúc. Bảng từng seed và bảng tổng hợp phải được giữ cùng nhau; điểm checkpoint demo khác trung bình kiến trúc.

Ngưỡng riêng được chọn bằng validation của đúng model; không dùng test để tìm ngưỡng [4].

## 3. Phân tích lỗi có hệ thống


Đếm trên cùng ID. Seed đại diện chọn bằng validation @0,5; đây là phân tích checkpoint đại diện, không phải trung bình lỗi qua ba seed.

| Kiến trúc | Seed | Nhóm lỗi | Số câu | Mẫu phù hợp định nghĩa | Tỷ lệ trong mẫu phù hợp |
|---|---:|---|---:|---:|---:|
| bert | 123 | partial_multi_label | 482 | 837 | 57.59% |
| bert | 123 | rare_false_negative | 72 | 93 | 77.42% |
| bert | 123 | missed_extra_pair | 1582 | 5427 | 29.15% |
| roberta | 2026 | partial_multi_label | 473 | 837 | 56.51% |
| roberta | 2026 | rare_false_negative | 87 | 93 | 93.55% |
| roberta | 2026 | missed_extra_pair | 1216 | 5427 | 22.41% |
| distilbert | 123 | partial_multi_label | 449 | 837 | 53.64% |
| distilbert | 123 | rare_false_negative | 90 | 93 | 96.77% |
| distilbert | 123 | missed_extra_pair | 1140 | 5427 | 21.01% |

Ba nhóm có thể chồng lấp; không cộng tỷ lệ thành 100%. `pairs.csv` đếm FN(A)+FP(B) đồng thời, khác với nhãn thật đồng xuất hiện. Một câu có thể đóng góp nhiều cặp.

`examples.csv` lấy cùng tập ID cho ba mô hình; `error_present=false` nghĩa là mô hình đó không gặp nhóm lỗi đang xét tại ID này. Đọc text, nhãn, điểm rồi điền `manual_linguistic_notes`; không tự quy kết mỉa mai/phủ định hoặc nguyên nhân.

Định nghĩa và quy trình: `docs/ERROR_ANALYSIS.md`. Manifest lưu nguồn/hashes.


Dự đoán đã có trước khi đọc lỗi; dùng ngưỡng cố định 0.5. Checkpoint đại diện của mỗi kiến trúc được chọn bằng validation, không chọn lại theo các câu test này. Đây là ba ví dụ minh họa của các nhóm lỗi có thể chồng lấp; không đại diện toàn bộ phân bố lỗi.

Nhận xét do trợ lý ghi sau khi đọc văn bản, nhãn thật và 28 scores đã lưu. Nhóm cần tự xác nhận cách diễn giải trước khi nộp/bảo vệ. Không thay nhãn thật, scores, ngưỡng hay cấu hình sau khi xem test.

### Case 1: partial_multi_label — ID eczj48j

Văn bản: “This!!! 🐃 and 💍 for your hard work!”.

Nhãn thật: admiration, excitement, neutral.

**Bảng 5-2g. Ba C trên cùng ID eczj48j.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | caring | admiration, excitement, neutral | caring | admiration=0.3333; excitement=0.0121; neutral=0.0666; caring=0.5696 | False |
| C2 RoBERTa | 2026 | admiration | excitement, neutral | không | admiration=0.6858; excitement=0.0218; neutral=0.0426 | True |
| C3 DistilBERT | 123 | admiration | excitement, neutral | không | admiration=0.6251; excitement=0.0158; neutral=0.0455 | True |

Nhận xét: Văn bản rất ngắn, có dấu chấm than, emoji và cụm 'hard work'. Đây là các dấu hiệu có thể khiến việc suy ra đủ bộ nhãn khó hơn, nhưng không chứng minh nguyên nhân trong mô hình. C1 dự đoán caring và bỏ cả ba nhãn thật; cờ partial=False ở C1 chỉ vì không có TP, không có nghĩa là dự đoán đúng. C2/C3 nhận ra admiration nhưng bỏ excitement và neutral. Giữ nguyên ground truth kể cả neutral đồng xuất hiện.

### Case 2: rare_false_negative — ID ed0jr9i

Văn bản: “Try nonchalantly handing them your card as if they had dropped it. I think its normal to be shy. *handing on exit, otherwise it could get awkward”.

Nhãn thật: embarrassment.

**Bảng 5-2h. Ba C trên cùng ID ed0jr9i.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | embarrassment | không | không | embarrassment=0.7527 | False |
| C2 RoBERTa | 2026 | không nhãn | embarrassment | không | embarrassment=0.3418 | True |
| C3 DistilBERT | 123 | không nhãn | embarrassment | không | embarrassment=0.1956 | True |

Nhận xét: Các từ 'shy', 'awkward' và tình huống đưa danh thiếp là dấu hiệu ngôn ngữ về sự ngượng ngùng. C1 nhận ra embarrassment; C2/C3 có score nhãn này dưới 0.5 nên bỏ sót. Embarrassment có 303 mẫu train và thuộc nhóm năm nhãn hiếm đã xác định từ train. Không suy diễn cơ chế attention, nguyên nhân do độ dài hoặc tác dụng của weighting từ riêng một ví dụ.

### Case 3: missed_extra_pair — ID eczcvgx

Văn bản: “I always plan that, my wife usually has other ideas though. ”.

Nhãn thật: neutral.

**Bảng 5-2i. Ba C trên cùng ID eczcvgx.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | neutral | không | không | neutral=0.6780 | False |
| C2 RoBERTa | 2026 | neutral | không | không | neutral=0.6429 | False |
| C3 DistilBERT | 123 | approval | neutral | approval | neutral=0.4173; approval=0.5729 | True |

Nhận xét: Câu kể về dự định và ý kiến khác của vợ; không có từ thể hiện sự tán thành rõ ràng. Ground truth là neutral. C1/C2 trả đúng neutral; C3 chọn approval và bỏ neutral vì hai score nằm ở hai phía ngưỡng 0.5. Đây là cặp FN neutral / FP approval ở C3, không phải bằng chứng chắc chắn về mỉa mai hay cảm xúc thật của người viết.

### Cách giải thích khi bảo vệ

Cờ nhóm lỗi chỉ trả lời mẫu có thuộc đúng định nghĩa nhóm đó hay không; cờ False không chứng minh mọi nhãn đều đúng. Ví dụ case 1, C1 sai hoàn toàn nhưng không thuộc lỗi nhận được một phần nhãn. C1 thắng trung bình theo tiêu chí chọn trên validation vẫn có thể thua ở một câu riêng.

Scores là đầu ra sigmoid của các classifier, không phải xác suất đã được kiểm chuẩn. Các quan sát trên không xác lập quan hệ nhân quả. Số lỗi toàn tập xem group_summary.csv; không cộng các nhóm chồng lấp.

Nguồn đối chiếu: reports/errors_test_standard_fixed/examples.csv và manifest.json. Scores trong bảng làm tròn bốn chữ số; CSV giữ độ chính xác gốc.


Ba nhóm tự động gồm bỏ sót một phần nhãn, bỏ nhãn hiếm và cùng lúc FN/FP; các nhóm có thể chồng lấp. Ba case được diễn giải từ câu/nhãn/scores, không tự quy thành mỉa mai hoặc nguyên nhân trong encoder. Mỗi ví dụ đối chiếu giữ cùng ID và true labels. Một câu có thể đóng góp nhiều cặp FN/FP; không cộng mọi cặp để tính số câu sai.

Ví dụ A standard @0,5 đã kiểm: `eczwil0` bỏ sót pride; `ed832y6` dự đoán love thừa; `eczdvun` tìm gratitude nhưng thiếu admiration. Đây là ví dụ A thực tế, chưa phải lỗi của các C. Chỉ suy ra nguyên nhân sau khi đọc ngữ cảnh/annotation.

## 4. Demo D

Selection manifest đã có: `bert`, seed `123`. Luật lựa chọn lấy từ validation; cần đọc `selected_model.json` và hồ sơ demo.

Kiểm suy luận lúc 2026-10-08T16:31:14.108678+00:00: PASS, model bert, seed 123; đối chiếu 3 câu với scores validation và kiểm luồng nhập. Đây là kiểm suy luận, không phải kiểm UI.

Kiểm UI riêng lúc 2026-10-08T16:33:39.810Z: PASS, 28/28 hàng trong DOM. Ảnh thực tế: reports/demo_ui/demo_ui.png; hash nằm trong evidence.json. Hồ sơ ghi nhận thời điểm kiểm, không bảo đảm server đang chạy khi đọc báo cáo.

Kiểm mã: 69/69 unit tests đạt; notebook 3/3 đạt. Nguồn verification_project.json; phép kiểm mã không thay benchmark full.

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


### 1. Năm nhãn hiếm: trước và sau cải tiến

**Bảng chính: TEST.** Ngưỡng của các cấu hình được khóa trên validation trước khi đánh giá test. Bảng giữ cả mức tăng, giảm và không đổi.

Nhãn hiếm lấy theo năm support thấp nhất trên train từ metadata A standard; hòa theo ID nhãn. A: standard fixed → balanced tuned. C1/C2/C3: cùng kiến trúc, cùng ít nhất ba seed, fixed → tuned. C báo mean ± sample std (`ddof=1`); Δ tính theo cặp seed. A là một run, không tạo std bằng 0.

| Nhãn | Mô hình | Train + | Validation + | Test + | F1 trước | F1 sau | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| grief | A: TF-IDF + LR | 77 | 13 | 6 | 0.0000 | 0.4615 | +0.4615 |
| grief | C1: BERT | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0444 ± 0.0770 | +0.0444 ± 0.0770 |
| grief | C2: RoBERTa | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| grief | C3: DistilBERT | 77 | 13 | 6 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| pride | A: TF-IDF + LR | 111 | 15 | 16 | 0.0000 | 0.4167 | +0.4167 |
| pride | C1: BERT | 111 | 15 | 16 | 0.1133 ± 0.1112 | 0.4713 ± 0.0445 | +0.3580 ± 0.0843 |
| pride | C2: RoBERTa | 111 | 15 | 16 | 0.0000 ± 0.0000 | 0.1000 ± 0.1732 | +0.1000 ± 0.1732 |
| pride | C3: DistilBERT | 111 | 15 | 16 | 0.0000 ± 0.0000 | 0.4458 ± 0.0710 | +0.4458 ± 0.0710 |
| relief | A: TF-IDF + LR | 153 | 18 | 11 | 0.0000 | 0.1176 | +0.1176 |
| relief | C1: BERT | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.2356 ± 0.1042 | +0.2356 ± 0.1042 |
| relief | C2: RoBERTa | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.0800 ± 0.1386 | +0.0800 ± 0.1386 |
| relief | C3: DistilBERT | 153 | 18 | 11 | 0.0000 ± 0.0000 | 0.0417 ± 0.0722 | +0.0417 ± 0.0722 |
| nervousness | A: TF-IDF + LR | 164 | 21 | 23 | 0.0000 | 0.1714 | +0.1714 |
| nervousness | C1: BERT | 164 | 21 | 23 | 0.3318 ± 0.0503 | 0.3457 ± 0.0408 | +0.0139 ± 0.0773 |
| nervousness | C2: RoBERTa | 164 | 21 | 23 | 0.0000 ± 0.0000 | 0.3078 ± 0.0443 | +0.3078 ± 0.0443 |
| nervousness | C3: DistilBERT | 164 | 21 | 23 | 0.0000 ± 0.0000 | 0.3316 ± 0.1238 | +0.3316 ± 0.1238 |
| embarrassment | A: TF-IDF + LR | 303 | 35 | 37 | 0.0000 | 0.2778 | +0.2778 |
| embarrassment | C1: BERT | 303 | 35 | 37 | 0.5089 ± 0.0175 | 0.4831 ± 0.0536 | -0.0257 ± 0.0370 |
| embarrassment | C2: RoBERTa | 303 | 35 | 37 | 0.1222 ± 0.1347 | 0.4461 ± 0.0268 | +0.3238 ± 0.1149 |
| embarrassment | C3: DistilBERT | 303 | 35 | 37 | 0.1297 ± 0.0258 | 0.3730 ± 0.0700 | +0.2432 ± 0.0795 |

Support là số câu có nhãn thật, không phải số lần model dự đoán nhãn. Δ > 0 là tăng, Δ < 0 là giảm. Bảng làm tròn bốn chữ số; số gốc và mọi mức giảm được giữ trong `rare_before_after.csv`. Các nhóm C thiếu seed/config/revision nhất quán sẽ không có mean/std.



Đã có bảng test trước/sau với support, mean±std và cả mức giảm/không đổi. C weighting chưa chạy; chín C standard dùng threshold tuning, không làm encoder học thêm. Nhóm cần đọc và xác nhận các case, giữ nguyên protocol và artifacts khi bàn giao.

## 6. Vai trò, đóng góp và phần chưa hoàn thành

| Thành viên | Vai trò | Bằng chứng hiện có | % công sức |
| --- | --- | --- | --- |
| Bảo Duy Nguyễn | A; điều phối B; data/metrics và bảng nâng cao | A full validation; B full đã có | ____________________ |
| Quốc Khánh | C1 BERT; phần đầu/tổng hợp báo cáo | C1 3/3 seed full; xem bảng run | ____________________ |
| Đức Trí | C2 RoBERTa; hỗ trợ/bàn giao B | C2 3/3 seed full; xem bảng run | ____________________ |
| Nhật Huy | C3 DistilBERT; tích hợp demo best C | C3 3/3 seed full; demo theo hồ sơ | ____________________ |

Đức Trí là “Thợ Săn Thập Cẩm”; nhóm có bốn người. Tỷ lệ công sức để trống cho nhóm thống nhất bằng artifacts, không tự đánh đồng phân công và việc đã làm.

Theo summary hiện đọc: `complete=true`. Những phần cần bổ sung:

- Không còn mục thiếu trong summary A/B/C; hồ sơ kiểm demo và đọc lỗi đã nêu trên. Nhóm cần xác nhận thông tin hành chính, đóng góp và kiểm demo khi chuyển máy.

Mức hoàn thành được ghi theo artifacts hiện có; không xác nhận điểm số hoặc đã nộp giảng viên. Các thông tin hành chính và tỷ lệ đóng góp vẫn để trống cho nhóm xác nhận.

## 7. Hồ sơ và bước hoàn tất

Summary: `reports/project_results/summary.json`; SHA-256: `2c73aa31fad9df0a8b0b46bedbb3c69c1316fa4e225f32529811e80ccd2e09b6`. Cấu hình, môi trường và thời gian thực tế nằm trong `run_metadata.json`. Thời gian có gián đoạn do ngủ máy không được dùng như benchmark tốc độ được kiểm soát.

Kho chung: https://github.com/trangkhanh-ai/goemotions-multilabel-classification

Quy trình đã áp dụng: đủ seed → tổng hợp validation → chọn C/ngưỡng → khóa protocol → test → đối chiếu lỗi/demo → hoàn thiện báo cáo. Bước tiếp theo là đọc code, tập bảo vệ, điền thông tin hành chính và xác nhận đóng góp. Script tạo báo cáo chỉ đọc artifacts, không chạy GPU, không thay summary và không tự gửi/nộp.

## Tài liệu tham khảo

[1] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, “GoEmotions: A Dataset of Fine-Grained Emotions,” in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics, 2020, pp. 4040–4054, doi: 10.18653/v1/2020.acl-main.372. https://aclanthology.org/2020.acl-main.372/

[2] scikit-learn developers, “precision_recall_fscore_support,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

[3] Facebook AI, “bart-large-mnli model card,” Hugging Face. Accessed: Oct. 8, 2026. [Online]. Available: https://huggingface.co/facebook/bart-large-mnli

[4] scikit-learn developers, “Tuning the decision threshold for class prediction,” scikit-learn 1.7.2 documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://scikit-learn.org/1.7/modules/classification_threshold.html

[5] Gradio, “Quickstart,” Gradio documentation. Accessed: Oct. 8, 2026. [Online]. Available: https://www.gradio.app/guides/quickstart
