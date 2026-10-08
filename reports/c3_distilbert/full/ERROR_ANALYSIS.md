# Phân tích lỗi C3 DistilBERT — Nhật Huy

Phân tích trên **5.426 mẫu validation**, ngưỡng cơ sở **0,5**, giữ nguyên nhãn nguồn.
Seed **123** đại diện C3 vì có Macro-F1 validation @0,5 cao nhất trong ba seed theo protocol;
không thay mean/std bằng seed tốt nhất và chưa chọn kiến trúc thắng C1/C2/C3.

## 1. Số lượng lỗi có thể kiểm lại

FN: nhãn thật bị bỏ sót; FP: nhãn dự đoán thừa. Ba nhóm đếm tự động là:

- FN và FP cùng câu: có cả ít nhất một FN và một FP.
- Thiếu một phần nhãn: mẫu có ít nhất hai nhãn thật, nhận đúng ít nhất một và bỏ sót ít nhất một.
- Sai nhãn hiếm/neutral: có FN hoặc FP ở năm nhãn hiếm đã xác định từ train hoặc neutral.

Các nhóm **chồng lấn**, không cộng số hàng thành tổng số câu sai. Các nhóm diễn giải nội dung
ở phần 2 cũng không phải bộ nhãn lỗi đã gán thủ công cho toàn bộ validation.


| group                      |   42 |   123 |   2026 |
|:---------------------------|-----:|------:|-------:|
| FN_va_FP_cung_cau          | 1167 |  1168 |   1184 |
| sai_nhan_hiem_hoac_neutral | 1255 |  1218 |   1224 |
| thieu_mot_phan_nhan_that   |  458 |   480 |    463 |


Nguồn: [error_group_counts.csv](error_group_counts.csv). Cặp FN→FP đếm theo từng mẫu có cả hai
nhãn tương ứng; một mẫu có thể đóng góp nhiều cặp. Đây không phải confusion matrix single-label.
Các cặp nhiều nhất seed 123 được lấy trực tiếp từ bảng sau:


| missed_label   | extra_label   |   count |   missed_label_total_fn |   fraction_of_fn |   example_row |
|:---------------|:--------------|--------:|------------------------:|-----------------:|--------------:|
| approval       | neutral       |      87 |                     312 |        0.278846  |           106 |
| disapproval    | neutral       |      71 |                     241 |        0.294606  |            22 |
| annoyance      | neutral       |      53 |                     266 |        0.199248  |           197 |
| neutral        | curiosity     |      45 |                     711 |        0.0632911 |             0 |
| realization    | neutral       |      32 |                     113 |        0.283186  |            62 |
| neutral        | approval      |      29 |                     711 |        0.0407876 |           279 |
| caring         | neutral       |      27 |                     109 |        0.247706  |           498 |
| curiosity      | neutral       |      27 |                     155 |        0.174194  |           186 |
| neutral        | amusement     |      27 |                     711 |        0.0379747 |          1185 |
| neutral        | admiration    |      26 |                     711 |        0.0365682 |            18 |


## 2. Đọc các ví dụ thật

Nguồn [error_examples.json](error_examples.json) lấy tối đa 12 mẫu mỗi nhóm tự động theo ID tăng dần.
Dưới đây chọn bảy mẫu trong danh sách đó để giải thích ba dạng khó khăn; đây không phải mẫu ngẫu
nhiên đại diện toàn tập. Nội dung và nhãn giữ nguyên văn, score sigmoid chưa được hiệu chuẩn xác suất.
Nhận xét là diễn giải thủ công dựa trên câu và đầu ra, không phải kết luận nhân quả.


### Cảm xúc gần nghĩa và ranh giới nhãn


**ID `eczqguq` — validation, seed 123**

> Woot! Happy New Year!


| nhãn       |   gold |    score |   dự đoán @0,5 |
|:-----------|-------:|---------:|---------------:|
| joy        |      1 | 0.443437 |              0 |
| excitement |      0 | 0.558164 |              1 |


FN: joy; FP: excitement.

Câu chúc vui mừng được gán joy, trong khi mô hình chọn excitement. Hai điểm nằm hai phía ngưỡng 0,5. Đây là minh họa ranh giới giữa hai cảm xúc tích cực; một mẫu không đủ kết luận nguyên nhân hoặc mức độ phổ biến.


**ID `eczuvim` — validation, seed 123**

> 6 and he died of cancer iirc


| nhãn    |   gold |    score |   dự đoán @0,5 |
|:--------|-------:|---------:|---------------:|
| grief   |      1 | 0.006402 |              0 |
| sadness |      0 | 0.625659 |              1 |


FN: grief; FP: sadness.

Nhãn grief bị bỏ sót, sadness được dự đoán thừa. Nội dung nói về cái chết phù hợp để thảo luận ranh giới grief/sadness. Grief ít mẫu, nhưng ví dụ này không tự chứng minh mất cân bằng là nguyên nhân.


### Bỏ sót một phần cảm xúc trong câu đa nhãn


**ID `eczfzzk` — validation, seed 123**

> I drove off 170 onto 40 and was surprised by the lack of brakelights.... keep it closed haha


| nhãn      |   gold |    score |   dự đoán @0,5 |
|:----------|-------:|---------:|---------------:|
| amusement |      1 | 0.328294 |              0 |
| surprise  |      1 | 0.653454 |              1 |


FN: amusement; FP: không.

Nhận đúng surprise nhưng bỏ sót amusement. Người đọc có thể liên hệ surprised với surprise và haha với amusement; đây là diễn giải nội dung, chưa phải phép giải thích cơ chế chú ý của mô hình.


**ID `ecztw5u` — validation, seed 123**

> Sorry for your losses. Keep up the awesome work. Your son will remember your efforts and hopefully be part of your life soon


| nhãn       |   gold |    score |   dự đoán @0,5 |
|:-----------|-------:|---------:|---------------:|
| admiration |      1 | 0.495647 |              0 |
| optimism   |      1 | 0.698114 |              1 |
| remorse    |      1 | 0.066661 |              0 |


FN: admiration, remorse; FP: không.

Câu gồm lời chia buồn, lời khen và hy vọng; chỉ optimism vượt ngưỡng. Admiration sát ngưỡng (0,495647) còn remorse thấp hơn nhiều (0,066661). Các nhãn bị bỏ sót không nhất thiết cùng được sửa bằng một mức giảm ngưỡng.


### Diễn đạt hàm ý và lỗi với neutral


**ID `eczj0in` — validation, seed 123**

> Kinda let down tbh


| nhãn           |   gold |    score |   dự đoán @0,5 |
|:---------------|-------:|---------:|---------------:|
| annoyance      |      1 | 0.026834 |              0 |
| disappointment |      1 | 0.019797 |              0 |
| neutral        |      0 | 0.903061 |              1 |


FN: annoyance, disappointment; FP: neutral.

Cụm let down diễn đạt cảm xúc theo cách khẩu ngữ. Mô hình cho neutral điểm cao và bỏ sót cả hai nhãn thật. Có thể xem đây là khó khăn với cách diễn đạt này; chưa có thí nghiệm chứng minh viết tắt tbh là nguyên nhân.


**ID `eczdu8p` — validation, seed 123**

> Oooh yes, great idea. Too bad there aren't any other duplicate names


| nhãn        |   gold |    score |   dự đoán @0,5 |
|:------------|-------:|---------:|---------------:|
| admiration  |      1 | 0.901688 |              1 |
| disapproval |      1 | 0.045034 |              0 |


FN: disapproval; FP: không.

Mô hình chọn admiration nhưng bỏ disapproval. Sự đối lập great idea / Too bad có thể mang hàm ý; không có nhãn sarcasm trong dữ liệu đang dùng nên không khẳng định đây là lỗi châm biếm đã được kiểm chứng.


**ID `eczg2g7` — validation, seed 123**

> I'm in the Central Time Zone in the US so I have some time to go. But Happy New Year to you man!


| nhãn    |   gold |    score |   dự đoán @0,5 |
|:--------|-------:|---------:|---------------:|
| neutral |      1 | 0.020306 |              0 |
| joy     |      0 | 0.748874 |              1 |


FN: neutral; FP: joy.

Gold là neutral, mô hình chọn joy. Lời chúc Happy New Year có thể khiến người đọc liên tưởng niềm vui; đánh giá vẫn tính FP joy và FN neutral theo gold. Không tự sửa nhãn hoặc kết luận gold sai.


## 3. Liên hệ nhãn hiếm và ngưỡng

Grief có 77 mẫu train và 13 mẫu validation. F1 grief bằng 0 ở cả ba seed, cả ngưỡng 0,5
và ngưỡng riêng đã khảo sát. Không thể viết rằng tuning cải thiện tất cả nhãn hiếm.
Relief có F1 tuned trung bình 0,048780 ± 0,084490, cho thấy kết quả không ổn định giữa ba seed.
Pride, nervousness và embarrassment có F1 trung bình tăng; xem đủ năm nhãn trong
[rare_labels_mean_std.csv](rare_labels_mean_std.csv), không chỉ chọn nhãn có lợi.

Tuning trên cùng validation tăng mean Macro-F1 từ 0,406084 lên 0,510645 và mean Micro-F1
0,572944 lên 0,602453. Tuy nhiên, micro precision giảm 0,709882 → 0,552489, micro recall tăng
0,480303 → 0,662435 và Hamming Loss tăng (xấu hơn) 0,030067 → 0,036715.
Đây là đánh đổi giữa bỏ sót và dự đoán thừa, không phải mọi metric đều tốt lên.
Ngưỡng vừa chọn vừa đo trên validation nên các số tuned có thể lạc quan; chưa có đánh giá test.

## 4. Phần bàn giao và giới hạn

C3 đã có ba dạng lỗi để thảo luận cùng ID, nhãn và scores thật. Khi nhận scores C1/C2,
cần đối chiếu trên cùng ID và quy tắc ngưỡng đã thống nhất để hoàn thành phân tích giữa
ba kiến trúc. Hiện chưa có bằng chứng kết luận dạng lỗi nào riêng của DistilBERT, hoặc
C1/C2 khắc phục được các mẫu này. Không thay nhãn, không bổ sung mẫu nhân tạo, không dùng test.

