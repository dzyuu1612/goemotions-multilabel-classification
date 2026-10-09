# Ba case phân tích lỗi trên test: đối chiếu cùng ID giữa C1/C2/C3

Dự đoán đã có trước khi đọc lỗi; dùng ngưỡng cố định 0.5. Checkpoint đại diện của mỗi kiến trúc được chọn bằng validation, không chọn lại theo các câu test này. Đây là ba ví dụ minh họa của các nhóm lỗi có thể chồng lấp; không đại diện toàn bộ phân bố lỗi.

Nhận xét do trợ lý ghi sau khi đọc văn bản, nhãn thật và 28 scores đã lưu. Nhóm cần tự xác nhận cách diễn giải trước khi nộp/bảo vệ. Không thay nhãn thật, scores, ngưỡng hay cấu hình sau khi xem test.

## Case 1: partial_multi_label — ID eczj48j

Văn bản: “This!!! 🐃 and 💍 for your hard work!”.

Nhãn thật: admiration, excitement, neutral.

**Bảng 5-2g. Ba C trên cùng ID eczj48j.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | caring | admiration, excitement, neutral | caring | admiration=0.3333; excitement=0.0121; neutral=0.0666; caring=0.5696 | False |
| C2 RoBERTa | 2026 | admiration | excitement, neutral | không | admiration=0.6858; excitement=0.0218; neutral=0.0426 | True |
| C3 DistilBERT | 123 | admiration | excitement, neutral | không | admiration=0.6251; excitement=0.0158; neutral=0.0455 | True |

Nhận xét: Văn bản rất ngắn, có dấu chấm than, emoji và cụm 'hard work'. Đây là các dấu hiệu có thể khiến việc suy ra đủ bộ nhãn khó hơn, nhưng không chứng minh nguyên nhân trong mô hình. C1 dự đoán caring và bỏ cả ba nhãn thật; cờ partial=False ở C1 chỉ vì không có TP, không có nghĩa là dự đoán đúng. C2/C3 nhận ra admiration nhưng bỏ excitement và neutral. Giữ nguyên ground truth kể cả neutral đồng xuất hiện.

## Case 2: rare_false_negative — ID ed0jr9i

Văn bản: “Try nonchalantly handing them your card as if they had dropped it. I think its normal to be shy. *handing on exit, otherwise it could get awkward”.

Nhãn thật: embarrassment.

**Bảng 5-2h. Ba C trên cùng ID ed0jr9i.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | embarrassment | không | không | embarrassment=0.7527 | False |
| C2 RoBERTa | 2026 | không nhãn | embarrassment | không | embarrassment=0.3418 | True |
| C3 DistilBERT | 123 | không nhãn | embarrassment | không | embarrassment=0.1956 | True |

Nhận xét: Các từ 'shy', 'awkward' và tình huống đưa danh thiếp là dấu hiệu ngôn ngữ về sự ngượng ngùng. C1 nhận ra embarrassment; C2/C3 có score nhãn này dưới 0.5 nên bỏ sót. Embarrassment có 303 mẫu train và thuộc nhóm năm nhãn hiếm đã xác định từ train. Không suy diễn cơ chế attention, nguyên nhân do độ dài hoặc tác dụng của weighting từ riêng một ví dụ.

## Case 3: missed_extra_pair — ID eczcvgx

Văn bản: “I always plan that, my wife usually has other ideas though. ”.

Nhãn thật: neutral.

**Bảng 5-2i. Ba C trên cùng ID eczcvgx.**

| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |
|---|---:|---|---|---|---|---|
| C1 BERT | 123 | neutral | không | không | neutral=0.6780 | False |
| C2 RoBERTa | 2026 | neutral | không | không | neutral=0.6429 | False |
| C3 DistilBERT | 123 | approval | neutral | approval | neutral=0.4173; approval=0.5729 | True |

Nhận xét: Câu kể về dự định và ý kiến khác của vợ; không có từ thể hiện sự tán thành rõ ràng. Ground truth là neutral. C1/C2 trả đúng neutral; C3 chọn approval và bỏ neutral vì hai score nằm ở hai phía ngưỡng 0.5. Đây là cặp FN neutral / FP approval ở C3, không phải bằng chứng chắc chắn về mỉa mai hay cảm xúc thật của người viết.

## Cách giải thích khi bảo vệ

Cờ nhóm lỗi chỉ trả lời mẫu có thuộc đúng định nghĩa nhóm đó hay không; cờ False không chứng minh mọi nhãn đều đúng. Ví dụ case 1, C1 sai hoàn toàn nhưng không thuộc lỗi nhận được một phần nhãn. C1 thắng trung bình theo tiêu chí chọn trên validation vẫn có thể thua ở một câu riêng.

Scores là đầu ra sigmoid của các classifier, không phải xác suất đã được kiểm chuẩn. Các quan sát trên không xác lập quan hệ nhân quả. Số lỗi toàn tập xem reports/errors_test_standard_fixed/counts.csv; không cộng các nhóm chồng lấp.

Nguồn đối chiếu: reports/errors_test_standard_fixed/examples.csv và manifest.json. Scores trong bảng làm tròn bốn chữ số; CSV giữ độ chính xác gốc.
