# So sánh lỗi C1/C2/C3 — validation, standard, fixed

Đếm trên cùng ID. Seed đại diện chọn bằng validation @0,5; đây là phân tích checkpoint đại diện, không phải trung bình lỗi qua ba seed.

| Kiến trúc | Seed | Nhóm lỗi | Số câu | Mẫu phù hợp định nghĩa | Tỷ lệ trong mẫu phù hợp |
|---|---:|---|---:|---:|---:|
| bert | 123 | partial_multi_label | 509 | 878 | 57.97% |
| bert | 123 | rare_false_negative | 78 | 102 | 76.47% |
| bert | 123 | missed_extra_pair | 1580 | 5426 | 29.12% |
| roberta | 2026 | partial_multi_label | 504 | 878 | 57.40% |
| roberta | 2026 | rare_false_negative | 93 | 102 | 91.18% |
| roberta | 2026 | missed_extra_pair | 1237 | 5426 | 22.80% |
| distilbert | 123 | partial_multi_label | 462 | 878 | 52.62% |
| distilbert | 123 | rare_false_negative | 94 | 102 | 92.16% |
| distilbert | 123 | missed_extra_pair | 1171 | 5426 | 21.58% |

Ba nhóm có thể chồng lấp; không cộng tỷ lệ thành 100%. `pairs.csv` đếm FN(A)+FP(B) đồng thời, khác với nhãn thật đồng xuất hiện. Một câu có thể đóng góp nhiều cặp.

`examples.csv` lấy cùng tập ID cho ba mô hình; `error_present=false` nghĩa là mô hình đó không gặp nhóm lỗi đang xét tại ID này. Đọc text, nhãn, điểm rồi điền `manual_linguistic_notes`; không tự quy kết mỉa mai/phủ định hoặc nguyên nhân.

Định nghĩa và quy trình: `docs/ERROR_ANALYSIS.md`. Manifest lưu nguồn/hashes.