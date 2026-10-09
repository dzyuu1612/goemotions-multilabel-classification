# So sánh lỗi C1/C2/C3 — test, standard, fixed

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