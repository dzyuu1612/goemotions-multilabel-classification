# Gán nhãn nguyên nhân lỗi: hai người độc lập

Hai tệp mẫu là `results/phieu_doc_lap_Duy.csv` và `results/phieu_doc_lap_Huy.csv`. Mỗi tệp có 145 mục câu–cặp lỗi sau loại trùng giữa mô hình. Hai người làm riêng, không xem phiếu của nhau trước khi nộp. Các dòng trống là phiếu chưa làm, không phải một nhãn đánh giá.

Mỗi mục chọn đúng một giá trị cho `nhan_thu_cong`:

| Mã | Cách dùng |
|---|---|
| `thieu_ngu_canh` | Văn bản đòi hỏi thông tin ngoài câu mới xác định được cảm xúc; ghi rõ phần thông tin thiếu. |
| `giao_thoa_sac_thai` | Nhãn thật và nhãn dự đoán gần nhau về cách hiểu; giải thích bằng từ/cụm từ trong câu. |
| `phu_dinh_mia_mai` | Có dấu hiệu phủ định, mỉa mai hoặc đảo nghĩa; trích đúng dấu hiệu trong ghi chú. |
| `van_ban_bat_thuong` | Ký hiệu, tên ẩn, cách viết hoặc lỗi văn bản gây khó đọc; nêu cụ thể. |
| `khong_du_can_cu` | Không có căn cứ đủ mạnh để gán một nhóm khác; không đoán nguyên nhân từ điểm số. |

Nếu có nhiều khả năng, chọn khả năng có bằng chứng rõ nhất và ghi các khả năng khác vào `ghi_chu`. Không dùng nhãn ít mẫu như nguyên nhân tự động cho mọi câu relief. Các nhãn này là diễn giải của người đọc; ngay cả hai người đồng ý cũng chưa chứng minh quan hệ nhân quả hoặc tác động của encoder.

Giữ nguyên các cột ID, văn bản và nhãn. Sau khi có hai phiếu điền thật:

```powershell
python experiments/threshold_error_20261110/annotation_agreement.py phieu_Duy_da_dien.csv phieu_Huy_da_dien.csv agreement.json
```

Mã từ chối phiếu thiếu dòng, ID trùng, thay ngữ cảnh hoặc nhãn còn trống. Nó báo số đồng ý/tổng, tỷ lệ đồng ý và Cohen's kappa. Không đưa kết quả đồng thuận vào báo cáo khi chưa có hai phiếu thật.
