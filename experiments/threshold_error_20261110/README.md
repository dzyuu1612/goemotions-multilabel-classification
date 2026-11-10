# Phân tích ngưỡng và lỗi sau nhận xét của giảng viên

Mở [notebook đã chạy](GoEmotions_PhanTich_Nguong_Loi_DaChay.ipynb) để xem hoặc tính lại từ điểm đã lưu. Không nạp mô hình, không huấn luyện lại. Cài các phụ thuộc theo README ở gốc dự án. Các điểm A/B được khôi phục có kiểm SHA-256 từ runtime ZIP đã công bố; validation chính thức chỉ được tải nếu chưa có, ở đúng revision và hash. Không tải weights Transformer để thực hiện phân tích này.

## Quy trình đã thực hiện

1. [Ghi quyết định](DECISION_PHANTICH.md), commit `35864a7`, trước các bảng test mới của lượt phân tích này. Test đã được xem ở báo cáo cũ, không gọi đây là đăng ký trước toàn bộ nghiên cứu.
2. Chỉ đọc validation, tính lại ngưỡng chung và riêng trên 19 mức; đối chiếu hồ sơ cũ. Cả 11 bộ kết quả lệch **0/28 nhãn**, ngưỡng chung cũng khớp.
3. Commit [tệp khóa validation](validation_lock.json) tại `1fe2ee2` trước khi chấm test. Ngưỡng chung đã được ghi trong kế hoạch ngày 03/11 và lưu cùng các checkpoint cũ; lần này công bố thêm phép đối chiếu.
4. Đọc điểm test, kiểm hash biên nhận, ID và thứ tự nhãn. Tính đủ 33 bộ metric; kiểm độc lập với scikit-learn. Giữ mọi kết quả giảm.

Muốn tái hiện bảng đã công bố, dùng notebook; nó chỉ đọc ngưỡng đã khóa và kiểm lại CSV. Nếu chạy lại quy trình khóa cho một lần phân tích mới, ghi quyết định mới đúng thời điểm rồi dùng:

```powershell
python experiments/threshold_error_20261110/analysis.py freeze
git add experiments/threshold_error_20261110/validation_lock.json experiments/threshold_error_20261110/threshold_audit.json experiments/threshold_error_20261110/validation_curves.csv
git commit -m "Freeze audited validation thresholds"
python experiments/threshold_error_20261110/analysis.py evaluate
```

`evaluate` từ chối tệp khóa chưa commit hoặc bị sửa. Bước khôi phục validation không giải nén payload điểm test. Không sửa/checkpoint/ngưỡng để làm test đẹp hơn.

## Kết quả chính

| Mô hình | Macro-F1 0,5 | Macro-F1 chung | Macro-F1 riêng |
|---|---:|---:|---:|
| A TF-IDF cân bằng, một lượt | 0,4441 | 0,4530 | 0,4493 |
| B BART-MNLI, một lượt | 0,1035 | 0,1473 | 0,1609 |
| C1 BERT, ba seed | 0,4696 ± 0,0055 | 0,4794 ± 0,0038 | 0,4938 ± 0,0075 |
| C2 RoBERTa, ba seed | 0,4765 ± 0,0079 | 0,4985 ± 0,0121 | 0,5101 ± 0,0076 |
| C3 DistilBERT, ba seed | 0,4729 ± 0,0039 | 0,4883 ± 0,0055 | 0,4972 ± 0,0027 |

Std mẫu dùng `ddof=1`, không phải khoảng tin cậy. A là phản ví dụ cho khẳng định ngưỡng riêng luôn hơn ngưỡng chung. F1 tăng cũng không bảo đảm Hamming Loss hoặc số câu đúng hoàn toàn tốt lên.

## Bảng dữ liệu

- [Từng lượt và ba chế độ](results/modes_per_checkpoint.csv), [mean/std](results/modes_mean_std.csv).
- [28 ngưỡng, support val/test, nhãn hiếm](results/thresholds_28.csv); [TP/FP/FN từng nhãn](results/label_tpfpfn.csv).
- [Năm loại câu loại trừ nhau](results/error_taxonomy.csv), [giao ba tập lỗi cũ](results/error_overlap.csv).
- [Lỗi sát ngưỡng](results/near_miss.csv), [các cặp FN→FP](results/pair_counts_long.csv).
- [F1 validation từng epoch](results/epoch_table_val_macro_f1.csv), [BCE từng epoch](results/epoch_table_val_bce.csv).
- [Kiểm caring/optimism](results/caring_optimism_verified.csv), [biên nhận kiểm số](results/verification.json).

Năm loại là một phân hoạch, tổng mỗi model/seed/chế độ bằng 5.427. “Đúng một phần” có thể kèm FP; “đoán sai nhãn” vẫn tính cả FN và FP. G1/G2/G3 là ba tập có thể giao nhau và có mẫu số khác nhau; không cộng các tỷ lệ. Near-miss là quy ước mô tả, không dùng để sửa ngưỡng sau test.

## Phần cần hai người thực hiện

Hai phiếu [Duy](results/phieu_doc_lap_Duy.csv) và [Huy](results/phieu_doc_lap_Huy.csv) có 145 mục sau loại trùng giữa mô hình. Đọc [quy tắc gán nhãn](CODEBOOK.md), làm độc lập rồi chạy `annotation_agreement.py`. Chưa có hai phiếu thật nên **chưa có tỷ lệ đồng thuận hoặc kết luận nguyên nhân**. Không điền thay hai người hoặc gọi số do mô hình sinh là nhãn của con người.

Tỷ trọng công sức Bảng B.1 do nhóm tự điền. Giữ các hạn chế: test đã được quan sát, 37 dòng test trùng train, chưa có ablation và chưa đo chi phí thời gian có đối chứng.
