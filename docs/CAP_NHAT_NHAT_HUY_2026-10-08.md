# Cập nhật công việc Nhật Huy — 08/10/2026

**Đề tài:** Phân loại cảm xúc đa nhãn trên văn bản mạng xã hội với GoEmotions.

**Phần phụ trách:** C3 DistilBERT và demo mô hình C tốt nhất sau khi nhóm có đủ kết quả.

**Trạng thái cuối ngày:** Đã hoàn thành phần C3 train/validation độc lập, notebook, báo cáo và demo C3. Cần kết quả C1/C2 để thực hiện phần so sánh và demo cuối của nhóm.

## 1. Công việc đã hoàn thiện hôm nay

| Công việc | Kết quả bàn giao |
|---|---|
| Hoàn tất bộ thí nghiệm full ba seed | Đủ seed 42, 123, 2026; giữ kết quả seed 42 đã chạy trước đó, hoàn tất các seed còn lại |
| Tổng hợp kết quả | Bảng từng seed và mean ± độ lệch chuẩn mẫu; đủ 28 nhãn, cùng hàm metrics của repo |
| Khảo sát ngưỡng riêng | Chọn ngưỡng từng nhãn trên validation của từng checkpoint; có bảng trước/sau và đủ năm nhãn hiếm |
| Phân tích lỗi | Thống kê ba nhóm lỗi; đọc bảy ví dụ validation thật, kèm ID, nhãn thật, nhãn dự đoán, FN/FP và scores |
| Notebook từng bước | 16 bước, 33 cell; cả 16 cell code đã chạy thành công; có bản HTML để đọc kết quả |
| Demo C3 | Nạp checkpoint seed 123; hỗ trợ ngưỡng 0,5 hoặc ngưỡng riêng; hiển thị scores đủ 28 nhãn |
| Kiểm tra và tài liệu | Đối chiếu app/CLI, kiểm input biên, kiểm khởi động lại server; hoàn thiện báo cáo và hướng dẫn bàn giao |

Chỉ sử dụng dữ liệu thật. Không tạo mẫu hoặc kết quả giả. Các lượt chạy bị gián đoạn được lưu log riêng và không đưa vào bảng kết quả.

## 2. Dữ liệu và cấu hình đã chạy

- GoEmotions simplified, giữ official split: **43.410 train, 5.426 validation**.
- C3 chưa đánh giá test. Phần thống kê EDA trước đây được trình bày riêng.
- DistilBERT-base-uncased; đầu phân loại 28 nhãn; BCEWithLogitsLoss và sigmoid.
- Max length 128; batch 16; learning rate 2e-5; 3 epoch cho mỗi seed.
- Ba seed: **42, 123, 2026**. Mỗi seed chọn checkpoint có Macro-F1 validation @0,5 cao nhất; cả ba chọn epoch 3.
- Dataset/model revision và cấu hình được ghi trong protocol; không trộn kết quả pilot với full.

## 3. Kết quả validation

### Kết quả cơ sở tại ngưỡng 0,5

| Seed | Macro-F1 | Micro-F1 |
|---|---:|---:|
| 42 | 0,403256 | 0,568935 |
| 123 | 0,411325 | 0,576176 |
| 2026 | 0,403670 | 0,573720 |
| **Mean ± sample std** | **0,406084 ± 0,004544** | **0,572944 ± 0,003683** |

Mean/std tính đủ ba seed, với độ lệch chuẩn mẫu `ddof=1`. Seed 123 chỉ là checkpoint đại diện trong C3 để minh họa; chưa có kết luận C3 thắng BERT hoặc RoBERTa.

### Sau chọn ngưỡng riêng từng nhãn

| Metric | Mean ± sample std |
|---|---:|
| Macro-F1 | 0,510645 ± 0,001230 |
| Micro-F1 | 0,602453 ± 0,003956 |

Ngưỡng được chọn và đo trên **cùng validation**, nên các số này có thể lạc quan; chưa phải kết quả test hoặc đánh giá độc lập. Tuning tăng recall/F1 nhưng micro precision giảm **0,709882 → 0,552489**, Hamming Loss tăng **0,030067 → 0,036715** (xấu hơn).

Năm nhãn hiếm được xác định từ train: grief, pride, relief, nervousness, embarrassment. **Grief vẫn có F1 bằng 0** ở cả ba seed trước/sau tuning; relief có kết quả không ổn định giữa các seed. Không kết luận tất cả nhãn hiếm đều được cải thiện.

## 4. Phân tích lỗi và kiểm chứng

Ba nhóm thống kê gồm: FN và FP cùng câu; bỏ sót một phần nhãn trong câu đa nhãn; lỗi nhãn hiếm/neutral. Các nhóm có thể chồng lấn, không cộng để suy ra tổng câu sai.

Bảy ví dụ được đọc từ validation thật để thảo luận ranh giới cảm xúc gần nghĩa, bỏ sót nhãn và diễn đạt hàm ý/neutral. Đã đối chiếu văn bản, nhãn và đủ 28 scores với dữ liệu gốc. Nhận xét nội dung là diễn giải thủ công, không phải bằng chứng xác định nguyên nhân lỗi.

Kiểm app/CLI đạt sai số cho phép `1e-6`; xử lý được input rỗng, input dài, nhiều nhãn và không nhãn vượt ngưỡng. Đã dừng/mở lại server và kiểm HTTP thành công. Kiểm widget dùng Streamlit AppTest; chưa có ảnh/video hoặc kiểm trực quan toàn giao diện bằng trình duyệt trong lần này. Bộ kiểm tra mã đã có **24 test đạt**.

## 5. Tài liệu nhóm có thể xem ngay

- [Notebook C3 có output](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/notebooks/distilbert_huy.ipynb).
- [Báo cáo full ba seed, bảng metrics và biểu đồ](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/reports/c3_distilbert/full/C3_RESULTS.md).
- [Bảy ví dụ và phân tích lỗi](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/reports/c3_distilbert/full/ERROR_ANALYSIS.md).
- [Hướng dẫn cài đặt, chạy và bàn giao](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/docs/HUY_DISTILBERT.md).
- [Minh chứng demo và kiểm khởi động lại](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/reports/c3_distilbert/full/DEMO_EVIDENCE.md).
- [Phần phương pháp C3](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/reports/PHUONG_PHAP_C3_NHAT_HUY.md).
- [Bản HTML notebook — tải về để mở](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/blob/main/reports/c3_distilbert/distilbert_huy.html).

Code, notebook và báo cáo nằm trong repo. Checkpoint/tokenizer/scores lớn và môi trường ảo không được đưa lên Git. Muốn chạy lại notebook hoặc demo, cần nhận nguyên thư mục run từ Huy theo hướng dẫn hoặc tự huấn luyện lại. Demo hiện chạy local; địa chỉ localhost trên máy Huy không phải link công khai cho thành viên khác.

## 6. Phần còn cần phối hợp với nhóm

| Việc tiếp theo | Đầu vào cần có |
|---|---|
| So sánh C1/C2/C3 | Kết quả BERT và RoBERTa đủ ít nhất ba seed, cùng split và cách tính metrics |
| Đối chiếu lỗi giữa ba kiến trúc | Validation scores có ID, mapping 28 nhãn và thông tin checkpoint/ngưỡng của C1/C2 |
| Chọn và tích hợp demo cuối | Nhóm chọn C thắng theo mean Macro-F1 validation; bàn giao checkpoint, tokenizer, ngưỡng rồi kiểm lại app/CLI |
| Đánh giá test cuối | Nhóm thống nhất và khóa danh sách cấu hình, checkpoint, ngưỡng, protocol trước khi chạy |

Hiện chưa cần chạy thêm thí nghiệm C3 ngoài phạm vi đã thống nhất. Phần độc lập của Huy sẵn sàng bàn giao; phần so sánh liên mô hình và demo cuối chờ đầu vào của nhóm.


## 7. Trạng thái bàn giao lên GitHub

Đã push code, notebook và báo cáo C3 lên nhánh **main** của repo nhóm ngày 08/10/2026. Commit: [1fc83ad](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/commit/1fc83adadd11e9eae3e46bb8a703c12a4e0c9978).

File cập nhật này được lưu tại `docs/CAP_NHAT_NHAT_HUY_2026-10-08.md` trong repo để nhóm theo dõi và dùng khi cập nhật Notion. Checkpoint và môi trường ảo vẫn được giữ cục bộ.
