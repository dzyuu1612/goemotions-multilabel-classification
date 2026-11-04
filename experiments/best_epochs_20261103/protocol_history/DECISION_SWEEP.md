# Quyết định quét epoch C2/C3

Ghi trước khi chạy vào 2026-11-03T08:27:54.293363+07:00.

Mục đích là kiểm tra giả thuyết C2/C3 học chưa đủ ở cấu hình ba epoch, dựa trên đường validation.

RoBERTa-base và DistilBERT-base-uncased, ba seed 42, 123, 2026, mỗi seed một lượt. Learning rate 2e-5, 8 epoch, batch 16, accumulation 1, max_length 128, AdamW, weight decay 0,01, warmup ratio 0,1, max grad norm 1, AMP fp16 và lịch tuyến tính. Tokenizer và checkpoint nền khóa cùng revision với kết quả cũ. Đánh giá toàn bộ train ở eval mode mỗi epoch. Không dùng tập con.

Mỗi lượt có 2.714 batch/epoch, 21.712 bước lịch dự kiến và 2.171 bước warmup lấy phần nguyên. Đây là số bước kế hoạch, AMP có thể bỏ qua bước cập nhật khi overflow. Chương trình kiểm các số kế hoạch trước batch huấn luyện đầu tiên.

Không nạp hoặc chấm test. Không tính metric hay lưu điểm test. Chọn checkpoint bằng Macro-F1 validation @0,5 cao nhất; hòa điểm giữ epoch sớm hơn. Sáu lượt chạy đủ tám epoch để quan sát đường học, không dừng sớm. Không loại seed kém hoặc thay seed.

## Luật kết luận cố định

Delta_seed = Macro-F1 validation ở checkpoint được chọn của lịch 8 epoch trừ Macro-F1 validation tại checkpoint được chọn trong lần chạy gốc cùng mô hình và seed.

(a) Mọi Delta_seed > 0 và Delta trung bình >= 0,01. Kết luận có dấu hiệu chưa học đủ ở cấu hình 3 epoch.

(b) Delta trung bình thuộc [-0,01; 0,01] hoặc dấu không nhất quán. Kết luận không có bằng chứng chưa học đủ.

(c) Delta trung bình < -0,01. Kết luận chạy dài hơn không giúp.

Áp dụng theo thứ tự (a), rồi (b), rồi (c) như MD. Nếu dấu không nhất quán đồng thời Delta trung bình < -0,01, mục (b) được áp dụng trước và nhận xét phải ghi đúng dấu không nhất quán. Ngưỡng 0,01 là quy ước của nhóm, không phải kiểm định thống kê.

## Biên giới diễn giải

Giữ cấu hình C2/C3 ngoài epoch. Thay tổng epoch cũng làm thay đổi lịch learning rate và số bước warmup. Epoch 3 trong lịch 8 epoch khác một lượt chỉ chạy 3 epoch.

Gộp BERT 8 epoch đã hoàn tất từ GoEmotions_ThuNghiem_Epoch_20261102. BERT này dùng learning rate 2e-5, 21.712 bước lịch và 2.171 bước warmup dự kiến; đã xác nhận chỉ train/validation. BERT gốc dùng 5e-5 và 4 epoch, do đó phép so sánh BERT thay cả learning rate lẫn lịch huấn luyện. Không gọi riêng BERT là đối chứng chỉ thay epoch.

Giữ bộ test, ngưỡng riêng, checkpoint và demo của kết quả chính. Chỉ thêm mục chẩn đoán ở bản sao báo cáo Word, slide và kịch bản khi sáu lượt hoàn tất. Chưa kết luận thứ hạng kiến trúc từ thí nghiệm phụ.

## Lỗi kỹ thuật

Ghi nhận lỗi trước khi thử lại đúng cấu hình và seed, giữ log và lần chạy lỗi trong thư mục riêng. Không xóa hoặc thay số kém. Bản sao bộ kết quả cũ và sáu checkpoint nền báo cáo đã lưu trong backup_original với SHA-256.
