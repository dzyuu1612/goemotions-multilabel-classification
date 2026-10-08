## Phạm vi

Tiếp tục đồ án GoEmotions theo yêu cầu giảng viên sau khi PR #3 đã được merge ngày 05/10/2026.

- A: TF-IDF + One-vs-Rest Logistic Regression; cấu hình chuẩn/cân bằng lớp, chọn ngưỡng trên validation.
- B: BART-large-MNLI zero-shot, 28 nhãn đúng thứ tự, không fine-tune trên GoEmotions.
- C: BERT, RoBERTa, DistilBERT; mỗi kiến trúc chạy ba seed 42/123/2026 và báo cáo mean ± sample std.
- D: chọn C bằng validation, khóa protocol trước test, dùng checkpoint thật trong demo.
- Module đánh giá chung, phân tích nhãn hiếm, ba nhóm lỗi, kiểm ID/checksum/config và nguồn tham khảo chính thức.
- Hướng dẫn baseline tiếng Việt, notebook theo thứ tự đọc, hồ sơ tái hiện và kế hoạch phân công bốn thành viên.
- Hai dạng báo cáo: sáu chương theo mẫu cô và bài báo IEEE hai cột; Word/PDF và hai báo cáo tiến độ.

## Trạng thái bằng chứng lúc mở PR

Đây là PR nháp trong khi thực nghiệm toàn bộ vẫn đang chạy. A đã có kết quả validation thật. BERT đủ ba seed và RoBERTa seed 42 đã hoàn tất training; các lượt C còn lại, zero-shot toàn bộ và đánh giá test cuối đang tiếp tục. Bảng tổng hợp/Word/PDF hiện là snapshot, có đánh dấu phần chưa đủ bằng chứng. Không dùng smoke thay benchmark hoặc điền kết quả còn thiếu bằng số 0.

Đã có 59 test của lượt kiểm ban đầu, các kiểm tra bổ sung được ghi riêng trong `reports/execution/`. Toàn bộ suite sẽ được chạy lại sau khi thực nghiệm kết thúc. Kiểm demo model và giao diện cũng sẽ ghi bằng chứng sau khi chọn được C từ đủ chín lượt.

## Tệp để đọc

- `docs/HUONG_DAN_DUY_GIAI_THICH_BASELINE.md`
- `docs/THU_TU_DOC_DO_AN.md` và `docs/REPRODUCIBILITY.md`
- `reports/project_results/RESULTS.md`
- `reports/REFERENCE_CLAIM_AUDIT.md`
- `reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.docx` / `.pdf`
- `reports/BAI_BAO_GOEMOTIONS_IEEE.docx` / `.pdf`

Các trường giảng viên, lớp, MSSV và tỷ lệ đóng góp chưa được cung cấp được để trống theo yêu cầu. Trọng số model và dữ liệu nặng không đưa vào Git; metadata đã được kiểm tra giữ nguyên byte/hash qua Git transport.
