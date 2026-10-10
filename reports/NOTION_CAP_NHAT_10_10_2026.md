# Notion — cập nhật nội dung khoa học ngày 10/10/2026

Đã sửa các đoạn hiện có của kế hoạch nhóm và trang phần việc Duy; đọc lại cả hai trang xác nhận thay đổi. Phân công bốn thành viên và các trang con vẫn được giữ. Bản này ghi các đoạn đã cập nhật, không phải bản sao toàn bộ workspace.

## 📚 Đồ án NLP — GoEmotions | Kế hoạch & phân công nhóm

[Trang Notion](https://app.notion.com/p/3ed7c27769028185af2dfbaac4c4586b?pvs=204)

- Lần sửa cuối từ Notion: 2026-10-10T08:21:44.118Z.
- Thao tác cập nhật: `task_f901502b61e0435daeb7b4ba6b951dd0`, trạng thái `succeeded`.
- Đã đọc lại và kiểm date10, roi, threshold, search, duy_child, huy_child.

### Nội dung các đoạn đã sửa

Cập nhật <mention-date start="2026-10-10"/>. Nhóm có đúng **4 thành viên: Bảo Duy Nguyễn, Quốc Khánh, Đức Trí, Nhật Huy**, chia theo đầu việc. A/B/C/D có đủ chín run C và test với protocol đã khóa; 77/77 kiểm mã sau ghép được ghi nhận ngày 09/10. Hai báo cáo cuối Word/PDF đã bổ sung giá trị ứng dụng/ROI chưa đo, validation tuned/test locked và căn cứ siêu tham số. [Bản cập nhật nội dung](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/CAP_NHAT_NOI_DUNG_10_10_2026.md) · [Kiểm Word/PDF](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/execution/final_report_verification.json). Hồ sơ riêng Duy ở trang con; các bảng có ngày cũ là lịch sử, đọc bản bàn giao GitHub làm căn cứ mới.

- **Cấu hình đã chạy:** max_length 128, batch 16; C1 BERT learning rate 5e-5, tối đa 4 epoch, tham khảo mục 5.3 GoEmotions; C2/C3 learning rate 2e-5, tối đa 3 epoch, là cấu hình nhóm. Ba seed mỗi kiến trúc kiểm biến động kết quả, không thay thế tìm kiếm siêu tham số. Chưa có hồ sơ để quy C2 cho gradient explosion hoặc kết luận C3 hội tụ nhanh hơn. Learning rate/epoch khác nhau là giới hạn khi cô lập tác động kiến trúc.

- **Ba hướng ứng dụng dự kiến:** hỗ trợ định tuyến/ưu tiên khách hàng bằng cảm xúc kết hợp loại yêu cầu và luật nghiệp vụ; theo dõi tín hiệu khủng hoảng thương hiệu bằng tổng hợp theo thương hiệu/thời gian; giảm công rà soát bình luận nhờ gợi ý nhãn và người đọc kiểm. Đo chuyển đúng/sai, thời gian phản hồi/giải quyết, cảnh báo/bỏ sót, công đọc và chi phí chạy/tích hợp/sửa dự đoán sai. MTTR là mean time to resolution. Đây là hướng phát triển chưa cài/đo tại doanh nghiệp; không quy số tiết kiệm năng lượng Lee thành ROI NLP.

- **Validation/test và đánh đổi:** validation tuned dùng tập đã chọn ngưỡng nên có thể lạc quan; test locked dùng ngưỡng/checkpoint khóa trước test. BERT test: Micro-Recall 0.5322→0.6437, Precision 0.6421→0.5446, Hamming 0.0318→0.0373. Đây là tổng hợp 28 nhãn, chưa quy riêng cho năm nhãn hiếm. A balanced là đối chứng: Precision 0.4043→0.4561, Hamming 0.0547→0.0467. Không kết luận tuning luôn tăng Recall hoặc cả năm nhãn hiếm đều cải thiện; không chọn lại ngưỡng bằng test.

## 📂 Phần của Duy — Baseline, tài liệu & báo cáo

[Trang Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d?pvs=204)

- Lần sửa cuối từ Notion: 2026-10-10T08:21:54.651Z.
- Thao tác cập nhật: `task_468bfa3bafa44716a783baa4760ec5b4`, trạng thái `succeeded`.
- Đã đọc lại và kiểm date10, roi, threshold, search.

### Nội dung các đoạn đã sửa

- **Báo cáo cập nhật 10/10/2026:** hai bản cuối Word/PDF đã bổ sung ba nội dung khoa học: ứng dụng/ROI dự kiến, đánh đổi test và căn cứ siêu tham số; hai báo cáo tiến độ giữ đúng snapshot lịch sử. Đã kiểm nguồn, số liệu và các trang quan trọng; không ghi đã nộp hoặc đã xuất bản IEEE. [Bản ghi chi tiết](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/CAP_NHAT_NOI_DUNG_10_10_2026.md).

- **Cách đọc kết quả:** validation tuned đo trên tập đã quét ngưỡng; test locked đo với checkpoint/ngưỡng đã khóa. BERT test Recall 0.5322→0.6437, Precision 0.6421→0.5446, Hamming 0.0318→0.0373; A balanced tuning lại tăng Precision 0.4043→0.4561 và giảm Hamming 0.0547→0.0467. F1 embarrassment của BERT giảm; grief của RoBERTa/DistilBERT vẫn 0. Không quy toàn bộ đánh đổi 28 nhãn cho riêng năm nhãn hiếm, không chọn lại bằng test. Ba seed C không chứng minh đã tìm learning rate tối ưu; C1 bám thông số bài gốc, C2/C3 là cấu hình nhóm.

- **Ứng dụng:** ba hướng đề xuất gồm hỗ trợ định tuyến khách hàng, theo dõi tín hiệu khủng hoảng thương hiệu và giảm công rà soát bình luận. Bộ phân loại trả 28 điểm/nhãn; tích hợp ticket, luật nghiệp vụ, nhận diện thương hiệu/tổng hợp thời gian và đo ROI là bước phát triển chưa thực hiện. Phải tính công sửa nhãn sai và chi phí model/tích hợp/vận hành; không ghi khoản tiết kiệm năng lượng Case Study 4 của Lee thành ROI GoEmotions.

Hồ sơ máy đọc: `reports/execution/notion_scientific_revision_10_10_2026.json`. Bản Word/PDF và nguồn đối chiếu nằm trong kho đồ án; báo cáo tiến độ giữ trạng thái lịch sử. Các ứng dụng/ROI mới là đề xuất chưa đo trong doanh nghiệp.
