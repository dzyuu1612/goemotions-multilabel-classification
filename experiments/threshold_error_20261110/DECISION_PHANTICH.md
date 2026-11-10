# Quyết định phân tích ngưỡng và lỗi từ điểm dự đoán đã lưu

Ngày ghi: 10/11/2026, múi giờ Asia/Bangkok. Người phụ trách: Nguyễn Bảo Duy.

Đây là kế hoạch phân tích sau nhận xét của giảng viên, được ghi nhận trước khi chạy các bảng test mới của đợt phân tích này. Test đã được xem ở các báo cáo trước; không gọi đây là đăng ký trước toàn bộ nghiên cứu hay đánh giá trên holdout mới.

- Không huấn luyện, không đổi checkpoint, learning rate, epoch hay seed. Giữ chín checkpoint chính của lịch tám epoch, seed 42/123/2026. A và B giữ đúng một lượt đã lưu; không tạo thêm seed hoặc std giả.
- Luật chọn checkpoint lịch sử: Macro-F1 validation ở ngưỡng 0,5 cao nhất; hòa lấy epoch sớm hơn. Kế hoạch này không thay đổi các lựa chọn lịch sử.
- Thứ tự: chỉ mở validation và ngưỡng đã lưu để kiểm toán, ghi tệp khóa có hash; commit tệp khóa rồi mới mở điểm test cho các bảng B1–B6. Không chọn chế độ tốt nhất bằng test; báo cáo đủ ba chế độ và mọi kết quả giảm.
- Lưới ngưỡng: 19 giá trị từ 0,05 đến 0,95. Tối đa Macro-F1 validation cho ngưỡng chung; tối đa F1 validation từng nhãn cho ngưỡng riêng. Hòa: gần 0,5 rồi lớn hơn. So sánh ngưỡng tính lại với ngưỡng đã lưu; nếu lệch, dừng trước test, ghi lý do và không tự thay ngưỡng báo cáo.
- Ngưỡng chung đã được đề cập trong DECISION_MAIN_3C ngày 03/11/2026; lần này kiểm chứng dữ liệu và công bố phép đối chiếu, không ghi lùi thời gian quyết định mới.
- B3 phân hoạch năm loại theo đúng script người dùng: đúng hoàn toàn; không đoán gì; không có nhãn nào trùng (đoán sai nhãn); giữ đủ nhãn thật nhưng thêm nhãn (đoán thừa); còn lại có nhãn đúng nhưng thiếu nhãn (đúng một phần, có thể kèm FP). Thứ tự ưu tiên làm năm loại loại trừ nhau, tổng bằng 5.427. Đồng thời giữ phép đếm FN/FP chuẩn.
- B4 giữ ba nhóm cũ có thể chồng nhau, công bố các giao và bảy miền loại trừ của ba tập. Không cộng các tỷ lệ khác mẫu số.
- B5 chọn cặp relief→admiration và neutral→approval từ yêu cầu, lấy tối đa 20 câu/cặp bằng RNG seed 20261107. Phiếu gán nhãn độc lập dành cho Duy và Huy; không điền thay người, không tạo tỷ lệ đồng thuận khi chưa có hai phiếu thật. Phân tích cặp chỉ mô tả đồng xuất hiện FN/FP, không chứng minh nhân quả.
- B6 giữ quy tắc định trước: FN có điểm ≥0,8×ngưỡng; FP vượt ngưỡng ≤0,1. Đây là mô tả lỗi, không dùng để sửa ngưỡng sau test.
- B7 đọc đầy đủ tám epoch của chín lượt từ sweep_log; chỉ dùng validation để bàn lựa chọn epoch. Quy ước chẩn đoán ±0,01 của hướng dẫn được áp dụng hồi cứu: tăng ≥0,01 ở cả ba seed là dấu hiệu phù hợp giả thuyết học chưa đủ ở lịch cũ, không phải kiểm định hay bằng chứng nhân quả. BERT cũ còn khác learning rate nên không cô lập ảnh hưởng của epoch.
- Tỷ trọng công sức Bảng B.1 để nhóm tự điền theo chỉ dẫn trực tiếp trước đây. Không tự đặt tỷ lệ.
- Giữ các hạn chế: test đã được quan sát; 37 dòng test trùng văn bản train; std mẫu không phải khoảng tin cậy; các nguyên nhân chưa có ablation; kết quả gán nhãn độc lập chưa có.

Nguồn yêu cầu: HUONG_DAN_LAM_THEO_NHAN_XET_CO.md và phan_tich_nguong_loi.py do người dùng cung cấp ngày 10/11/2026.
