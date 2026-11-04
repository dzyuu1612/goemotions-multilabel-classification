# Quyết định cập nhật kết quả chính của ba mô hình C

Thời điểm ghi nhận: 2026-11-03T10:51:49.642009+07:00

Duy yêu cầu áp dụng epoch tốt nhất cho cả C1, C2, C3 và vẽ lại biểu đồ, cập nhật trực tiếp báo cáo chính. Khi làm rõ phạm vi so với MD trước, Duy xác nhận: “Cập nhật cả kết quả test của 3C”.

## Quy trình khóa trước đánh giá test

- Dùng đủ ba seed 42, 123, 2026 của từng kiến trúc; không chọn hoặc bỏ seed theo test.
- Mỗi lượt có ngân sách tám epoch, learning rate 2e-5 và các cài đặt đã ghi trong kế hoạch. C1 dùng kết quả tám epoch đã hoàn tất; C2/C3 đang chạy theo DECISION_SWEEP.md.
- Checkpoint của từng seed được chọn bằng Macro-F1 validation tại ngưỡng 0,5 cao nhất; hòa chọn epoch sớm hơn. Không chọn lại checkpoint bằng BCE hoặc số test.
- Sau khi tất cả lượt hoàn tất, kiểm hash và CSV; khóa cả chín checkpoint trước khi nạp test.
- Với từng checkpoint, chọn ngưỡng bằng validation theo đúng thuật toán hiện có: fixed 0,5; một ngưỡng chung và ngưỡng riêng từng nhãn trên lưới 0,05..0,95. Khi hòa ưu tiên ngưỡng gần 0,5 nhất, sau đó ngưỡng lớn hơn. Báo cáo chính so sánh fixed và ngưỡng riêng; không chọn chế độ theo test.
- Chọn kiến trúc đại diện bằng mean Macro-F1 validation @0,5; hòa ưu tiên std mẫu nhỏ hơn, ít tham số hơn, tên. Chọn seed đại diện trong từng kiến trúc bằng validation @0,5 cao nhất; hòa ưu tiên seed nhỏ hơn.
- Khóa toàn bộ protocol, lựa chọn và hash trong main_results/LOCK_BEFORE_TEST.json trước mọi suy luận test. Test chỉ tính metric và phân tích lỗi ở checkpoint đã khóa, không quay lại đổi epoch, LR, seed hoặc ngưỡng.
- Giữ A/B như cũ. Đánh giá lại test của đủ chín checkpoint C, dùng cùng official test 5.427 câu và thứ tự 28 nhãn. Từ một bộ điểm test của mỗi checkpoint tính các chế độ ngưỡng đã khóa.
- Cập nhật toàn bộ bảng, hình, phân tích lỗi, cấu hình, kết luận và kịch bản liên quan trong báo cáo/PPT chính. Các đường học vẫn hiển thị đủ tám epoch và đánh dấu epoch chọn của từng seed.

## Phân biệt hai giai đoạn

Đợt quét theo DECISION_SWEEP.md chỉ sử dụng train/validation và không bị thay đổi hồi tố. Đây là quyết định tiếp theo, do yêu cầu mới của Duy, cho giai đoạn đánh giá test sau khi chọn cấu hình. Log và checkpoint huấn luyện được giữ nguyên, kết quả đánh giá mới lưu riêng. Không đổi luật chẩn đoán (a)/(b)/(c).

Test này đã từng được quan sát ở phiên bản báo cáo trước, vì vậy không gọi là holdout mới chưa quan sát. Không bảo đảm kết quả tăng. Các số đo tốt hoặc xấu đều phải được giữ và báo cáo. Không dùng kết quả này để tinh chỉnh tiếp; bước tinh chỉnh khác cần một quyết định mới.
