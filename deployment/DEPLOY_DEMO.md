# Đưa demo lên Hugging Face Spaces

Gói sẵn sàng nằm trong `hf_space/`: app.py, README.md, requirements.txt, model_manifest.json.
Không chứa trọng số trong Git thường; app tải checkpoint BERT/123 đã công bố từ GitHub Release và kiểm SHA-256. Không train, không chọn ngưỡng.

1. Tự đăng nhập Hugging Face, chọn New Space, đặt tên và chọn Gradio SDK.
2. Chọn CPU Basic nếu giao diện còn cung cấp lựa chọn miễn phí; kiểm lại điều kiện tài khoản. Không cần GPU cho demo một câu.
3. Upload toàn bộ bốn file trong `hf_space/` vào thư mục gốc Space. Chủ báo cáo thực hiện bước công bố.
4. Đợi build và lần tải checkpoint đầu (~435 MB ZIP, cần thêm dung lượng giải nén/cache). Các lần khởi động lại có thể tải lại khi cache tạm mất.
5. Mở App. Mặc định BERT seed 123 và Ngưỡng riêng. Thử câu “I was worried, but your help made me feel so relieved.”: nervousness/relief; chuyển 0,5 phải rỗng. Kiểm đủ 28 hàng điểm.
6. Thử ô trống và chuỗi quá dài; kiểm cảnh báo cắt 128 token. Thử năm câu gợi ý trong app.
7. Sao chép link dạng https://huggingface.co/spaces/<tai-khoan>/<ten-space> vào mục 5.6 và A.1 của báo cáo; xóa ô TODO tương ứng sau khi kiểm link bằng cửa sổ chưa đăng nhập.
8. Chụp ảnh vùng nhập liệu, chế độ ngưỡng và kết quả để thay ô ảnh ở 5.6. Không dùng ảnh lịch sử có mặc định khác để gọi là bản mới.

Nếu tải checkpoint lỗi: mở logs, kiểm GitHub Release còn công khai và URL trong model_manifest.json, kiểm dung lượng. App từ chối ZIP sai SHA-256; không sửa hash để bỏ qua lỗi. Không cần đưa token vào source. Các mô hình khác có trong demo local của repo, không nằm trong Space tối thiểu.

Kiểm offline: đặt biến môi trường GOEMOTIONS_CHECKPOINT tới thư mục checkpoint BERT/123 đã khôi phục, chạy `python app.py`. App vẫn kiểm từng file trong manifest. Chưa có link Space được tạo hoặc deploy tự động trong lượt sửa này.

Tài liệu chính thức: https://huggingface.co/docs/hub/spaces-sdks-gradio và https://huggingface.co/docs/hub/spaces-overview .
