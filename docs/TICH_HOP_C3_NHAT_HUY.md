# Tích hợp phần C3 của Nhật Huy từ kho chung

Cập nhật 09/10/2026. Giữ công việc đã có ở origin/main, commit `3acdfc6`
(gồm `1fc83ad` và bản tiến độ tiếp theo). Đây là nguồn bàn giao của Nhật Huy;
không ghi toàn bộ phần này là công của Duy hoặc tự tính phần trăm đóng góp.

## Hai bộ thực nghiệm được đọc riêng

| Hồ sơ | Phạm vi | Kết quả cần đọc |
|---|---|---|
| Thực nghiệm chung A/B/C/D | Ba C × ba seed, cùng module metric, lựa chọn trên validation và protocol khóa trước test | `reports/project_results/RESULTS.md` |
| Nghiên cứu C3 Nhật Huy | DistilBERT ba seed; trainer/protocol và môi trường ghi riêng, gradient checkpointing | `reports/c3_distilbert/full/C3_RESULTS.md` |

C3 của Huy có Macro-F1 validation @0.5 **0.4061 ± 0.0045**, ba seed.
C3 ở pipeline chung có **0.4064 ± 0.0060** ở cùng metric validation.
Cùng checkpoint/revision và ba epoch không làm hai lượt huấn luyện trở thành
cùng một run: trainer, gradient checkpointing, môi trường và lịch sử khác nhau.
Không cộng hai bộ thành sáu seed hoặc lấy điểm validation của Huy làm điểm test.

Trong bộ so sánh chính, BERT thắng bằng mean Macro-F1 validation @0.5 của
ba seed; checkpoint đại diện BERT seed 123 được dùng cho D.

## Mở đúng demo

- `app.py`: Gradio, C thắng trong bộ so sánh chính; đã kiểm suy luận và giao diện.
- `app_distilbert_huy.py`: giữ nguyên nội dung demo Streamlit của Huy từ kho chung,
  chỉ đổi tên file để hai demo cùng tồn tại. Đọc `docs/HUY_DISTILBERT.md` và
  `requirements-c3.txt` cho môi trường C3 riêng; không cài đè các thư viện này
  vào môi trường đã chạy bộ so sánh chính.
- Verifier C3 và hướng dẫn tạo notebook đã chuyển sang tên demo mới.
  Notebook/HTML/log/ảnh đã xuất của Huy vẫn là bằng chứng lịch sử có thể nhắc
  tên cũ `app.py`; không sửa chúng để tạo ấn tượng đã chạy lại.

Trọng số C3 Huy không nằm trong Git. Muốn suy luận từ bundle này phải nhận
checkpoint/tokenizer đúng checksum từ Huy hoặc tái lập trong đúng môi trường.
Việc tích hợp mã không tự xác nhận demo Streamlit đã chạy lại trên máy hiện tại.

**Kiểm bổ sung trên máy Huy ngày 09/10:** đã chạy `verify_distilbert` bằng bundle C3
full seed 123 với tên app mới; app/CLI, input biên và kiểm dừng/mở lại server đều đạt.
Hồ sơ riêng: [rà soát sau tích hợp](../reports/c3_distilbert/post_merge_20261009/REVIEW.md).
Đây là kiểm C3 trên máy Huy; chưa phải nghiệm thu demo BERT tại máy này vì chưa có
bundle BERT được chọn. Hồ sơ notebook/HTML và verification ngày 08/10 vẫn được giữ nguyên.

## Giữ nguyên nguồn gốc và protocol

Giữ các config/metadata/log của Huy, `.venv-huy/` và `data/cache/` trong ignore,
các quy tắc byte/hash của cả hai bên. Hàm SHA-256 chung tiếp tục đọc từng khối
để không nạp toàn bộ checkpoint vào RAM.

`run_c3_full` kiểm nguồn hiện hành với source SHA của protocol đã khóa.
Nguồn thay đổi sau tích hợp có thể bị từ chối bởi guard này. Không thay hash
lịch sử để vượt guard; dùng snapshot gốc `3acdfc6` nếu cần tái lập nguyên nghiên
cứu của Huy. Kết quả đã ghi vẫn được giữ và đọc riêng.

Kiểm thử tích hợp sau merge được ghi trong
`reports/execution/unit_tests_post_merge.log` và `reports/verification_project.json`.
Các lượt 69 tests của nhóm và 24 tests lịch sử của Huy là hai snapshot;
không cộng chúng thành số kiểm thử mới.
