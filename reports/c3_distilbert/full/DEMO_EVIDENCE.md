# Minh chứng demo C3 — Nhật Huy

## Truy cập và checkpoint

Địa chỉ trên máy đang chạy: **http://127.0.0.1:8501**. Đây là link local, không phải bản
triển khai công khai; thành viên khác cần nhận checkpoint và chạy trên máy của mình.
Server được để chạy sau lần kiểm cuối ngày 08/10/2026; nếu đã tắt máy, mở lại bằng:

```powershell
.\.venv-huy\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Chạy từ thư mục gốc repo. App mặc định chọn **C3 DistilBERT seed 123, best epoch 3**,
từ `data/processed/c3_distilbert/full/seed_123/`, theo `representative_c3.json`.
Không coi đây là demo mô hình thắng C1/C2/C3. Suy luận mặc định CPU, 28 scores sigmoid,
ngưỡng 0,5 hoặc ngưỡng riêng gắn đúng checkpoint; văn bản tiếng Anh.

## Khởi động, dừng và khởi động lại thật

| Lượt | UTC | PID server | GET / | GET /_stcore/health |
|---|---|---:|---:|---|
| Lần đầu | 2026-10-08T13:45:04.2280314Z | 29836 | 200 | 200, ok |
| Sau khởi động lại | 2026-10-08T13:45:57.0359056Z | 19968 | 200 | 200, ok |

Đã dừng đúng tiến trình server do lượt kiểm này tạo, kiểm cổng không còn listener rồi
khởi động tiến trình mới. File gốc: [demo_first_health.json](demo_first_health.json),
[demo_restart.json](demo_restart.json), [demo_restart_health.json](demo_restart_health.json).
HTTP 200 chứng minh server phục vụ được trang và health endpoint; riêng nó không chứng minh
đã thao tác dự đoán trong trình duyệt. Các kiểm dự đoán dưới đây dùng Streamlit AppTest.

## Kiểm dự đoán và input biên

Chạy lại `scripts.verify_distilbert` trong tiến trình mới sau khi server được khởi động lại.
Thời điểm: **2026-10-08T13:46:49.140609+00:00**. Kết quả **passed**.
AppTest thực thi `app.py`, nhập vào widget, bấm dự đoán và đọc đầu ra thực tế; CLI chạy
trong subprocess riêng. AppTest không kết nối vào phiên trình duyệt của server ở trên.

| Kiểm tra | Kết quả thật |
|---|---|
| Hash checkpoint, mapping và scores theo ID | Passed |
| Tính lại metrics validation | Passed |
| CLI so với hàm suy luận chung | Max abs diff 2.38418579e-07, trong atol 1e-06 |
| AppTest so với hàm suy luận chung | Max abs diff 0.0 |
| App/CLI với ngưỡng riêng | passed; app max abs diff 0.0 |
| Input rỗng | Hiển thị xử lý input rỗng, không dự đoán |
| Input dài | Mẫu train `edwy61c`, 316 token, có thông báo truncation |
| Đầu ra nhiều nhãn | Passed, mẫu validation `eczdvun` |
| Không nhãn vượt ngưỡng | Passed trên câu validation thật, không ép neutral |
| Ngưỡng sai seed/checkpoint | Bị từ chối trong kiểm tích hợp |

Mẫu đầu kiểm parity: validation `edgurhb`. Tất cả câu không rỗng dùng để kiểm
đều lấy từ train/validation thật. Không đánh giá test, không thêm câu/số liệu nhân tạo.
Chi tiết: [verification.json](verification.json); mã kiểm:
[scripts/verify_distilbert.py](../../../scripts/verify_distilbert.py).

Không có ảnh chụp/video trình duyệt trong lần kiểm này. Minh chứng hiện có là link local,
log HTTP và kiểm widget bằng AppTest; chưa có kiểm trực quan toàn giao diện trong trình duyệt.

## Cách nhóm thử và bàn giao

1. Mở link local, giữ checkpoint mặc định và đọc trạng thái C3.
2. Dùng câu mặc định nguyên văn validation, bấm dự đoán; xem nhãn cùng bảng đủ 28 scores.
3. Chuyển chế độ ngưỡng riêng: scores giữ nguyên, tập nhãn có thể thay đổi.
4. Khi clone repo, cần nhận nguyên thư mục run từ Huy hoặc chạy lại full: checkpoint và
   dữ liệu trong `data/processed` không được Git đưa lên repo.
5. Khi nhóm chọn C thắng, tích hợp checkpoint/tokenizer/ngưỡng đúng mô hình và kiểm parity lại.

Hướng dẫn đầy đủ: [HUY_DISTILBERT.md](../../../docs/HUY_DISTILBERT.md).
