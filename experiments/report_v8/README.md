# Kết quả bổ sung trong báo cáo IEEE v8

Chạy `python -m experiments.report_v8.verify_tables` từ thư mục gốc để tính lại lưới ngưỡng approval, số TP/FP/FN, thống kê suy luận và mẫu đọc 50 mục. Notebook chính hiển thị các bảng này.

## Đo suy luận trên Windows

Phép đo ngày 22/11/2026 dùng cùng 512 câu validation, seed lấy mẫu 20261122. A chạy CPU. B chạy CUDA fp16. Ba C chạy CUDA AMP fp16 với checkpoint đại diện seed 2026. Mỗi hệ có 50 lô khởi động và một lượt đo ở mỗi cỡ lô 1 và 16. B xử lý 28 cặp NLI trên mỗi câu, với tối đa 16 cặp mỗi lô nội bộ.

Cài phụ thuộc theo [README chính](../../README.md), tải checkpoint theo mục chạy demo và chạy từ thư mục gốc. Mỗi lần gọi sẽ nạp một hệ và ghi CSV vào `outputs/inference_v8/`.

```powershell
python -m pip install psutil
python -c "from src.paths import ROOT; from src.datasets.goemotions import load_goemotions; load_goemotions(ROOT, write_metadata=False, splits=('validation',))"
foreach ($system in 'baseline_balanced', 'zero_shot', 'bert', 'roberta', 'distilbert') {
    python -m experiments.report_v8.measure_inference $system data/raw/validation-00000-of-00001.parquet
}
```

Các CSV đã công bố phản ánh thiết bị và môi trường ghi trong [biên nhận phép đo](results/inference_setup.json). Số đo mới phụ thuộc thiết bị, driver và tải máy.
