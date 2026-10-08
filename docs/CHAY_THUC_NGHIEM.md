# Chạy thí nghiệm và giữ bằng chứng

## 1. Tạo môi trường riêng

Chạy từ gốc repo bằng PowerShell. Dùng Python 3.13 như môi trường đã đo; tạo môi
trường riêng để tránh trộn thư viện Anaconda. Không bật `--system-site-packages`.

```powershell
python -m venv .venv-models
$env:PYTHONUTF8='1'
& '.\.venv-models\Scripts\python.exe' -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
& '.\.venv-models\Scripts\python.exe' -m pip install -r requirements-models-lock.txt
```

Lệnh CUDA trên dùng cho máy NVIDIA tương thích, như RTX 5060 đã kiểm. Máy CPU
cài bản PyTorch CPU từ nguồn chính thức và dùng `--device cpu`. GPU ảnh hưởng thời
gian chạy; đầu ra có thể khác nhẹ vì số học dấu phẩy động. Môi trường thực tế được
lưu trong `reports/environment_models.json`; cấu hình từng lần chạy nằm trong metadata.

## 2. Dữ liệu và A

Giữ split chính thức. A đã có code; trên máy mới cần tạo lại model/scores lớn:

```powershell
& '.\.venv-models\Scripts\python.exe' -m scripts.run_baseline
& '.\.venv-models\Scripts\python.exe' -m scripts.run_baseline --variant balanced
& '.\.venv-models\Scripts\python.exe' -m scripts.analyze_baseline
```

Kiểm `--help` nếu tham số của script đổi. Dữ liệu/model/venv được bỏ qua bởi Git
để repo gọn; các bảng nhỏ, metadata báo cáo và log thực nghiệm được giữ lại.

## 3. Kiểm nhanh trước chạy đủ

```powershell
& '.\.venv-models\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv-models\Scripts\python.exe' -m scripts.train_transformer --architecture distilbert --seed 42 --smoke --device cuda
& '.\.venv-models\Scripts\python.exe' -m scripts.run_zero_shot --smoke --device cuda --dtype float16 --batch-size 16
```

Smoke nằm ở thư mục riêng, không được dùng làm số cuối hoặc để chọn demo.
Torch kiểm head nhỏ/backward trong unit tests không phải thí nghiệm GoEmotions.

## 4. Toàn bộ bảng thực nghiệm

```powershell
& '.\.venv-models\Scripts\python.exe' -m scripts.complete_project --device cuda
```

Thứ tự: ba C × ba seed → B validation → chọn best C theo **mean validation
Macro-F1@0,5** → khóa ba ngưỡng của mọi B/C và sáu cấu hình A → đánh giá test →
gom bảng → đối chiếu ba nhóm lỗi. Một tác vụ dùng GPU tại một thời điểm.

C có 3 seed 42/123/2026. C1 BERT cased dùng lr 5e-5, tối đa 4 epochs;
C2/C3 dùng lr 2e-5, tối đa 3 epochs. Cùng max_length=128, batch16 và accumulation1;
padding chỉ đến câu dài nhất trong mỗi batch, giữ nguyên token hợp lệ;
checkpoint tốt nhất mỗi seed chọn theo validation Macro-F1@0,5. AMP dùng trên CUDA.
B dùng float16 trên CUDA và float32 trên CPU; dtype được ghi và không đổi khi resume.

`--resume` bỏ qua C hoàn tất có hash/config đúng. Run C dở chưa có optimizer-state
resume: kiểm nguyên nhân rồi chạy lại đúng run với `--overwrite`. Không xóa các
run đã hoàn tất. B checkpoint từng batch, có thể chạy lại để tiếp phần chưa xong.

Sau khi đã đủ C/B validation, có thể dùng `--skip-training` để tiếp các bước sau.
Nếu đã mở test, giữ model/protocol đã khóa. Không sửa tham số rồi dùng cùng test
để chọn lại. Bản EDA đã khảo sát các split; nguyên tắc ở đây là **không dùng test
để fit/chọn cấu hình/ngưỡng**, không tuyên bố chưa từng nhìn dữ liệu test.

## 5. Đọc sản phẩm

- `reports/execution/full_pipeline.log`: lệnh, thời điểm và output thật.
- `reports/execution/frozen_protocols.json`: hash của các protocol trước test.
- `reports/project_results/all_runs.csv`: từng seed, A/B riêng, đủ chỉ số.
- `reports/project_results/mean_std.csv`: C đủ ba seed, sample std ddof=1.
- `reports/project_results/RESULTS.md`: bảng dễ đọc, liệt kê phần thiếu.
- `reports/errors_test_standard_fixed/`: ba nhóm lỗi, cùng ID cho C1/C2/C3.
- `data/processed/transformers/selected_model.json`: kiến trúc/seed thắng trên val.

Tuned-validation đã dùng chọn ngưỡng nên thường lạc quan. Dùng test sau khóa để
kết luận cải tiến; giữ cả nhãn tăng và giảm. A/B chỉ một cấu hình trọng số đã chạy
thì không ghi seed std bằng 0 như thể đã đo nhiều run.

## 6. Mở demo

```powershell
& '.\.venv-models\Scripts\python.exe' app.py --device cuda
```

Mặc định ngưỡng0,5. Muốn dùng ngưỡng riêng đã khóa, truyền `--thresholds` đường dẫn
`final_protocol.json` của đúng run trong selected_model. App kiểm mapping và hash.
Nhập bình luận **tiếng Anh**; chưa có kiểm chứng cho tiếng Việt hay dữ liệu doanh nghiệp.
Demo hiển thị đủ28điểm, ngưỡng và nhãn chọn; không có nhãn vượt ngưỡng không tự gán neutral.
