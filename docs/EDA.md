# Hướng dẫn thống kê dữ liệu GoEmotions

Phần đã thực hiện: **thống kê dữ liệu đã khám phá (EDA)**, dựa trên file
`KE_HOACH_A_Z_DE_TAI_01_GOEMOTIONS.docx`. Notebook giải thích từng bước bằng tiếng Việt,
có output thật, kiểm chứng số liệu và xuất các bảng/hình để dùng trong báo cáo nhóm.

## Mở kết quả

- [Notebook 16 bước có kết quả chạy sẵn](../notebooks/eda.ipynb).
- [Báo cáo thống kê chi tiết](../reports/THONG_KE_DU_LIEU.md).
- [Bản HTML của notebook](../reports/eda.html): tải file về và mở bằng trình duyệt; đã kèm kết quả chạy sẵn trong repo.
- [Phân bố đầy đủ 28 nhãn](../reports/tables/label_distribution.csv).
- [30 ví dụ có ID và nhãn gốc](../reports/tables/examples_30.csv).
- [Các biểu đồ PNG](../reports/figures/).

Notebook gồm: nguồn và hash dữ liệu → mapping nhãn → chất lượng/số mẫu → multi-hot →
phân bố nhãn → năm nhãn hiếm → số nhãn mỗi mẫu → neutral/đồng xuất hiện → trùng lặp →
độ dài ký tự/từ → độ dài ba tokenizer → ví dụ → đối chiếu kế hoạch → xuất báo cáo.

## Chạy trên Windows

Đã kiểm tra với **Python 3.12.6**, CPU. Dùng môi trường riêng để tránh xung đột thư viện
với các môn học/dự án khác. Tại thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_eda.py
```

Lệnh cuối chạy toàn bộ notebook, lưu output vào `notebooks/eda.ipynb`, xuất `reports/eda.html`,
báo cáo Markdown, bảng CSV, biểu đồ PNG, manifest và ma trận multi-hot.
Một cell lỗi sẽ dừng, không báo thành công giả. Thời gian phụ thuộc CPU và mạng;
lần đầu cần Internet để tải ba tệp Parquet và ba tokenizer.json. Không cần GPU hoặc tài khoản Hugging Face.

Để chạy từng ô và theo dõi:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name goemotions-eda --display-name "Python (GoEmotions EDA)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Mở `notebooks/eda.ipynb`, chọn kernel **Python (GoEmotions EDA)**, chạy lần lượt bằng
`Shift+Enter`, hoặc **Restart Kernel and Run All Cells**. Trên VS Code có thể chọn trực tiếp
interpreter `.venv\Scripts\python.exe` trong mục Select Kernel.

Trên Linux/macOS thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.
Trên Colab cần tải/clone **toàn bộ repo**, chuyển cwd đến gốc repo rồi dùng
`%pip install -r requirements.txt`. Notebook không có đường dẫn máy cá nhân cố định.

## Dữ liệu và quy ước

- Repository nguồn: [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions).
- Cấu hình `simplified`, revision `add492243ff905527e67aeb8b80c082af02207c3`.
- 43.410 train, 5.426 validation, 5.427 test; tổng 54.263 bình luận tiếng Anh.
- 28 nhãn theo thứ tự **metadata của Parquet**; 27 cảm xúc + neutral.
- Giữ nguyên split, text và nhãn gốc. Support đếm số mẫu mang nhãn, tổng có thể vượt số dòng.
- Neutral có thể đi cùng nhãn khác. Nhãn hiếm xác định từ train; token length đo trên train/validation.
- EDA test chỉ là thống kê mô tả. Không dùng test để chọn mô hình, ngưỡng hoặc max_length.
- PDF SharePoint gốc chưa đọc trực tiếp được; tiêu chí giảng viên được dẫn lại qua file kế hoạch người dùng cung cấp.

## Cấu trúc bàn giao

```text
notebooks/eda.ipynb                 Notebook chính có code, giải thích và output
src/data.py                        Tải/cache, SHA-256, schema, mapping, multi-hot
data/manifest.json                 Nguồn, revision, hash, số dòng
data/labels.json                   Danh sách 28 nhãn theo thứ tự chuẩn
data/tokenizer_revisions.json      Commit cố định của ba tokenizer
data/tokenizer_manifest.json       URL, revision và hash tokenizer đã đo
data/raw/                         Cache Parquet (không đưa vào Git)
data/tokenizers/                   Cache tokenizer (không đưa vào Git)
data/processed/*_multihot.npz       Y, ids, label_names theo thứ tự gốc
references/example_notes.json     Nhận xét đọc 30 ví dụ, không thay ground truth
reports/THONG_KE_DU_LIEU.md         Nội dung thống kê để dùng trong báo cáo
reports/eda.html                   Bản đọc bằng trình duyệt
reports/tables/                    CSV UTF-8 BOM, đầy đủ số liệu
reports/figures/                   Bảy biểu đồ PNG
reports/environment.json          Môi trường chạy thực tế
reports/verification.json         Trạng thái đối chiếu các mốc dữ liệu
reports/summary.json               Các số liệu chính dạng máy đọc được
scripts/run_eda.py                 Chạy notebook và xuất HTML
scripts/build_notebook.py          Nguồn tạo notebook (dành cho sửa cấu trúc)
```

Chạy lại sẽ ghi đè **output do EDA tạo**. Nhóm ghi nhận xét vào bản sao
`examples_30.csv` để giữ phần làm thủ công. `scripts/build_notebook.py` tạo lại notebook
và xóa output cũ; chỉ dùng khi sửa cấu trúc notebook, sau đó chạy `scripts/run_eda.py`.

## Kiểm chứng và giới hạn

Notebook kiểm hash cả ba file; schema và thứ tự nhãn; support của 28 nhãn × 3 split;
số mẫu theo số nhãn; neutral đồng xuất hiện; ID/text trùng; shape và phép đọc lại multi-hot;
30 ví dụ không trùng ID và phủ đủ 28 nhãn. Các mốc kiểm chứng lấy từ file kế hoạch,
còn các kết quả được tính lại từ dữ liệu thực tế.

30 ví dụ được chọn có chủ đích để phủ nhãn, không đại diện ngẫu nhiên cho toàn bộ dữ liệu.
Nhận xét là diễn giải của người chuẩn bị, nhóm cần thảo luận thêm các trường hợp mơ hồ.
Đếm trùng nguyên văn không phát hiện mọi dạng rò rỉ; test phụ 5.390 mẫu chỉ loại text trùng train.
Thống kê tokenizer không thay cho thử nghiệm F1. Chưa triển khai baseline, zero-shot,
fine-tune hoặc demo trong phần bàn giao này.

## Tài liệu

- File kế hoạch người dùng cung cấp: mục 4, 5, 13.1, 15.1 và Phụ lục A.
- [Bài báo GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/).
- [Google Research README](https://github.com/google-research/google-research/blob/master/goemotions/README.md).
- [Dataset card và giấy phép Apache 2.0](https://huggingface.co/datasets/google-research-datasets/go_emotions).
