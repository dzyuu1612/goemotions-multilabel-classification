# Nhật Huy — C3 DistilBERT và khung demo

Phần này triển khai công việc độc lập của Huy theo phân công nhóm 04/10/2026:
C3 DistilBERT full ba seed, metrics, ngưỡng riêng, lỗi C3, notebook/báo cáo và demo C3. Không thay thế script huấn luyện
chung mà Khánh sẽ tích hợp, không sửa phần baseline của Duy.

## Trạng thái và đường dẫn

| Hạng mục | Đầu ra |
|---|---|
| Môi trường Python/CUDA riêng | `.venv-huy`, `requirements-c3.txt`, `reports/c3_distilbert/environment.json` |
| Fine-tune C3 | `src/c3.py`, `scripts/train_distilbert.py`, `configs/distilbert_*.json` |
| Notebook từng bước | `notebooks/distilbert_huy.ipynb`; bản đọc `reports/c3_distilbert/distilbert_huy.html` |
| Kết quả full ba seed | `reports/c3_distilbert/full/C3_RESULTS.md` |
| Pilot kỹ thuật trước full | `reports/c3_distilbert/PILOT_RESULTS.md` |
| Checkpoint/tokenizer/scores cục bộ | `data/processed/c3_distilbert/full/seed_<seed>/` (Git bỏ qua) |
| Khung demo | `app_distilbert_huy.py`; CLI `scripts/predict_distilbert.py` |
| Kiểm tra tích hợp | `reports/c3_distilbert/full/verification.json` |
| Phân tích bảy ví dụ lỗi thật | `reports/c3_distilbert/full/ERROR_ANALYSIS.md` |
| Demo và kiểm khởi động lại | `reports/c3_distilbert/full/DEMO_EVIDENCE.md` |
| Rà soát bàn giao cuối | `reports/c3_distilbert/full/final_review.json` |
| Viết phương pháp C3 | `reports/PHUONG_PHAP_C3_NHAT_HUY.md` |

Full dùng `configs/distilbert_full.json`, protocol được ghi trước khi train. Chỉ các run
có `run.json` trạng thái complete, đủ full train/validation và đúng hash mới vào bảng
mean/std. Không dùng test. Demo hiện nạp bundle C3; demo cuối cần kết quả C1/C2 để chọn C thắng.
Cấu hình này áp dụng cho phần C3 của Huy, không tự thay đổi cấu hình của các bạn khác.

Cập nhật 08/10/2026: cả ba seed đã hoàn thành; notebook 33 cell đã chạy hết và có HTML.
Đã đọc/đối chiếu bảy ví dụ lỗi với dữ liệu và scores gốc, rà hai biểu đồ, kiểm app/CLI
và dừng/mở lại server demo. Bản ghi tự động `automatic_checks_complete.json` là trạng thái
trước rà soát thủ công; kết quả rà soát tiếp theo nằm trong `final_review.json`.

## 1. Môi trường đã kiểm tra trên máy Huy

Python 3.12.6, Windows, NVIDIA GeForce RTX 3050 Laptop GPU 4 GB.
PyTorch `2.11.0+cu128`, Transformers `4.57.6`. Các phiên bản phụ thuộc thực tế ghi
trong `reports/c3_distilbert/environment-lock.txt`. Không dùng môi trường Python CPU
toàn hệ thống để chạy lệnh train mặc định.

Tại thư mục gốc repo, nếu chưa có môi trường:

```powershell
python -m venv .venv-huy
New-Item -ItemType Directory -Force data/cache/pip, data/cache/tmp
$env:PIP_CACHE_DIR = (Resolve-Path data/cache/pip).Path
$env:TMP = (Resolve-Path data/cache/tmp).Path
$env:TEMP = $env:TMP
.\.venv-huy\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-huy\Scripts\python.exe -m pip install -r requirements-c3.txt
.\.venv-huy\Scripts\python.exe -m scripts.check_c3_environment
.\.venv-huy\Scripts\python.exe -m pip check
```

Máy hiện tại đã có môi trường này, không cần cài lại. Cache Hugging Face mặc định ở
`data/cache/huggingface` trên ổ chứa repo. Cài đặt trên GPU khác cần kiểm tra driver
và bản PyTorch phù hợp theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/).
Không kỳ vọng mọi bit kết quả giống nhau giữa các GPU/thư viện khác nhau; giữ revision,
split, seed, cấu hình và môi trường để tăng khả năng tái lập.
Trong pilot, PyTorch có cảnh báo backward của memory-efficient attention không bảo đảm
deterministic. Runner đang dùng `torch.use_deterministic_algorithms(True, warn_only=True)`;
không tuyên bố hai lần train sẽ giống từng bit. Kiểm reload checkpoint là một kiểm tra
khác: cùng trọng số đã lưu phải cho scores khớp trong sai số cho phép.

## 2. Pilot và luồng train

```powershell
.\.venv-huy\Scripts\python.exe -m scripts.train_distilbert --config configs/distilbert_pilot.json
```

Lệnh mặc định tạo `data/processed/c3_distilbert/pilot/seed_42/`. Run đã có dữ liệu sẽ bị
từ chối ghi đè. Muốn chạy lại, thêm `--output data/processed/c3_distilbert/pilot/recheck_42`.

1. Đọc snapshot dữ liệu đã ghim; kiểm SHA-256, mapping 28 nhãn, ID không giao nhau.
2. Chỉ đọc train/validation; lấy 1.024/256 mẫu bằng `subset_seed=2026`.
3. Tokenize tối đa 128 token, padding động; target multi-hot float32.
4. Nạp `distilbert/distilbert-base-uncased` tại revision
   `12040accade4e8a0f71eabdb258fecc2e7e948be`; tạo đầu ra 28 nhãn.
5. Fine-tune toàn bộ bằng BCEWithLogitsLoss, AdamW, AMP và gradient checkpointing.
6. Batch vật lý 4, tích lũy 4, batch hiệu dụng 16; lr `2e-5`, warmup 10%, weight decay
   0,01 (không decay bias/LayerNorm), gradient clipping 1,0. Pilot một epoch.
7. Chọn epoch bằng validation Macro-F1 tại ngưỡng 0,5; hòa giữ epoch sớm nhất.
8. Lưu checkpoint tốt nhất, checkpoint cuối, cấu hình, ID mẫu, lịch sử, metrics,
   scores và metadata; nạp lại checkpoint rồi đối chiếu scores.

Mã có lưu `training_state.pt` ở cuối run để hỗ trợ phát triển chức năng resume sau này;
hiện **chưa có lệnh resume**. Không coi file này là đủ để tự động tiếp tục một run bị ngắt.

## 3. Notebook và xuất báo cáo

Chọn interpreter `.venv-huy/Scripts/python.exe` khi mở notebook trong VS Code.
Notebook mặc định đọc ba run full đã chạy và tính lại các bảng; Run All không tự huấn luyện lại.

```powershell
.\.venv-huy\Scripts\python.exe -m scripts.run_c3_full
.\.venv-huy\Scripts\python.exe -m scripts.analyze_c3_full
.\.venv-huy\Scripts\python.exe -m scripts.report_c3_full
$chosen = Get-Content reports/c3_distilbert/full/representative_c3.json -Raw | ConvertFrom-Json
.\.venv-huy\Scripts\python.exe -m scripts.verify_distilbert --run $chosen.run --output reports/c3_distilbert/full/verification.json
.\.venv-huy\Scripts\python.exe -m scripts.build_c3_full_notebook
.\.venv-huy\Scripts\python.exe -m scripts.run_distilbert_notebook
```

Notebook có 16 bước: cấu hình, dữ liệu thật, multi-hot/tokenizer, kiểm loss, vòng train,
log ba seed, metric theo ID, mean/std, per-label, ngưỡng, nhãn hiếm, lỗi và demo.
Chạy kiểm demo (mục 7) trước bước execute notebook vì notebook đọc file verification.
HTML có sẵn để thành viên không có GPU đọc kết quả. Người clone repo không nhận checkpoint
vì Git bỏ qua `data/processed`; cần chạy full hoặc nhận nguyên thư mục run hoàn thành từ Huy.

## 4. Demo và CLI

```powershell
.\.venv-huy\Scripts\python.exe -m streamlit run app_distilbert_huy.py
.\.venv-huy\Scripts\python.exe -m scripts.predict_distilbert --text "I've never been this sad in my life!"
```

Mặc định suy luận bằng CPU; nếu muốn dùng GPU, đặt `$env:C3_DEMO_DEVICE = 'cuda'`
trước khi mở app. App hiển thị nhãn vượt ngưỡng và điểm đủ 28 nhãn. Mặc định chọn seed
C3 đại diện trong `full/representative_c3.json`; nếu chưa có full thì dùng pilot và báo rõ.
Câu mặc định của full là nguyên văn mẫu validation có ID, không phải ví dụ tự tạo.
Câu trong lệnh CLI trên là mẫu validation thật `edcu99z`.
Câu rỗng bị từ chối; câu dài bị truncate theo `max_length` và có thông báo. Không nhãn nào
đạt ngưỡng thì trả tập rỗng, không ép neutral. Văn bản đầu vào là tiếng Anh theo dữ liệu.

App và CLI đều dùng `src.c3.predict_texts`, cùng tokenizer, thứ tự nhãn và ngưỡng từ run.
Ngưỡng cơ sở là 0,5; có thể chọn ngưỡng từng nhãn từ validation của đúng checkpoint.
App/CLI dùng `src/c3_thresholds.py` để kiểm ngưỡng gắn đúng hash model/scores, seed, mapping.
CLI hỗ trợ `--threshold-mode per_label`. Không tự nhận checkpoint bất kỳ của C1/C2;
cần bàn giao và kiểm parity lại khi đã chọn mô hình thắng.

## 5. Artifact để bàn giao cho nhóm

Trong mỗi thư mục run:

| File/thư mục | Ý nghĩa |
|---|---|
| `best/` | Model safetensors, config, tokenizer của epoch được chọn |
| `last/`, `training_state.pt` | Trọng số/trạng thái cuối run; chưa có chức năng resume |
| `run.json`, `config.json` | Revision, môi trường, seed, config, nguồn code/hash, trạng thái run |
| `train_ids.csv`, `validation_ids.csv` | Đối chiếu chính xác tập mẫu đã dùng |
| `validation_scores.npz` | `ids` N phần tử, `scores` N×28 sigmoid, `label_names` 28 phần tử |
| `validation_metrics.json`, `per_label_validation.csv` | Metrics chung và TP/FP/FN/TN/support từng nhãn |
| `history.json` | Loss train, Micro/Macro-F1 validation theo epoch và thời gian |
| `inference_example.json` | Ví dụ dự đoán thật của checkpoint |

Không đổi thứ tự cột nhãn hoặc copy ngưỡng từ baseline A. Metadata ghi commit nền và
hash mã nguồn thực thi, bao gồm việc worktree có thay đổi chưa commit lúc chạy.
Không commit môi trường ảo, cache, checkpoint hoặc scores lớn vào Git.

## 6. Full, nâng cao và phần còn cần phối hợp

Cấu hình full: max length 128, batch vật lý/hiệu dụng 16, lr 2e-5, 3 epoch,
AMP, gradient checkpointing, 4 CPU threads. Kiểm khả năng chạy batch thật được ghi
trong `full_preflight.json`. Ba seed 42, 123, 2026 chạy tuần tự qua `scripts.run_c3_full`.
Runner kiểm run hoàn thành trước khi bỏ qua, từ chối trộn config/code hoặc ghi đè run dở.

`analyze_c3_full` kiểm hash/ID/mapping và tính lại metrics trước xuất bảng. Chọn ngưỡng
riêng từng nhãn, riêng mỗi seed trên lưới 0,05..0,95; hòa F1 chọn gần 0,5 nhất rồi ngưỡng
lớn hơn. Bảng cơ sở và tuned tách riêng. Tuned đo trên cùng validation nên có thể lạc quan;
chưa là kết luận cải thiện test. Không chạy thêm weighting/contrastive trong phần này.

Nhãn hiếm định nghĩa từ train: grief, pride, relief, nervousness, embarrassment. Bảng
trước/sau giữ đủ năm nhãn và đủ ba seed. Ví dụ lỗi dùng nguyên văn/ID thật; ba nhóm lỗi
C3 có thể giao nhau. Nhận xét thủ công nằm trong `full/ERROR_ANALYSIS.md`.

Các việc cần nhóm cung cấp đầu vào: đối chiếu lỗi cùng ID với C1/C2; chọn kiến trúc thắng
bằng mean Macro-F1 validation; bàn giao checkpoint/ngưỡng của C thắng cho demo cuối;
khóa protocol đánh giá test và tổng hợp báo cáo toàn nhóm. Huy không cần đợi kết quả
C1/C2 để hoàn thành phần train/validation C3 độc lập.

### Lượt chạy bị gián đoạn và tính minh bạch

Một lượt seed 42 ban đầu dừng ở epoch 3 do gradient clipping chặn cơ chế AMP overflow.
Log giữ trong `reports/c3_distilbert/interrupted_fp16_gradcheck/`; không đưa vào bảng
chính. Sau sửa, GradScaler bỏ bước lỗi, giảm scale và scheduler chỉ tiến khi cập nhật
thành công; mọi bước bỏ được ghi trong run metadata. Kiểm sửa lỗi dùng batch train thật
ở `amp_regression_check.json`. Ba seed đưa vào bảng chính phải chạy từ đầu với cùng mã.
Một lượt seed 123 bị gián đoạn trước khi hoàn thành epoch đầu; log được lưu riêng ở
`full/train_seed_123_interrupted_20261008.log`, không đưa vào thống kê. Run hoàn thành
seed 42 được giữ nguyên; seed 123 chạy lại từ checkpoint gốc, rồi đến seed 2026.

## 7. Kiểm tra mã và demo

```powershell
.\.venv-huy\Scripts\python.exe -m unittest discover -s tests -v
$chosen = Get-Content reports/c3_distilbert/full/representative_c3.json -Raw | ConvertFrom-Json
.\.venv-huy\Scripts\python.exe -m scripts.verify_distilbert --run $chosen.run --output reports/c3_distilbert/full/verification.json
```

Các kiểm tra gồm BCE đa nhãn, checkpoint round-trip, ngưỡng nhiều nhãn/tập rỗng,
chặn input sai, không đọc test, sample std, không tổng hợp pilot như full, đối chiếu
NPZ/metrics và điểm số CLI/app. File `verification.json` ghi kết quả kiểm thực tế.

Minh chứng demo, đường dẫn truy cập và kiểm khởi động lại được ghi trong
`reports/c3_distilbert/full/DEMO_EVIDENCE.md`.
