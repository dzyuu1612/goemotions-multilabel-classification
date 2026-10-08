# Hồ sơ tái hiện thí nghiệm A/B/C

## 1. Vì sao có hồ sơ JSON trong Git?

`data/processed/` được Git bỏ qua vì có model và scores lớn. Các bạn trong nhóm
cần đọc được cấu hình, seed, môi trường, revision, lịch sử epoch và kết quả để
đối chiếu báo cáo. `scripts/export_run_metadata.py` copy **nguyên byte JSON nhỏ
đã tồn tại** sang `reports/reproducibility/artifacts/`, giữ cấu trúc đường dẫn
nguồn. `manifest.json` ghi source path, source/export SHA-256, bytes và trạng thái.

Hồ sơ chứa metadata và kết quả có thật; các mục đang thiếu/chưa chạy xong vẫn
được ghi rõ, không sinh config, scores hoặc metric thay thế. File config/history
của A/B/C nằm trong metadata gốc; history C không được dựng từ mô tả kế hoạch.

## 2. Lệnh export

Chạy từ gốc repo, với môi trường đã cài dependencies dự án:

```powershell
python -m scripts.export_run_metadata
```

Mặc định C dùng `42 123 2026`, run **standard/full**. Nếu nhóm dùng tập seed
khác, truyền đúng cùng tập đã huấn luyện:

```powershell
python -m scripts.export_run_metadata --seeds 42 123 2026
```

Chạy lại sau khi hoàn thành thêm full runs hoặc đánh giá test. Export không
huấn luyện, chọn ngưỡng, gọi inference, tải model hay mở nhãn dataset test.
C dùng `load_transformer_run(..., require_full=True)` để kiểm completed flag,
sizes, mapping, checkpoint config và hash trước khi copy. B kiểm full validation
và scores hash; protocol/test JSON chỉ copy khi khớp khóa và artifact đã có.

Exporter chỉ cập nhật bản copy trong thư mục output. Nó gỡ các copy cũ có hash
đúng manifest cũ rồi xuất snapshot hiện tại; file copy đã được người khác sửa
được giữ và ghi `conflict`. Source `data/processed` được giữ nguyên.

## 3. Cấu trúc và phạm vi

```text
reports/reproducibility/
  README.md                     # trạng thái thật của A/B và 9 C full runs
  manifest.json                 # source/export SHA-256, trạng thái, code provenance
  artifacts/
    data/labels.json
    data/manifest.json
    data/processed/baseline/...  # metadata, analysis, thresholds, frozen/test JSON
    data/processed/zero_shot/... # full metadata, frozen protocol, test metadata
    data/processed/transformers/ # completed metadata, history, config, mapping,
                                 # validation/test metrics, selection và seed summary
```

Whitelist chỉ cho các JSON nhỏ (mặc định tối đa 1 MiB mỗi file). Các thành phần
model lớn, prediction NPZ, Parquet, văn bản ví dụ, tokenizer vocabulary và venv
được giữ trong kho làm việc/tệp bàn giao riêng. Metadata có model hash vẫn giúp
đối chiếu một bản model đầy đủ nhận từ đồng đội.

## 4. Cách đồng đội đối chiếu

1. Đọc README và `groups` trong manifest để biết run nào `complete`.
2. Chỉ dùng artifact có `status=exported`. `missing`, `incomplete`, `invalid`
   hoặc `conflict` không phải bằng chứng đã có kết quả.
3. Mở metadata đúng architecture/seed: kiểm checkpoint và **model_revision SHA
   thật**, data revision, labels, config, environment, sizes và history.
4. So `source_sha256` với `export_sha256`: phải bằng nhau. Đây là copy nguyên
   byte, không sửa số hoặc viết lại JSON cho đẹp.
5. Với bảng test, kiểm frozen protocol, thời điểm khóa và hash metadata/scores.
   Validation tuned đã dùng nhãn validation để chọn ngưỡng; phân biệt với test.
6. Khi có model/scores đầy đủ, dùng module kiểm của A/C và protocol checker của
   B/C để đối chiếu toàn bộ hash, rồi chạy bước báo cáo/demonstration phù hợp.

Ví dụ kiểm một bản copy trên Windows:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'reports/reproducibility/artifacts/data/processed/transformers/bert/seed_42/full/standard/run_metadata.json'
```

File JSON của thí nghiệm không chứa trọng số huấn luyện. Để suy luận/demo, cần
checkpoint thật hoặc chạy lại training. Model C có head đa nhãn 28 logits;
checkpoint pretrained và checkpoint sau fine-tune là hai giai đoạn trong quy
trình này. Bộ demo vẫn kiểm model thực và full-run hash trước khi chạy.

## 5. Tái hiện từ code

- Dùng data revision và checksum được ghi trong `data/manifest.json`; giữ split
  train/validation/test chính thức và thứ tự `data/labels.json`.
- A: dùng config/seed/môi trường trong `validation_metrics.json`; fit TF-IDF
  trên train, rồi đọc lại analysis và ngưỡng chọn trên validation.
- B: dùng checkpoint + commit SHA + template + `multi_label=True`; không
  fine-tune. Precision/dtype của inference ghi trong metadata nếu có.
- C: cùng architecture, checkpoint revision, seed, hyperparameters và môi
  trường; đọc `selected_epoch` và history để hiểu checkpoint được giữ.
- Khóa lại protocol sau khi tái hiện validation, trước đánh giá test. Kết quả
  có thể khác theo máy/thư viện; deterministic settings và version thật giúp
  giải thích giới hạn đó, không hứa giống từng bit trên mọi thiết bị.

`code_provenance` ghi Git HEAD, working-tree dirty flag và SHA-256 các source
chính **tại thời điểm export**. Trường `scope` ghi rõ phạm vi đó. Trainer hiện
chưa ghi Git commit lúc huấn luyện, nên hồ sơ này không khẳng định HEAD hoặc
source hashes là code đã chạy training. Không sửa metadata run đã hoàn tất
để điền một training commit suy đoán. Các JSON nguồn vẫn giữ nguyên byte/hash.

## 6. Kiểm tra exporter

```powershell
python -m unittest tests.test_run_metadata -v
```

Tests dùng thư mục tạm/metadata giả để kiểm copy nguyên byte, whitelist/no raw
text, JSON quá lớn, path traversal, incomplete C, missing sources và bảo toàn
file đã được người khác sửa. Không chạy training hoặc GPU.
