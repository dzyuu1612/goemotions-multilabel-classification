# GoEmotions — phân loại cảm xúc đa nhãn

**Nguyễn Bảo Duy — Nhóm 6, HUFLIT.** Mã nguồn và notebook đối chiếu **Word Final v7.1 / PPT v9**. A: TF-IDF + Logistic Regression; B: BART-MNLI; C1: BERT; C2: RoBERTa; C3: DistilBERT.

Mở notebook theo bảng dưới. Mỗi phần có lời giải thích, cell thực hiện và đầu ra thật. Các bảng test tính lại từ điểm dự đoán đã khóa; biểu đồ được vẽ từ CSV tương ứng. Không cần huấn luyện lại để trình bày. Notebook 00 là ví dụ nhỏ tự tạo, tách rõ khỏi benchmark.

## Notebook đã chạy

Mỗi notebook chỉ giữ một bản có đầu ra thật, không có bản `.executed` trùng lặp.

| Nội dung | Notebook |
|---|---|
| Ví dụ baseline nhỏ trên CPU | [00_quick_start](notebooks/00_quick_start.ipynb) |
| Phân tích dữ liệu GoEmotions | [01_eda](notebooks/01_eda.ipynb) |
| A — TF-IDF và Logistic Regression | [02_baseline](notebooks/02_baseline.ipynb) |
| B — BART-MNLI, tính lại từ điểm đã lưu | [03_zero_shot](notebooks/03_zero_shot.ipynb) |
| **04 · Ba mô hình C, đủ 9 lượt, F1/BCE và overfit** | [Kết quả mô hình C](experiments/best_epochs_20261103/GoEmotions_KetQua_EpochTotNhat.ipynb) |
| **05 · Ba chế độ ngưỡng, năm loại lỗi, giao nhóm, near-miss** | [Ngưỡng và lỗi](experiments/threshold_error_20261110/GoEmotions_PhanTich_Nguong_Loi_DaChay.ipynb) |
| 06 · Input ID, mask, P95 và số câu có [UNK] | [06_inputs_tokenizers](notebooks/06_inputs_tokenizers.ipynb) |
| 07 · Tính tay TF-IDF, attention, LayerNorm/GELU, ba head, BCE/SGD, KL | [07_hand_calculations](notebooks/07_hand_calculations.ipynb) |

Notebook C tính lại 18 hàng test × 7 chỉ số. Notebook 05 kiểm đủ 33 cấu hình, 11 bộ ngưỡng validation và cách đếm lỗi. Notebook 01/06 dùng đúng **BERT cased** của C1, không dùng nhầm BERT uncased lịch sử. Notebook 07 dùng đúng ví dụ trong Chương 2: ba head 0,7109/0,7159/0,6942; KL 0,0191; SGD một nhãn 1,2000035. `notebooks/execution_manifest.json` kiểm mã băm, thứ tự thực thi và lỗi của toàn bộ tám notebook.

Chạy lại mọi notebook bằng **chính Python đang gọi lệnh**, cập nhật đầu ra vào một bản duy nhất:

```powershell
python -m scripts.data.run_notebooks --notebook all --publish
```

Chạy riêng một phần bằng tên tệp không có `.ipynb`, ví dụ `--notebook 07_hand_calculations --publish`. Cần cài phụ thuộc bên dưới. A khôi phục hai model TF-IDF đã fit và suy luận test thật; B/C kiểm từ scores đã lưu. Chạy lại fine-tuning có thể lệch do GPU/AMP, không hứa trùng từng bit với checkpoint lịch sử.

[Hướng dẫn phân tích theo nhận xét của giảng viên](experiments/threshold_error_20261110/README.md) · [Kế hoạch đã ghi](experiments/threshold_error_20261110/DECISION_PHANTICH.md) · [Bảng so ba chế độ](experiments/threshold_error_20261110/results/modes_mean_std.csv). Phần gán nhãn nguyên nhân cần hai người độc lập; các phiếu vẫn để trống, chưa công bố tỷ lệ đồng thuận.

## Cài đặt

Python 3.12–3.13; cài PyTorch phù hợp máy trước nếu cần CUDA khác.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[eda,transformers,notebooks]"
```

Môi trường đã chạy: Python 3.13.9, PyTorch 2.13.0+cu130, Transformers 4.57.6, RTX 5060 Laptop.

## Kết quả dùng trong Word v7.1 / PPT v9

Bộ chia chính thức: **43.410 train / 5.426 validation / 5.427 test**, 28 nhãn. Cả ba C dùng LR `2e-5`, batch 16, max_length 128, tối đa 8 epoch và ba seed 42/123/2026. Epoch chọn: **BERT 7/6/7; RoBERTa 7/8/5; DistilBERT 7/8/8**. Chọn checkpoint bằng Macro-F1 validation @0,5, hòa chọn epoch sớm; khóa checkpoint và ngưỡng trước khi chấm test.

| Mô hình · ngưỡng riêng | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| BERT | 0,4938 ± 0,0075 | 0,5717 ± 0,0051 | 0,0405 ± 0,0016 |
| RoBERTa | 0,5101 ± 0,0076 | 0,5909 ± 0,0033 | 0,0377 ± 0,0002 |
| DistilBERT | 0,4972 ± 0,0027 | 0,5779 ± 0,0069 | 0,0389 ± 0,0012 |

Trung bình và std mẫu (`ddof=1`) từ đủ ba seed. RoBERTa cao nhất trung bình trong cấu hình đã đo; chưa kiểm định thống kê. C2/C3 tăng Macro-F1 so với lượt ban đầu, nhưng Micro-F1 cả ba giảm và Hamming Loss tăng. BERT còn đổi LR so với lượt cũ. Train BCE giảm trong khi validation BCE tăng ở epoch sau: vẫn có dấu hiệu overfit theo loss. Test cũ đã được xem trước khi đổi lịch; cần holdout mới để kết luận mạnh hơn.

[CSV từng seed](experiments/best_epochs_20261103/main_results/report_data/all_runs.csv) · [Mean/std](experiments/best_epochs_20261103/main_results/report_data/mean_std.csv) · [Lịch sử 72 epoch](experiments/best_epochs_20261103/sweep_log.csv) · [Bảng lịch sử B.3](experiments/best_epochs_20261103/main_results/report_data/original_test_summary.csv)

Kiểm lại số liệu mà không huấn luyện:

```powershell
python tools/check_repository.py
python experiments/best_epochs_20261103/verify_report_tables.py
python -m unittest discover -s tests -v
```

## Demo và huấn luyện

```powershell
python -c "from src.final_submission import restore_baselines; restore_baselines()"
python experiments/best_epochs_20261103/restore_best_models.py
python app.py --device auto
```

Demo nạp checkpoint đã lưu, mặc định **RoBERTa/2026, ngưỡng riêng**, chạy cục bộ tại `http://127.0.0.1:7861`. Chưa có URL demo công khai. [Gói checkpoint tải riêng](https://github.com/dzyuu1612/goemotions-multilabel-classification/releases/tag/goemotions-best-epochs-20261103) được kiểm SHA-256; không đưa weights lớn vào cây mã nguồn.

Huấn luyện C1/C2/C3 dùng [trainer đã chạy](experiments/best_epochs_20261103/train_transformer.py); lệnh và cấu hình trong [README thí nghiệm](experiments/best_epochs_20261103/README.md). Seed không bảo đảm GPU huấn luyện lại trùng từng bit; notebook dùng điểm đã lưu để tái hiện bảng.

## Thư mục

- `src/`, `scripts/`: xử lý dữ liệu, huấn luyện, đánh giá và ứng dụng.
- `notebooks/`: dữ liệu, A/B, đầu vào và tính tay. Hai notebook C/ngưỡng nằm cạnh bộ điểm đã khóa trong `experiments/`; README ở đây là mục lục chính, không có notebook output trùng.
- `experiments/best_epochs_20261103/`: trainer C đã dùng, chín bộ điểm, log 72 epoch, biên nhận và hàm nạp checkpoint. Tên đường dẫn được giữ vì hồ sơ khóa có ràng buộc hash.
- `experiments/threshold_error_20261110/`: kiểm toán validation, 33 cấu hình, lỗi và phiếu đọc độc lập.
- `src/report_figures.py`: hàm vẽ biểu đồ từ bảng số, dùng trực tiếp trong notebook.
- `tests/`, `examples/`, `configs/`, `requirements/`: kiểm thử, ví dụ và cấu hình.
- `results/final_submission/`: bốn tệp cần để khôi phục A/B và đối chiếu số lịch sử. Gói runtime chứa dữ liệu/điểm đã lưu để notebook và demo chạy được; không phải bản Word/PPT hay bản code thứ hai.

Đã bỏ 35 tệp điều phối/đồ thị/bộ sinh notebook cũ không còn được gọi. Không bỏ log, protocol, điểm dự đoán, kiểm thử hay giấy phép cần để đối chiếu. Dữ liệu, weights và báo cáo sinh khi chạy ở các thư mục bị Git bỏ qua, không đưa lên cây mã nguồn. Các bản trước còn trong lịch sử Git.

Giới hạn được giữ đúng như Word: ngưỡng chỉ tốt nhất trong lưới 0,05–0,95; B có 26/28 ngưỡng riêng ở 0,95. Không mở rộng lưới theo test. Cặp relief→admiration chỉ là ví dụ một câu ở mô hình đại diện; 145 mục đọc tay chưa có hai bộ nhãn độc lập, nên chưa có tỷ lệ đồng thuận hoặc kết luận nguyên nhân. Demo chỉ chạy cục bộ, chưa có URL công khai.

[Bài báo GoEmotions](https://aclanthology.org/2020.acl-main.372/) · [Dữ liệu chính thức](https://github.com/google-research/google-research/tree/master/goemotions)
