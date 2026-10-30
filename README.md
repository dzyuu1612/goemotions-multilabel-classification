# GoEmotions phân loại cảm xúc đa nhãn

Mã nguồn cho TF-IDF + Logistic Regression (A), zero-shot BART-MNLI (B), BERT (C1), RoBERTa (C2) và DistilBERT (C3). GoEmotions có 28 nhãn, official split **43.410 train / 5.426 validation / 5.427 test**. Mỗi kiến trúc C có ba seed **42, 123, 2026**.

Bộ dùng cho báo cáo nằm ở `results/final_submission/`. Notebook đã chạy có kết quả sẵn ngay trong `notebooks/`; bản sao và biên nhận thực thi nằm ở `notebooks/executed/`. Không cần train lại để xem kết quả hoặc demo.

- [Nhánh mã nguồn hiện tại](https://github.com/dzyuu1612/goemotions-multilabel-classification/tree/codex/goemotions-upgrade-20261029)
- [Kết quả từng lượt](results/final_submission/report_data/all_runs.csv)
- [Mean và std mẫu ddof=1](results/final_submission/report_data/mean_std.csv)
- [Cấu hình và môi trường C3](results/c3_rerun/ENV.json)

## Cài đặt

Python 3.12 hoặc 3.13. Môi trường huấn luyện C1/C2/C3 thực tế: Python 3.13.9, PyTorch 2.13.0+cu130, Transformers 4.57.6, CUDA runtime 13.0, RTX 5060 Laptop. PyTorch cần bản phù hợp GPU/CPU của máy sử dụng; môi trường khác có thể có sai lệch số học gần ngưỡng.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[eda,transformers,notebooks]"
```

## Xem kết quả và chạy demo

```powershell
# Tải checkpoint đã huấn luyện, kiểm SHA-256; không train.
python -m scripts.verification.prepare_submission --checkpoints --demo-only
python app.py --device auto
```

Gradio tại `http://127.0.0.1:7860`, mặc định **BERT seed 123 / ngưỡng riêng**. Có sáu lựa chọn mô hình và hai chế độ **0,5 / ngưỡng riêng**. BART-MNLI nạp khi được chọn. Thiếu checkpoint của mô hình khác thì khôi phục tất cả:

```powershell
python -m scripts.verification.prepare_submission --checkpoints
```

Đầu ra gồm tập nhãn và bảng 28 điểm, ngưỡng áp dụng, trạng thái đạt ngưỡng. Không có nhãn đạt ngưỡng thì trả tập rỗng, không tự thêm neutral. Điểm sigmoid/NLI chưa được hiệu chuẩn thành độ tin cậy tuyệt đối.

Năm câu tự soạn để thử, không dùng tính accuracy:

1. You stayed late to help me. I really appreciate it!
2. I was worried, but your help made me feel so relieved.
3. I am not angry with you, just disappointed in the result.
4. Wonderful, another three hours stuck in traffic.
5. The package will arrive on Tuesday afternoon.

Với câu 2, BERT ở ngưỡng riêng chọn nervousness và relief; tại 0,5 không có nhãn đạt ngưỡng. [Gói chuẩn bị cho Hugging Face Spaces](deployment/hf_space/README.md) chỉ dùng BERT/123 trên CPU. [Hướng dẫn triển khai](deployment/DEPLOY_DEMO.md); chưa có link Space công khai.

## Notebook

| Tệp | Nội dung |
|---|---|
| [00 Quick start](notebooks/00_quick_start.ipynb) | Ví dụ baseline nhỏ tự tạo, tách khỏi benchmark |
| [01 EDA](notebooks/01_eda.ipynb) | Dữ liệu, phân bố nhãn, trùng văn bản, độ dài tokenizer |
| [02 Baseline](notebooks/02_baseline.ipynb) | Hai biến thể TF-IDF + LR, ngưỡng và kiểm kết quả |
| [03 Zero shot](notebooks/03_zero_shot.ipynb) | BART-MNLI và đánh đổi ngưỡng |
| [04 C1 C2](notebooks/04_transformers.ipynb) | Cấu hình, đường học, bảng báo cáo và suy luận thật sáu checkpoint |
| [05 C3](notebooks/05_distilbert.ipynb) | Ba seed DistilBERT, validation/test, nhãn hiếm, lỗi và demo |
| [06 EDA bổ sung](notebooks/06_eda_extra.ipynb) | Đồng xuất hiện nhãn thật và kiểm chất lượng dữ liệu |
| [07 Tính tay](notebooks/07_improvement_workflow.ipynb) | Attention, sigmoid, BCE, gradient và cách đọc tham số |

Sau khi khôi phục đủ checkpoint:

```powershell
python -m scripts.data.run_notebooks --notebook all --publish
```

Lệnh thực thi toàn bộ cell, dừng khi có lỗi, lưu đầu ra vào notebook chính và bản sao `executed/`, kèm thời gian và SHA-256. **Run All không huấn luyện benchmark**. Bảng báo cáo được tính từ đúng scores của lượt đã chốt, kiểm ID, thứ tự nhãn và đủ ba seed. Các cell suy luận mới hiển thị riêng sai lệch nếu AMP/kernel khác; không sửa số để ép khớp. Huấn luyện và thực thi notebook được chạy trên máy có GPU; GitHub Actions kiểm mã, biên nhận notebook và tính lại 72 dòng metric trên CPU.

## Dữ liệu huấn luyện và đánh giá

TF-IDF chỉ fit trên train. Checkpoint chọn bằng Macro-F1 validation tại 0,5. Ngưỡng riêng chọn trên validation trong lưới 0,05–0,95, bước 0,05; hòa điểm ưu tiên gần 0,5, rồi ngưỡng lớn hơn. Không dùng test chọn seed hoặc ngưỡng. CSV giữ thêm chế độ global để lưu vết; báo cáo chỉ trình bày 0,5 và ngưỡng riêng.

```powershell
python -m scripts.verification.verify_submission --output reports/verification_current --full
python -m scripts.verification.crosscheck_submission
python -m scripts.analysis.plot_report_revision --output ../report_out/C3_20261030
```

Ma trận lỗi dùng **Blues**. Ô FN(A)–FP(B) đếm câu vừa bỏ sót A vừa dự đoán thừa B; một câu có thể đóng góp nhiều ô. Đây không phải confusion matrix của bài toán đơn nhãn. F1 từng nhãn giữ đủ 28 nhãn, mean/std của ba seed.

## Huấn luyện

Trainer chung: `scripts/transformers/train_transformer.py`. Ba C dùng BCEWithLogitsLoss không trọng số, AdamW, batch 16, accumulation 1, max length 128, warmup 10%, weight decay 0,01, clip norm 1 và AMP fp16 trên CUDA. C1 dùng LR 5e-5 / 4 epoch; C2/C3 dùng 2e-5 / 3 epoch. C3 dùng cùng trainer với C1/C2 và không bật gradient checkpointing.

Ví dụ tạo **thí nghiệm riêng**, không ghi đè bộ báo cáo:

```powershell
python -m scripts.transformers.train_transformer --architecture distilbert --seed 42 --device cuda --revision 12040accade4e8a0f71eabdb258fecc2e7e948be --output-root artifacts/my_experiment
```

Thực thi C3 của bộ báo cáo được lưu ở `results/c3_rerun/`: quyết định trước khi chạy, log theo epoch, môi trường, ba seed và biên nhận. `scripts/transformers/rerun_c3.py` từ chối khởi động lại một thí nghiệm đã có. `scripts/transformers/finalize_c3.py` chỉ đánh giá sau khi cả ba seed hoàn tất; khóa ngưỡng trước test và giữ nguyên byte của các dòng A/B/C1/C2. Bộ trước được lưu tại `results/archive_before_c3_rerun/`, checkpoint C3 trước tại `results/archive_c3_old/` (trọng số lớn tải qua Releases).

## Giới hạn

Có 37 dòng test trùng nguyên văn với train. Test của bộ trước và bộ chính đã được quan sát; so sánh là mô tả. C1/C2/C3 khác LR/epoch, chỉ ba seed; chưa có ablation để kết luận nguyên nhân chênh lệch. Ngưỡng validation có thể quá khớp nhãn hiếm. Tăng epoch hoặc đổi LR chưa mặc nhiên làm F1 tốt hơn.

## Nguồn

- [GoEmotions ACL 2020](https://aclanthology.org/2020.acl-main.372/)
- [Dữ liệu chính thức Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

Nguồn và trích dẫn chi tiết nằm trong báo cáo. Không coi dữ liệu minh họa tự tạo hoặc câu demo là bằng chứng chất lượng mô hình.
