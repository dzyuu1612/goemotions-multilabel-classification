# Thứ tự đọc phần baseline của Bảo Duy Nguyễn

Cập nhật 09/10/2026; kết quả/kiểm thử full ngày 08/10. Vai trò trong nhóm 4 người:
baseline A và điều phối B làm chung; GitHub `dzyuu1612`.
Đây là trang đánh dấu để bạn quay lại học sau. Mã và output đã có, nhưng bạn cần
tự đọc, chạy và giải thích được trước khi bảo vệ.

## Bốn tài liệu chính — đọc đúng thứ tự này

| Thứ tự | Tài liệu | Cách đọc | Sau khi đọc cần làm được |
|---|---|---|---|
| 1 | [Notebook baseline](../notebooks/baseline.ipynb) | Đọc từng cell và output; thử một câu tiếng Anh khác | Nối text → TF-IDF → 28 scores → ngưỡng → nhãn |
| 2 | [Hướng dẫn baseline](BASELINE.md) | Mục 1–5 trước, rồi 6–10; có lệnh cài và chạy | Chạy hai model, giải thích OvR, metric, weighting, tuning |
| 3 | [Bảng kết quả](../reports/BASELINE_RESULTS.md) | Đọc sáu hàng validation và sáu hàng test, năm nhãn hiếm và cặp lỗi | Phân biệt kết quả validation với test; giải thích cả nhãn giảm |
| 4 | [Hồ sơ đối chiếu yêu cầu cô](BASELINE_REVIEW.md) | Đọc yêu cầu, thiếu sót đã sửa và bằng chứng kiểm chứng | Biết phần nào A đã có và phần nào nhóm còn phải làm |

Nếu GitHub chưa hiển thị notebook ngay, tải file và mở bằng Jupyter/VS Code.
Notebook đã có output thật nên có thể đọc trước khi cài môi trường.
Đọc thêm [hướng dẫn Duy giải thích với cô](HUONG_DAN_DUY_GIAI_THICH_BASELINE.md):
ví dụ số TF-IDF đúng sklearn, sigmoid, từng dòng code và 20 câu hỏi bảo vệ.

Luồng A: **text → TF-IDF đã fit train → 28 LR → 28 scores → ngưỡng từ val → tập nhãn**.
A đáp ứng baseline cổ điển/đa nhãn; số A làm đối chứng, không làm đầu vào B/C.
D hiện dùng BERT cased seed123 chọn theo validation của 9 run C.

**Học để giải thích:** [TF-IDF tính tay, sigmoid/OvR, từng dòng code và 20 câu hỏi bảo vệ](HUONG_DAN_DUY_GIAI_THICH_BASELINE.md).
Đọc tài liệu này sau notebook và mục 1–5 của BASELINE.md; ví dụ toán được ghi rõ
là minh họa, tách với số GoEmotions đo thật.

## Các kiến thức cần học

Đọc theo chủ đề dưới đây và tự kiểm bằng bài tập; không đặt thời lượng.

| Thứ tự chủ đề | Nội dung | Bài tập ngắn để tự kiểm |
|---|---|---|
| 1 | Python list, dict, hàm, import; NumPy shape và indexing | Với ba nhãn, đổi `[0, 2]` thành `[1, 0, 1]`; giải thích `Y.shape` |
| 2 | Train/validation/test và TF-IDF | Chỉ ra dòng `.fit` dùng train; giải thích tại sao val không mở rộng từ vựng |
| 3 | Logistic Regression và One-vs-Rest | Với score `[0.8, 0.3, 0.7]`, ngưỡng 0.5, chọn đúng hai nhãn |
| 4 | TP/FP/FN, Precision/Recall/F1 | Tính tay một ví dụ hai nhãn và đối chiếu `tests/test_metrics.py` |
| 5 | Class weighting và chọn ngưỡng | Giải thích Recall tăng nhưng Precision hoặc Hamming có thể xấu đi |
| 6 | Bảng kết quả và lỗi theo ID | Chọn một ví dụ, đọc nhãn thật/đoán, chỉ ra FN và FP, nêu giới hạn diễn giải |

Không cần học PyTorch/Transformer để chạy A. Cần hiểu khái quát B/C để giải thích
baseline là mốc so sánh trong đồ án. Bạn điều phối B làm chung, tiếp nhận phần
zero-shot Đức Trí đã nhận trước nếu có; Đức Trí sở hữu RoBERTa C2.
Để làm B, học thêm NLI/entailment, candidate labels, `multi_label=True`,
template và ánh xạ scores về thứ tự nhãn chuẩn. Tham khảo R07/R08/R18 trong
[kế hoạch nhóm](KE_HOACH_NHOM.md). A/B/C đã có validation/test full:
[72 bản ghi/36 dòng tổng hợp](../reports/project_results/RESULTS.md),
[95 JSON metadata/protocol](../reports/reproducibility/README.md).
Giữ trọng số B không fine-tune; ngưỡng B tuned dùng nhãn validation và phải công bố rõ.

## Code đọc sau khi hiểu notebook

1. [Mô hình A](../scripts/run_baseline.py): `build_model`, fit train, predict validation.
2. [Chỉ số chung](../src/metrics.py): ngưỡng và TP/FP/FN/TN, micro/macro.
3. [Ngưỡng và ghép ID](../src/baseline.py): mỗi model có bộ ngưỡng riêng.
4. [Phân tích](../scripts/analyze_baseline.py): sáu cấu hình, nhãn hiếm, cặp lỗi.
5. [Dự đoán một câu](../scripts/predict_baseline.py): thử câu tiếng Anh.
6. [Xuất bảng lên GitHub](../scripts/export_baseline_results.py): JSON/CSV nhỏ để bàn giao.
7. [Khóa cấu hình](../scripts/freeze_baseline.py) và [đánh giá test](../scripts/evaluate_baseline_test.py):
   đọc protocol đã có và kết quả đã đo; không chốt lại cấu hình theo test.

## Tài liệu nhóm và báo cáo cá nhân

- [Kế hoạch toàn nhóm](KE_HOACH_NHOM.md): A/B/C/D, phân công theo tên và sản phẩm.
- [Mục riêng của Duy trên Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d): tài liệu, bảng kết quả và báo cáo baseline.
- [Báo cáo tiến độ của bạn](../reports/BAO_CAO_BASELINE_BAO_DUY.md): phần đã làm, kết quả và việc còn lại.
- [Bảng validation dạng CSV](../reports/baseline_validation/comparison.csv): số đầy đủ để copy vào bảng báo cáo.
- [Bảng test/từng seed](../reports/project_results/all_runs.csv), [mean/std](../reports/project_results/mean_std.csv)
  và [nhãn hiếm trước/sau](../reports/project_results/rare_before_after.csv).
- Tiến độ 1: [Word](../reports/BAO_CAO_TIEN_DO_1.docx)/[PDF](../reports/BAO_CAO_TIEN_DO_1.pdf);
  tiến độ 2: [Word](../reports/BAO_CAO_TIEN_DO_2.docx)/[PDF](../reports/BAO_CAO_TIEN_DO_2.pdf).
  Có tệp không xác nhận đã nộp hoặc đã xác thực đóng góp.
- [Kiểm notebook A12/B4/C7](../reports/execution/notebook_verification.json),
  [69 tests ngày 08/10](../reports/verification_project.json),
  [ảnh demo](../reports/demo_ui/demo_ui.png) và [evidenceUI](../reports/demo_ui/evidence.json).

## Checklist tự học trước buổi báo cáo

- [ ] Tôi tự chạy được A và dự đoán một câu.
- [ ] Tôi giải thích được 27 cảm xúc + neutral và multi-hot N×28.
- [ ] Tôi biết TF-IDF chỉ fit trên train.
- [ ] Tôi phân biệt Micro-F1, Macro-F1, Precision, Recall, Hamming Loss.
- [ ] Tôi giải thích được tác dụng/đánh đổi của weighting và ngưỡng.
- [ ] Tôi đọc được một cặp FN/FP có ID và không gọi EDA là lỗi dự đoán.
- [ ] Tôi gọi đúng bảng validation/test; biết điểm tuned-val có thể lạc quan.
- [ ] Tôi giải thích được balanced global Macro-F1 test cao hơn balanced tuned,
  standard tuned Micro-F1 test cao hơn balanced tuned, và bốn nhãn giảm khi tuning sau weighting.
- [ ] Tôi phân biệt đóng góp A của mình với EDA và B/C/demo của cả nhóm.
