# Nội dung cập nhật mục riêng của Duy

## Bộ code, kết quả và báo cáo — cập nhật 09/10/2026
**Đã hoàn tất hồ sơ thực nghiệm chính:** A hai biến thể; B zero-shot full; C1/C2/C3 mỗi kiến trúc ba seed 42, 123, 2026; đánh giá test sau khóa protocol. Bảng có **72 hàng thực nghiệm, 36 nhóm tổng hợp, không còn mục thiếu**. D dùng **BERT seed 123**, chọn từ tiêu chí Macro-F1 validation trung bình ba seed của kiến trúc, rồi seed đại diện tốt nhất trên validation. Không chọn lại bằng test.
- **Phần baseline của Duy:** TF-IDF học từ train, 58.338 đặc trưng, 28 Logistic Regression độc lập; hai model standard/balanced, ngưỡng gốc/chung/riêng, sáu cấu hình có cả validation và test. Output A dùng làm mốc so sánh; không đưa vào BERT làm input.
- **Kiểm mã và notebook:** lượt kiểm trước ghép kho chung đạt 69/69; sau ghép phần Huy đạt **77/77**, có log riêng. Baseline 12/12 ô mã, B 4/4, C 7/7, không có output lỗi. Hồ sơ kiểm sau ghép và trạng thái cuối xem [verification_project.json](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/verification_project.json), không cộng các lượt lịch sử thành số test khác nhau.
- **Demo:** kiểm checkpoint trên ba câu validation và 28 scores; kiểm app thật bằng nút bấm, bảng đủ 28 nhãn, có [ảnh và hồ sơ giao diện](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/demo_ui/evidence.json). Bằng chứng suy luận và giao diện là hai phép kiểm riêng.
- **Phân tích lỗi:** ba ID khác nhau, mỗi ID đối chiếu cả BERT/RoBERTa/DistilBERT, giữ nguyên ground truth/scores/ngưỡng; có nhận xét từ văn bản và giới hạn diễn giải. [Ba case thật](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/error_case_studies.md).
- **Hai dạng bản nộp:** [báo cáo sáu chương Word](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.docx) · [PDF](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.pdf); [bài IEEE hai cột Word](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/BAI_BAO_GOEMOTIONS_IEEE.docx) · [PDF](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/BAI_BAO_GOEMOTIONS_IEEE.pdf). Cùng bộ số thật; không tự nhận đã xuất bản IEEE.
- **Tiến độ:** có hai bản Word/PDF phản ánh trạng thái lúc xuất; không ghi đã nộp cho cô. Giữ trống giảng viên, lớp, MSSV và tỷ lệ đóng góp chưa xác nhận.
- **Học để bảo vệ:** [TF-IDF tính tay, từng dòng code, ba ví dụ A thật và 20 câu hỏi với đáp án](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/docs/HUONG_DAN_DUY_GIAI_THICH_BASELINE.md). Ví dụ toán minh họa được phân biệt với benchmark.
### Sáu cấu hình baseline: kết quả test đã khóa
<table fit-page-width="true" header-row="true">
<tr>
<td>Cấu hình A</td>
<td>Macro-F1</td>
<td>Micro-F1</td>
<td>Hamming Loss</td>
</tr>
<tr>
<td>Standard @0.5</td>
<td>0.1963</td>
<td>0.3800</td>
<td>0.0348</td>
</tr>
<tr>
<td>Standard ngưỡng chung</td>
<td>0.4096</td>
<td>0.5047</td>
<td>0.0553</td>
</tr>
<tr>
<td>Standard ngưỡng riêng</td>
<td>0.4134</td>
<td>0.5330</td>
<td>0.0444</td>
</tr>
<tr>
<td>Balanced @0.5</td>
<td>0.4441</td>
<td>0.5024</td>
<td>0.0547</td>
</tr>
<tr>
<td>Balanced ngưỡng chung</td>
<td>0.4530</td>
<td>0.5157</td>
<td>0.0480</td>
</tr>
<tr>
<td>Balanced ngưỡng riêng</td>
<td>0.4493</td>
<td>0.5277</td>
<td>0.0467</td>
</tr>
</table>
So standard @0.5, balanced + ngưỡng riêng tăng Macro-F1 từ 0.1963 lên 0.4493. Tuy nhiên balanced ngưỡng chung có Macro-F1 test cao hơn ngưỡng riêng; standard ngưỡng riêng có Micro-F1 cao hơn balanced ngưỡng riêng. Giữ cấu hình đã chọn bằng validation; không đổi lựa chọn sau khi thấy test. Hamming tăng khi bổ sung nhiều nhãn hơn: phải đọc cùng Precision/Recall/F1.
### Nâng cao và nhãn hiếm của phần A
Năm nhãn hiếm được xác định từ train: grief 77, pride 111, relief 153, nervousness 164, embarrassment 303 mẫu. Trên test, standard @0.5 có F1 bằng 0 ở cả năm; balanced + ngưỡng riêng lần lượt 0.4615 / 0.4167 / 0.1176 / 0.1714 / 0.2778. So với balanced @0.5, tuning vẫn có nhãn giảm: xem [bảng trước/sau đầy đủ](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/BASELINE_RESULTS.md); không ghi mọi cải tiến đều thắng.
### C và B liên quan đến phần của Duy như thế nào?
C là đối chứng để nhóm biết fine-tuning cải thiện đến đâu. BERT test Macro-F1 ba seed @0.5 **0.4720 ± 0.0045**, sau ngưỡng riêng **0.5038 ± 0.0097**. RoBERTa tuned có Micro-F1 cao hơn BERT tuned; “BERT tốt nhất” chỉ theo tiêu chí chọn validation đã công bố. Không dùng std như kiểm định ý nghĩa thống kê.
BART-MNLI không fine-tune GoEmotions; fixed test Macro-F1 **0.1035**, tuned **0.1609**. Audit không thấy bug cụ thể ở head MNLI/mapping; fixed dự đoán trung bình 15.3818 nhãn/câu trong khi nhãn thật 1.1662, nên nhiều FP. [Hồ sơ audit B](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/ZERO_SHOT_DIAGNOSTICS.md) nêu rõ chưa đối chiếu raw logits mới, không kết luận nguyên nhân nhân quả.
[Hồ sơ cấu hình và 95 JSON có checksum](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/reproducibility/manifest.json) · [Bảng tổng hợp](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/project_results/RESULTS.md) · [Đối chiếu yêu cầu cô](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/docs/DOI_CHIEU_YEU_CAU_CO.md) · [Kiểm nguồn](https://github.com/dzyuu1612/goemotions-multilabel-classification/blob/codex/baseline-starter/reports/REFERENCE_CLAIM_AUDIT.md).
**Git:** PR #3 đã nhập kho chung ngày 05/10/2026; [PR #4](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/4) là phần bổ sung trên nhánh Git của Duy, chờ nhóm review/merge. Công việc Nhật Huy đã có trên kho chung được giữ nguyên như nghiên cứu C3 riêng; không trộn các run khác trainer/protocol vào chín run chính.
Phân công là trách nhiệm bàn giao; mã chuẩn bị bằng công cụ không tự chứng minh đóng góp của từng người. Nhóm cần tự đọc, chạy lại và xác nhận công việc thực tế. Các mục có ngày 03/10 bên dưới là lịch sử baseline, cập nhật này là trạng thái mới hơn.
