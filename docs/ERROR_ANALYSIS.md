# So sánh lỗi C1/C2/C3 theo yêu cầu cô

## 1. Điều kiện trước khi chạy

C1 = BERT (Quốc Khánh), C2 = RoBERTa (Đức Trí), C3 = DistilBERT (Nhật Huy).
Cần ít nhất ba run **full** với cùng tập seed cho mỗi kiến trúc, đủ train/val
chính thức, đủ 28 nhãn và artifact hợp lệ. Công cụ không huấn luyện hay tải model.

Mỗi kiến trúc lấy một checkpoint đại diện: seed có validation Macro-F1 **ở
ngưỡng 0,5** lớn nhất; hòa chọn seed nhỏ. Việc này không nhìn test. Bảng lỗi của
checkpoint đại diện không được ghi thành mean ± std qua các seed. Bảng chất
lượng mean ± std vẫn lấy từ `scripts.select_best_transformer`.

## 2. Ba nhóm lỗi có định nghĩa đo được

| Nhóm | Điều kiện ở một câu | Mẫu dùng làm mẫu số |
|---|---|---|
| `partial_multi_label` | Nhãn thật có ≥2 cảm xúc; đoán đúng ≥1 nhãn, đồng thời bỏ sót ≥1 nhãn thật | Những câu có ≥2 nhãn thật |
| `rare_false_negative` | Có ≥1 nhãn hiếm thật bị bỏ sót (FN) | Những câu có ít nhất một nhãn hiếm thật |
| `missed_extra_pair` | Có ≥1 nhãn thật bị bỏ sót (FN) và ≥1 nhãn không có trong nhãn thật bị đoán thừa (FP), trong cùng câu | Tất cả câu của split |

Năm nhãn hiếm lấy theo số mẫu dương **trên train**, hòa dùng ID nhãn; không chọn
nhãn nào đang có F1 test thấp. Nhóm partial có thể đồng thời có FP. Một câu có
thể thuộc hai hoặc ba nhóm; không cộng số câu/tỷ lệ của ba nhóm thành tổng lỗi.

Các nhóm này là quan hệ giữa nhãn thật và dự đoán, chưa phải lý do ngôn ngữ
của lỗi. Sau khi đọc văn bản, nhóm có thể ghi nhận phủ định, mỉa mai, thiếu ngữ
cảnh, nhiều cảm xúc hoặc nhãn chưa rõ, nhưng chỉ ghi khi có bằng chứng ở ví dụ.
Điểm số không tự chứng minh nguyên nhân; nhãn dataset cũng có thể chưa đầy đủ.

`pairs.csv` đếm **FN của nhãn A + FP của nhãn B trong cùng câu**. Đây không phải
bảng đồng xuất hiện nhãn thật, cũng không phải confusion matrix một lớp. Một
câu bị bỏ sót hai nhãn và đoán thừa hai nhãn đóng góp bốn cặp. `sample_count`
của từng cặp vẫn là số câu có cặp đó; tổng các cặp có thể lớn hơn số câu lỗi.

## 3. Lệnh chạy

Phân tích chính trên validation, threshold 0,5 cho cả ba kiến trúc:

```powershell
python -m scripts.analyze_project_errors --split validation
```

Mặc định seeds `42 123 2026`. Khi cần, khai báo đúng tập seed nhóm đã chạy:

```powershell
python -m scripts.analyze_project_errors --seeds 42 123 2026 --max-examples 8
```

Global/tuned là chẩn đoán thêm và phải có `final_protocol.json` của từng run
đại diện; script lấy ngưỡng đã khóa, không tự chọn ngưỡng lại:

```powershell
python -m scripts.analyze_project_errors --split validation --threshold-mode tuned
```

Trước phân tích test, khóa protocol từ validation cho từng checkpoint đại
diện và chạy `scripts.evaluate_transformer_test`. Công cụ đòi protocol và
`test_results.json` đã hoàn tất, kiểm checkpoint/seed/hash của tất cả ba mô hình
**trước khi mở split test**. Không tự suy luận hoặc tune bằng test.

```powershell
python -m scripts.analyze_project_errors --split test
python -m scripts.analyze_project_errors --split test --threshold-mode tuned
```

`--weighted` phân tích nhóm run weighted riêng; không trộn standard/weighted.
Mỗi lần ghi ra `reports/errors_<split>_<standard|weighted>_<fixed|global|tuned>`.
`--output` cho phép thư mục riêng. Chỉ dùng ngưỡng chọn trên validation để đánh
giá cuối; không quay lại thay cấu hình sau khi đọc lỗi test.

## 4. File bàn giao

- `counts.csv`: kiến trúc/seed/nhóm lỗi, số câu lỗi, số toàn split, số mẫu phù
  hợp định nghĩa, tỷ lệ trên toàn split và trong mẫu phù hợp. Mẫu số khác nhau
  giữa các nhóm; phải giữ cột mẫu số khi trích báo cáo.
- `rare_fn.csv`: năm nhãn hiếm, support train/evaluation, TP/FN/FP, P/R/F1 của
  checkpoint đại diện; số quyết định nhãn khác với số câu có lỗi.
- `pairs.csv`: cặp FN+FP, số câu, tổng FN của nhãn bị bỏ sót, tỷ lệ và ID ví dụ.
- `examples.csv`: ID, văn bản, nhãn thật, nhãn dự đoán, nhãn thiếu/thừa và điểm
  đủ 28 nhãn của mỗi mô hình. Mỗi nhóm lấy union câu lỗi rồi chọn ID tăng dần,
  dùng **cùng tập ID cho C1/C2/C3**. Một mô hình có `error_present=false` khi
  không mắc nhóm lỗi đó ở câu được chọn.
- `summary.md`: bảng so sánh có thể đọc và trích vào báo cáo.
- `manifest.json`: selection rule, ngưỡng, seed, nhãn hiếm, snapshot/hash và
  nguồn run; hỗ trợ đối chiếu kết quả.

`manual_linguistic_notes` ban đầu **trống**. Nhóm cần đọc và viết nhận xét bằng
lời của mình, không đưa các nhận xét tự suy đoán vào báo cáo. Không coi 5–8 ID
minh họa là toàn bộ split hoặc một mẫu ngẫu nhiên đại diện.

## 5. Cách viết mục phân tích lỗi trong báo cáo IEEE

1. Nêu split, ngưỡng, seed đại diện và quy tắc chọn trên validation.
2. Định nghĩa ba nhóm ở bảng trên; nêu rõ nhóm có thể chồng lấp.
3. Đưa bảng counts C1/C2/C3 cùng mẫu số; trích ít nhất một ID cụ thể mỗi nhóm.
4. Với từng ID, so nhãn thật, dự đoán và điểm của cả ba; ghi nhận mô hình nào
   khắc phục/bỏ sót, rồi bổ sung quan sát ngôn ngữ đã được người đọc kiểm.
5. Tách kết luận thống kê (đo được) khỏi giả thuyết nguyên nhân (chưa chứng minh).
6. Nếu ngưỡng tuned cải thiện, đối chiếu counts/F1 với fixed, cùng split và
   checkpoint. Tăng validation sau tuning vẫn cần được xác nhận bằng test.

Không điền số liệu hoặc ví dụ model giả khi các C chưa chạy xong. Chỉ trình bày
kết quả xuất từ artifact thực, đúng hash và đủ run theo yêu cầu.

## 6. Kiểm tra công cụ

```powershell
python -m unittest tests.test_project_errors -v
```

Tests dùng ma trận nhỏ và mock artifact để kiểm định nghĩa, cặp FN+FP, ID chung,
seed tie, chặn test thiếu protocol/hash sai trước khi mở dữ liệu. Không chạy GPU
và không chứng minh chất lượng mô hình.
