# Hồ sơ đối chiếu yêu cầu cô — toàn đồ án GoEmotions

Cập nhật **09/10/2026**, dùng bằng chứng full/kiểm thử ngày 08/10.
Căn cứ: **ảnh đề tài 1** và **văn bản yêu cầu chung cô**
do Duy cung cấp. Bài nền tảng chỉ định là **GoEmotions, ACL 2020**, không tự thay
bằng bài khác chỉ để đạt khoảng năm 2022–2026. Nguồn kỹ thuật giải thích cách làm;
yêu cầu nộp/chấm lấy từ tài liệu cô.

**Cách dùng:** `[x]` bên dưới xác nhận phần kỹ thuật có artifact/bằng chứng;
không xác nhận điểm của cô, việc đã nộp hoặc hiểu/đóng góp của từng thành viên.
A/B/C đã full, **9/9 run C**, [summary](../reports/project_results/summary.json)
`complete=true`, `missing=[]`, **72 bản ghi/36 dòng tổng hợp**.
[Hồ sơ 95 JSON](../reports/reproducibility/README.md) giữ config/revision/hash/protocol.
Lần kiểm 08/10 có **69/69 tests**, notebook A12/B4/C7 cell mã thực thi PASS;
demo có bằng chứng suy luận và UI thật riêng. Không dùng smoke làm số cuối.

## 1. Phân công đúng nhóm 4 người

| Thành viên | Trách nhiệm chính |
|---|---|
| **Bảo Duy Nguyễn** | A cổ điển; điều phối B làm chung; bảng so sánh, nhãn hiếm và nâng cao |
| **Quốc Khánh** | C1 BERT cased; phần đầu và tổng hợp báo cáo |
| **Đức Trí — Thợ Săn Thập Cẩm** | C2 RoBERTa; hỗ trợ/bàn giao phần B từng nhận |
| **Nhật Huy** | C3 DistilBERT; D từ C thắng theo validation |

Ba người còn lại mỗi người phụ trách **trọn một kiến trúc C**. B được làm chung
hoặc gộp vào A theo yêu cầu cô. Phân công là trách nhiệm; bảng đóng góp cuối phải
ghi công việc/% công sức thực tế, không tự đặt mọi người 25% khi chưa thống kê.

## 2. Checklist yêu cầu và bằng chứng nghiệm thu

| Nghiệm thu | Yêu cầu cô | Cách triển khai và bằng chứng phải kiểm |
|---|---|---|
| [ ] | Đọc bài nền tảng, tóm tắt bằng lời nhóm | [Bài ACL](https://aclanthology.org/2020.acl-main.372/) và mục paper trong báo cáo đã có; từng thành viên vẫn cần đọc/tự giải thích |
| [x] | GoEmotions, 27 cảm xúc + neutral; official split | Manifest/labels/source: 43.410 train, 5.426 val, 5.427 test cùng revision/SHA |
| [x] | Khảo sát nhãn và tiền xử lý | EDA/EDA bổ sung, bảng/figures và `reports/verification.json`; phân bố, nhãn hiếm, độ dài, đồng xuất hiện, kiểm trùng |
| [x] | A cổ điển, 1 SV phụ trách | Duy theo phân công: TF-IDF + 28 OvR LR, hai biến thể và sáu cấu hình val/test; [bảng A](../reports/BASELINE_RESULTS.md) |
| [x] | B pretrained trực tiếp, KHÔNG fine-tune | Full BART-MNLI, 28 candidate labels, `multi_label=True`, SHA/template/batch manifest; val/test đủ 5.426/5.427 |
| [x] | C đủ 3 kiến trúc khác nhau | BERT cased/RoBERTa/DistilBERT; full model/scores/metadata trong hồ sơ 95 JSON và log full |
| [x] | Mỗi kiến trúc ≥3 random seed | 42/123/2026 cho từng C, 9 run full; không dùng smoke thay seed |
| [x] | Mean±std | [all_runs.csv](../reports/project_results/all_runs.csv), [mean_std.csv](../reports/project_results/mean_std.csv); sample std `ddof=1`, A/B một run để trống std |
| [x] | Xử lý ngưỡng đa nhãn | Fixed/global/tuned chọn trên val, protocol khóa trước test; giữ 28 scores/mapping |
| [x] | Macro/Micro-F1, P/R, Hamming | 72 bản ghi/36 dòng tổng hợp, micro/macro và per-label/support; val/test tách rõ |
| [x] | Cặp cảm xúc dễ nhầm | FN+FP cùng câu có ID/scores; bảng pairs test và A validation, không dùng co-occurrence thay lỗi |
| [x] | ≥3 loại lỗi so C1/C2/C3 | [Summary](../reports/errors_test_standard_fixed/summary.md): partial_multi_label, rare_false_negative, missed_extra_pair; cùng ID. Nhóm cần tự đọc/giải thích nguyên nhân ngôn ngữ |
| [x] | Nâng cao OR | A class weighting + threshold tuning; C threshold tuning val/test. C weighted chỉ là tùy chọn code, không ghi đã có 9 weighted run |
| [x] | F1 nhãn hiếm trước/sau | 5 nhãn chọn từ train, [rare_before_after.csv](../reports/project_results/rare_before_after.csv) và [bảng A](../reports/BASELINE_RESULTS.md); giữ delta âm/zero và support |
| [x] | D dùng best trong 3 C | BERT cased seed123, kiến trúc chọn bằng mean Macro-F1 val@0,5; checkpoint chọn trên val, không chọn bằng test |
| [x] | Demo có ảnh thật và hoạt động | [Suy luận](../reports/demo_verification.json) ba câu/28 scores/input edge; [UI](../reports/demo_ui/evidence.json) HTTP200/28 hàng; [PNG](../reports/demo_ui/demo_ui.png) |
| [x] | Bằng chứng độ ổn định C | Ba seed/mean std/per-label/lỗi có số thật; wall time có ngủ máy nên không dùng để xếp tốc độ kiến trúc |
| [ ] | Hai báo cáo tiến độ + báo cáo cuối đã nộp | Các tệp Word/PDF có bên dưới; nhóm kiểm bản cuối, thông tin hành chính/đóng góp và xác nhận nộp |

Yêu cầu nâng cao là lựa chọn **OR**, không yêu cầu nhóm làm cả weighting,
threshold tuning và contrastive. Tài liệu cô không giới hạn nâng cao chỉ ở C;
A là bằng chứng hợp lệ về phương pháp nếu đánh giá và báo đúng. Nâng cao không
thay thế nghĩa vụ ba kiến trúc C × ba seed và demo D.

**Đánh đổi phải trình bày:** A balanced global Macro-F1 test 0,4530 > balanced
tuned 0,4493; standard tuned Micro-F1 test 0,5330 > balanced tuned 0,5277. Lựa chọn
A giữ `balanced_tuned` từ val. Năm nhãn hiếm test tăng so standard fixed nhưng
tuning sau weighting giảm 4/5 nhãn; grief standard vẫn F1=0. Không chỉ chọn hàng đẹp.

**Nhóm còn tự xác nhận:** đọc/giải thích paper/code/lỗi; công việc và% đóng góp
thực tế; giảng viên/lớp/MSSV; rà nguồn/nội dung, đúng bản báo cáo đã nộp và bảo vệ.
Các mục kỹ thuật PASS không xác nhận những việc con người này.

## 3. Các quy tắc kỹ thuật bắt buộc giữ đúng

1. A/B/C nhận văn bản cùng split và nhãn. **Output A không làm input C**;
   kết quả A làm mốc so sánh. C tự tokenization và fine-tune từ train.
2. Giữ đủ 28 nhãn tiếng Anh, không gộp thành 7 nhãn Ekman trong benchmark chính.
   Multi-hot float/BCE cho C; sigmoid khi xuất điểm, không argmax/softmax.
3. TF-IDF/weighting học từ train; checkpoint/ngưỡng chọn bằng val. Test mở sau
   khóa protocol; không sửa cấu hình để chọn bằng test.
4. Checkpoint C1 `google-bert/bert-base-cased`, không ghi nhầm uncased. Tham khảo
   batch hiệu dụng 16/lr 5e-5/4 epoch; khai báo optimizer, max_length và khác biệt
   của nhóm. Không tự gọi mọi thí nghiệm là “tái lập chính xác bài báo”.
5. B @0,5 giữ trọng số không fine-tune; B tuned dùng nhãn validation để hiệu chỉnh
   ngưỡng nên phải ghi rõ bước dùng nhãn này.
6. Model/scores/thresholds phải cùng SHA, seed, revision và mapping; ghép scores
   theo ID. `neutral` không tự thêm khi tất cả score dưới ngưỡng.
7. Macro-F1 0,49 không có nghĩa 49% câu dự đoán đúng toàn bộ. Hamming thấp không
   tự chứng minh tốt khi số nhãn âm áp đảo. Tuned-val có thể lạc quan.
8. Case ngôn ngữ như mỉa mai/phủ định cần đọc câu thật để xác nhận. Các nhóm lỗi
   tự động có thể chồng lấp; không cộng tỷ lệ thành 100%.

## 4. Nội dung từng báo cáo theo tài liệu cô

### Báo cáo tiến độ lần 1

- Tóm tắt paper/mục tiêu bằng lời nhóm.
- Thống kê dữ liệu và ví dụ.
- Kết quả **A và B có số cụ thể**, không chỉ “đang làm”.
- Mỗi người C ghi môi trường, số seed thực sự đã chạy, lỗi/vướng mắc.
- Phân công bốn vai trò và mức hoàn thành thực tế.

### Báo cáo tiến độ lần 2

- C đầy đủ ba kiến trúc × ba seed; mean±std so với A/B.
- ≥3 nhóm lỗi, ví dụ và so sánh giữa các C.
- Demo đang chạy, kèm link/ảnh/video.
- Kế hoạch hoặc kết quả bước đầu nâng cao.

### Báo cáo cuối

- Toàn bộ nội dung trên đã hoàn thiện.
- Nâng cao có số liệu và F1 nhãn hiếm trước/sau.
- C thắng/thua, lý do có bằng chứng, ý nghĩa std và giới hạn diễn giải.
- Đóng góp thực tế từng người (% công sức và phần việc).
- Mã nguồn đầy đủ, hướng dẫn, checkpoint bàn giao, link/minh chứng demo.
- Hạn chế, hướng phát triển, chuẩn bị câu hỏi phản biện.

Mẫu cô dùng để giữ bố cục và hình thức học phần; trích dẫn số `[n]`/danh mục nguồn
dùng chuẩn IEEE. Tên học phần/giảng viên/lớp/MSSV phải đúng thông tin nhóm, không
chép thông tin của báo cáo mẫu. Nội dung sáu chương của báo cáo học phần không
bắt buộc trở thành bài hội nghị hai cột chỉ vì dùng trích dẫn IEEE.

Tệp để kiểm/nộp: tiến độ1 [Word](../reports/BAO_CAO_TIEN_DO_1.docx)/[PDF](../reports/BAO_CAO_TIEN_DO_1.pdf),
tiến độ2 [Word](../reports/BAO_CAO_TIEN_DO_2.docx)/[PDF](../reports/BAO_CAO_TIEN_DO_2.pdf),
báo cáo cuối [Word](../reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.docx)/[PDF](../reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.pdf).
Đối chiếu số liệu bản xuất với artifacts trước nộp; sự tồn tại tệp không xác nhận đã nộp.

## 5. Hồ sơ artifact cần có để đánh dấu đạt

| Phần | Hồ sơ full cần kiểm |
|---|---|
| Dữ liệu/EDA | `data/manifest.json`, labels, SHA; `reports/verification.json`, bảng/figures và notebook |
| A | `data/processed/baseline/{full,balanced/full}/`; scores/model/metadata/thresholds; `final_protocol.json`, `final/` và bảng A |
| B | `data/processed/zero_shot/full/`; run metadata, validation/test scores, batch manifest, protocol và test metadata |
| Mỗi C/seed | `data/processed/transformers/{architecture}/seed_{seed}/full/standard/`; checkpoint, mapping, scores, metadata/history, protocol và test results |
| Mean/std | `data/processed/transformers/seed_summary.json`, `reports/project_results/{all_runs.csv,mean_std.csv,summary.json,RESULTS.md}` |
| Lỗi | Output `analyze_project_errors`: manifest, counts, pairs, examples cùng ID ba C; bổ sung nhận xét thủ công |
| D | `selected_model.json` gắn hash run, `app.py`, protocol ngưỡng nếu dùng; ảnh/video/link app đang chạy |
| Báo cáo/nguồn | `reports/BAO_CAO_DO_AN_NOI_DUNG.md`, `references_ieee.json`, DOCX/PDF xuất cuối, bảng đóng góp và báo cáo tiến độ |

Chạy `python -m scripts.summarize_project --require-complete` để kiểm bảng thực
nghiệm. Hiện `complete=true`, `missing=[]`, đủ 9 run C/72 bản ghi/36 dòng tổng hợp.
Nếu tái chạy làm artifacts thiếu/hỏng, ghi đúng vấn đề; không điền điểm dự kiến.
Lệnh này không thay nghiệm thu thủ công về giải thích paper/lỗi/nguồn/đóng góp.
Proof kiểm mã: [verification_project.json](../reports/verification_project.json),
[notebook execution](../reports/execution/notebook_verification.json). Hồ sơ mới
cần đối chiếu theo ngày/phiên bản; số 69 tests là lần kiểm 08/10.

Case Study 4 đã đối chiếu với Jay Lee, *Industrial AI* (2020), mục 4.2.3.1,
trang in 82–88 (PDF 98–104), DOI `10.1007/978-981-15-2144-7`.
Xem mục **5.6** trong [báo cáo](../reports/BAO_CAO_DO_AN_NOI_DUNG.md) và
[nguồn Springer](https://link.springer.com/book/10.1007/978-981-15-2144-7).
Đây là ví dụ tiết kiệm năng lượng cho thiết bị dịch vụ nhà máy, dùng để liên hệ
vai trò dữ liệu → mô hình → quyết định công nghiệp, không phải thí nghiệm NLP.
Liên hệ đầu ra GoEmotions với quyết định chăm sóc khách hàng là đề xuất của nhóm;
không chuyển mức tiết kiệm/ROI của nhà máy thành ROI GoEmotions chưa được đo.
