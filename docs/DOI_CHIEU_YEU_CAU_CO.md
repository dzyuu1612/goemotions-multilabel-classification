# Hồ sơ đối chiếu yêu cầu cô — toàn đồ án GoEmotions

Cập nhật **08/10/2026**. Căn cứ: **ảnh đề tài 1** và **văn bản yêu cầu chung cô**
do Duy cung cấp. Bài nền tảng chỉ định là **GoEmotions, ACL 2020**, không tự thay
bằng bài khác chỉ để đạt khoảng năm 2022–2026. Nguồn kỹ thuật giải thích cách làm;
yêu cầu nộp/chấm lấy từ tài liệu cô.

**Cách dùng:** đánh dấu nghiệm thu sau khi mở bằng chứng và kiểm số thật.
“Code có hỗ trợ” không phải “thực nghiệm full đã đạt”. Các mục B/C/D bên dưới
**cần log full** và artifact cuối; tài liệu này không tự chứng nhận kết quả chưa có.

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
| [ ] | Đọc bài nền tảng, tóm tắt bằng lời nhóm | [Bài ACL](https://aclanthology.org/2020.acl-main.372/), phần paper trong [báo cáo](../reports/BAO_CAO_DO_AN_NOI_DUNG.md); nguồn/đóng góp/taxonomy/BERT |
| [ ] | GoEmotions, 27 cảm xúc + neutral; official split | `data/manifest.json`, `data/labels.json`, `src/data.py`; 43.410 train/5.426 val/5.427 test, cùng snapshot/SHA |
| [ ] | Khảo sát nhãn và tiền xử lý | `notebooks/eda.ipynb`, `eda_extra.ipynb`, `reports/THONG_KE_DU_LIEU.md`; phân bố, nhãn hiếm, độ dài, đồng xuất hiện và kiểm trùng |
| [ ] | A cổ điển, 1 SV | Duy: `scripts/run_baseline.py`; TF-IDF + 28 OvR LR; full model/scores/config; [bảng A validation](../reports/BASELINE_RESULTS.md) đã có, test kiểm riêng |
| [ ] | B pretrained trực tiếp, KHÔNG fine-tune | `scripts/run_zero_shot.py`; BART-MNLI pipeline, đủ 28 candidate labels, `multi_label=True`; **cần log full**, SHA/template, scores và metrics |
| [ ] | C đủ cả 3 kiến trúc khác nhau | BERT **cased**, RoBERTa, DistilBERT; `src/neural.py`, `scripts/train_transformer.py`; **cần log full** cả ba |
| [ ] | Mỗi kiến trúc ít nhất 3 random seed | Kế hoạch dùng 42/123/2026; `run_transformer_seeds.py`; **cần log full 9 run**, không thay bằng 3 run tổng hoặc 1 seed mỗi C |
| [ ] | Báo mean ± std, không chỉ một số | `select_best_transformer.py`, `summarize_project.py`; từng seed + mean/sample std `ddof=1`; **cần đủ run full**, không tự tạo std cho A/B một run |
| [ ] | Xử lý ngưỡng đa nhãn | Scores sigmoid/28 điểm, ngưỡng @0,5 và tuned/global trên validation; `src/baseline.py`, `src/experiment.py`; protocol trước test |
| [ ] | Macro/Micro-F1, Precision/Recall, Hamming Loss | `src/metrics.py`; đủ 28 nhãn; P/R micro+macro, per-label/support; bảng validation và test tách rõ |
| [ ] | Phân tích cặp cảm xúc dễ nhầm | FN nhãn A + FP nhãn B trong cùng câu, ID/văn bản/nhãn thật/score; `label_error_pairs`; không gọi co-occurrence là nhầm lẫn |
| [ ] | Ít nhất 3 loại lỗi, đối chiếu C1/C2/C3 | `scripts/analyze_project_errors.py`, [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md); **cần scores full** và ví dụ cùng ID, nhận xét thủ công về nguyên nhân |
| [ ] | Nâng cao: weighting hoặc ngưỡng riêng hoặc contrastive | A balanced/threshold tuning đã có bằng chứng validation; C có `--weighted` và protocol thresholds; báo trước/sau trên cùng split/seed/config |
| [ ] | Cải thiện F1 nhãn hiếm | Chọn nhóm hiếm theo **train**; per-label F1/support/delta trên val và test; cần số thật, kể cả nhãn không cải thiện |
| [ ] | D Gradio/Streamlit dùng best trong 3 C | `app.py`, `selected_model.json`; **cần đủ log full C và app chạy thật**; không dùng A hoặc B thay C |
| [ ] | Demo có link hoặc video/ảnh | Hướng dẫn chạy, checkpoint/tokenizer/ngưỡng đúng model, ảnh/video thực tế; input rỗng/dài/no-label được xử lý |
| [ ] | Giải thích C tốt/kém, độ ổn định | Mean/std + per-label/lỗi/tài nguyên thực đo; không kết luận “vì may mắn” từ một seed |
| [ ] | Đủ 2 báo cáo tiến độ + báo cáo cuối | Đối chiếu mục 4 bên dưới; kết quả đầy đủ, nguồn IEEE, đóng góp, mã/demo, hạn chế và phản biện |

Yêu cầu nâng cao là lựa chọn **OR**, không yêu cầu nhóm làm cả weighting,
threshold tuning và contrastive. Tài liệu cô không giới hạn nâng cao chỉ ở C;
A là bằng chứng hợp lệ về phương pháp nếu đánh giá và báo đúng. Nâng cao không
thay thế nghĩa vụ ba kiến trúc C × ba seed và demo D.

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
nghiệm. Nếu còn `missing`, ghi đúng mục thiếu và **cần log full**; không điền điểm
dự kiến. Lệnh này không thay phần nghiệm thu thủ công về giải thích paper, lỗi,
nguồn, đóng góp và demo đang chạy.

Case Study 4 đã đối chiếu với Jay Lee, *Industrial AI* (2020), mục 4.2.3.1,
trang in 82–88 (PDF 98–104), DOI `10.1007/978-981-15-2144-7`.
Xem mục **5.6** trong [báo cáo](../reports/BAO_CAO_DO_AN_NOI_DUNG.md) và
[nguồn Springer](https://link.springer.com/book/10.1007/978-981-15-2144-7).
Đây là ví dụ tiết kiệm năng lượng cho thiết bị dịch vụ nhà máy, dùng để liên hệ
vai trò dữ liệu → mô hình → quyết định công nghiệp, không phải thí nghiệm NLP.
Liên hệ đầu ra GoEmotions với quyết định chăm sóc khách hàng là đề xuất của nhóm;
không chuyển mức tiết kiệm/ROI của nhà máy thành ROI GoEmotions chưa được đo.
