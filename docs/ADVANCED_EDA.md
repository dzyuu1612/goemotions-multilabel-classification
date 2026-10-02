## Tổng Kết Đóng Góp Kỹ Thuật, Quản Lý Phiên Bản (Git Commit) & Quy Trình Bàn Giao

1. **Kiểm toán rò rỉ dữ liệu chéo split (Cross-Split Leakage):** Trích xuất và phát hiện 32 mẫu bình luận trùng lặp giữa Train và Test, chỉ ra tỷ lệ bất đồng nhãn lên tới 40.62% do tính chủ quan của người gán nhãn mạng xã hội[cite: 4].
2. **Ánh xạ hệ phân loại tâm lý (Valence Taxonomy):** Gom cụm 28 nhãn cảm xúc thành 4 nhóm phân cực cảm xúc (Positive 39.18%, Negative 21.78%, Neutral 28.42%, Ambiguous 10.01%) theo bài báo nền tảng của Demszky et al. (ACL 2020)[cite: 4, 6].
3. **Tính toán sẵn vector trọng số lớp (`pos_weight`):** Xuất bảng tra cứu hệ số trọng số nguyên bản (`raw_pos_weight`) và trọng số làm mượt (`smooth_pos_weight`) trên 43.410 mẫu Train, tạo tiền đề kiểm soát hiện tượng mất cân bằng cực đoan (184.66×) mà không gây nổ gradient khi huấn luyện[cite: 4].
4. **Định lượng tương quan đa nhãn ($Y_{train}^T Y_{train}$):** Xây dựng ma trận đồng xuất hiện nhãn $28 \times 28$, chứng minh sự tồn tại của 1.396 mẫu chứa đồng thời nhãn *neutral* với cảm xúc khác và xác lập các cụm cảm xúc song hành (*admiration* + *gratitude*, *amusement* + *joy*)[cite: 4].
5. **Khai phá đặc thù ngôn ngữ Reddit:** Thống kê sự hiện diện của các thẻ ẩn danh (`[NAME]`, `[RELIGION]`) và chứng minh có tới 21.6% bình luận chứa từ phủ định (*not*, *no*, *never*, *n't*)[cite: 4].
6. **Benchmark phân vị độ dài chuỗi Tokenizer:** Phân tích thực nghiệm qua tokenizer `bert-base-uncased` và chứng minh ngưỡng `max_length = 64` bao phủ tới 99.8% dữ liệu, tạo cơ sở khoa học để cắt giảm hơn 50% chi phí VRAM self-attention[cite: 4].

Toàn bộ kết quả định lượng đã được tôi kết xuất thành các bảng số liệu sạch trong `reports/tables/` và hệ thống biểu đồ trực quan độ phân giải cao trong `reports/figures/`[cite: 4, 7].

Để "đóng băng" các tài nguyên nền tảng này, đảm bảo tính tái lập thực nghiệm và bàn giao các ràng buộc kỹ thuật quan trọng cho các thành viên khác trước khi bước vào giai đoạn huấn luyện mô hình, toàn bộ quy trình quản lý phiên bản (Git commit) và phân phối nhiệm vụ được chuẩn hóa như sau:

* **`notebooks/eda.ipynb`** (hoặc notebook phân tích nâng cao tương đương): Trước khi commit, bắt buộc thực hiện **Restart Kernel & Run All** để đảm bảo toàn bộ biểu đồ, bảng dữ liệu và log thực thi được hiển thị đầy đủ, không bị lỗi gãy cell.
* **`reports/ADVANCED_EDA_AUDIT.md`**: Lưu trữ toàn bộ kết quả phân tích chuyên sâu, phát hiện dữ liệu và cơ sở lý thuyết định hướng cho các giai đoạn tiếp theo.

#### 2. Dữ liệu Dạng Bảng (`reports/tables/*.csv`)

Tất cả các bảng tra cứu và thống kê rút gọn (chỉ vài KB) đều được đưa vào theo dõi phiên bản:

* `cross_split_leakage_analysis.csv`: Danh sách các mẫu trùng chéo giữa Train – Test và trạng thái nhất quán nhãn.
* `class_weights_train.csv`: Vector trọng số lớp (`raw_pos_weight` và `smooth_pos_weight`) của 28 nhãn trích xuất từ tập Train[cite: 1].
* `valence_taxonomy_distribution.csv`: Phân bố và tỷ trọng của 4 nhóm phân cực cảm xúc (Positive, Negative, Ambiguous, Neutral)[cite: 1].
* `label_co_occurrence_matrix.csv`: Ma trận đồng xuất hiện nhãn $28 \times 28$ trên tập Train[cite: 1].
* `reddit_lexical_features.csv`: Tần suất và tỷ lệ các đặc trưng từ vựng/cú pháp Reddit (từ phủ định, all caps, tag ẩn danh, dấu câu)[cite: 1].
* `sequence_length_percentiles.csv`: Bảng phân vị so sánh độ dài số từ thô và số BERT token[cite: 1].

#### 3. Biểu đồ Trực quan (`reports/figures/*.png`)

* `04_label_co_occurrence_heatmap.png`: Heatmap tương quan đồng xuất hiện giữa các cảm xúc[cite: 1].
* `05_bert_token_length_distribution.png`: Histogram phân phối token BERT và vạch mốc `max_length = 64`[cite: 1].
* `06_valence_taxonomy_pie.png`: Biểu đồ tròn thể hiện tỷ trọng 4 cụm phân cực cảm xúc[cite: 1].

> **Lưu ý về `.gitignore`:** Tuyệt đối không commit các file trọng số mô hình lớn (`*.pt`, `*.bin`, `*.safetensors`, `checkpoints/`), thư mục môi trường ảo (`venv/`, `__pycache__/`) hoặc cache dữ liệu của Hugging Face (`.cache/`).

```bash
# 1. Thêm các bảng kết quả và hình ảnh trực quan
git add reports/tables/*.csv reports/figures/*.png

# 2. Thêm notebook và tài liệu báo cáo
git add notebooks/eda.ipynb reports/ADVANCED_EDA_AUDIT.md

# 3. Tạo commit với thông điệp rõ ràng theo chuẩn Conventional Commits
git commit -m "feat(eda): audit dataset leakage, export class weights, and token length distribution"

# 4. Đẩy lên repository chung
git push origin <ten-branch>
```
