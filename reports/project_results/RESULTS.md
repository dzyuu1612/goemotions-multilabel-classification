# Kết quả thí nghiệm thực tế

Std mẫu ddof=1; dấu — nghĩa là không áp dụng hoặc chưa đủ ba seed.

| Hệ thống | Split | Ngưỡng | Số run | Macro-F1 mean±std | Micro-F1 mean±std | Hamming Loss |
|---|---|---|---:|---:|---:|---:|
| A_standard | validation | fixed | 1 | 0.2025 | 0.3760 | 0.0354 |
| A_standard | validation | global | 1 | 0.4094 | 0.5100 | 0.0544 |
| A_standard | validation | tuned | 1 | 0.4391 | 0.5427 | 0.0433 |
| A_balanced | validation | fixed | 1 | 0.4562 | 0.5099 | 0.0532 |
| A_balanced | validation | global | 1 | 0.4660 | 0.5176 | 0.0473 |
| A_balanced | validation | tuned | 1 | 0.4901 | 0.5467 | 0.0443 |

**Bảng 5-3. Precision/Recall theo cùng split và cấu hình.**

| Hệ thống | Split | Ngưỡng | Macro-P | Macro-R | Micro-P | Micro-R |
|---|---|---|---:|---:|---:|---:|
| A_standard | validation | fixed | 0.5571 | 0.1436 | 0.7254 | 0.2538 |
| A_standard | validation | global | 0.4578 | 0.4455 | 0.4102 | 0.6741 |
| A_standard | validation | tuned | 0.5408 | 0.4513 | 0.4876 | 0.6119 |
| A_balanced | validation | fixed | 0.3858 | 0.5801 | 0.4158 | 0.6592 |
| A_balanced | validation | global | 0.4142 | 0.5470 | 0.4524 | 0.6047 |
| A_balanced | validation | tuned | 0.4874 | 0.5219 | 0.4796 | 0.6356 |

## Phần chưa có bằng chứng đầy đủ

- A test
- B full validation + frozen protocol
- C bert seed 42 frozen thresholds
- C bert seed 42 test
- C bert seed 123 full
- C bert seed 2026 full
- C roberta seed 42 full
- C roberta seed 123 full
- C roberta seed 2026 full
- C distilbert seed 42 full
- C distilbert seed 123 full
- C distilbert seed 2026 full
- D selected_model from 3 architectures × 3 seeds
