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
| A_standard | test | fixed | 1 | 0.1963 | 0.3800 | 0.0348 |
| A_standard | test | global | 1 | 0.4096 | 0.5047 | 0.0553 |
| A_standard | test | tuned | 1 | 0.4134 | 0.5330 | 0.0444 |
| A_balanced | test | fixed | 1 | 0.4441 | 0.5024 | 0.0547 |
| A_balanced | test | global | 1 | 0.4530 | 0.5157 | 0.0480 |
| A_balanced | test | tuned | 1 | 0.4493 | 0.5277 | 0.0467 |
| B_bart_mnli | validation | fixed | 1 | 0.1060 | 0.1027 | 0.5307 |
| B_bart_mnli | validation | global | 1 | 0.1502 | 0.1467 | 0.2839 |
| B_bart_mnli | validation | tuned | 1 | 0.1633 | 0.1762 | 0.2955 |
| B_bart_mnli | test | fixed | 1 | 0.1035 | 0.1008 | 0.5315 |
| B_bart_mnli | test | global | 1 | 0.1473 | 0.1449 | 0.2831 |
| B_bart_mnli | test | tuned | 1 | 0.1609 | 0.1752 | 0.2941 |
| C_bert | validation | fixed | 3 | 0.4713 ± 0.0066 | 0.5807 ± 0.0042 | 0.0322 ± 0.0005 |
| C_bert | validation | global | 3 | 0.4975 ± 0.0052 | 0.5858 ± 0.0055 | 0.0375 ± 0.0027 |
| C_bert | validation | tuned | 3 | 0.5290 ± 0.0060 | 0.5993 ± 0.0016 | 0.0365 ± 0.0004 |
| C_bert | test | fixed | 3 | 0.4720 ± 0.0045 | 0.5820 ± 0.0040 | 0.0318 ± 0.0003 |
| C_bert | test | global | 3 | 0.4929 ± 0.0084 | 0.5848 ± 0.0072 | 0.0376 ± 0.0027 |
| C_bert | test | tuned | 3 | 0.5038 ± 0.0097 | 0.5900 ± 0.0029 | 0.0373 ± 0.0003 |
| C_roberta | validation | fixed | 3 | 0.4234 ± 0.0108 | 0.5767 ± 0.0033 | 0.0300 ± 0.0000 |
| C_roberta | validation | global | 3 | 0.4793 ± 0.0084 | 0.5983 ± 0.0135 | 0.0396 ± 0.0046 |
| C_roberta | validation | tuned | 3 | 0.5072 ± 0.0180 | 0.6144 ± 0.0047 | 0.0355 ± 0.0012 |
| C_roberta | test | fixed | 3 | 0.4219 ± 0.0088 | 0.5803 ± 0.0009 | 0.0295 ± 0.0001 |
| C_roberta | test | global | 3 | 0.4697 ± 0.0092 | 0.5940 ± 0.0158 | 0.0402 ± 0.0048 |
| C_roberta | test | tuned | 3 | 0.4872 ± 0.0156 | 0.6048 ± 0.0057 | 0.0365 ± 0.0014 |
| C_distilbert | validation | fixed | 3 | 0.4064 ± 0.0060 | 0.5712 ± 0.0038 | 0.0302 ± 0.0003 |
| C_distilbert | validation | global | 3 | 0.4749 ± 0.0081 | 0.5868 ± 0.0076 | 0.0421 ± 0.0028 |
| C_distilbert | validation | tuned | 3 | 0.5077 ± 0.0071 | 0.6019 ± 0.0021 | 0.0369 ± 0.0007 |
| C_distilbert | test | fixed | 3 | 0.4116 ± 0.0035 | 0.5731 ± 0.0017 | 0.0297 ± 0.0001 |
| C_distilbert | test | global | 3 | 0.4646 ± 0.0045 | 0.5836 ± 0.0096 | 0.0428 ± 0.0030 |
| C_distilbert | test | tuned | 3 | 0.4866 ± 0.0093 | 0.5935 ± 0.0047 | 0.0377 ± 0.0010 |

**Bảng 5-3. Precision/Recall theo cùng split và cấu hình.**

| Hệ thống | Split | Ngưỡng | Macro-P | Macro-R | Micro-P | Micro-R |
|---|---|---|---:|---:|---:|---:|
| A_standard | validation | fixed | 0.5571 | 0.1436 | 0.7254 | 0.2538 |
| A_standard | validation | global | 0.4578 | 0.4455 | 0.4102 | 0.6741 |
| A_standard | validation | tuned | 0.5408 | 0.4513 | 0.4876 | 0.6119 |
| A_balanced | validation | fixed | 0.3858 | 0.5801 | 0.4158 | 0.6592 |
| A_balanced | validation | global | 0.4142 | 0.5470 | 0.4524 | 0.6047 |
| A_balanced | validation | tuned | 0.4874 | 0.5219 | 0.4796 | 0.6356 |
| A_standard | test | fixed | 0.6128 | 0.1396 | 0.7383 | 0.2558 |
| A_standard | test | global | 0.4631 | 0.4476 | 0.4025 | 0.6766 |
| A_standard | test | tuned | 0.4384 | 0.4361 | 0.4744 | 0.6082 |
| A_balanced | test | fixed | 0.3777 | 0.5696 | 0.4043 | 0.6631 |
| A_balanced | test | global | 0.4029 | 0.5363 | 0.4447 | 0.6135 |
| A_balanced | test | tuned | 0.4372 | 0.4883 | 0.4561 | 0.6260 |
| B_bart_mnli | validation | fixed | 0.0652 | 0.8936 | 0.0553 | 0.7229 |
| B_bart_mnli | validation | global | 0.0998 | 0.7635 | 0.0839 | 0.5810 |
| B_bart_mnli | validation | tuned | 0.0990 | 0.7903 | 0.0998 | 0.7524 |
| B_bart_mnli | test | fixed | 0.0635 | 0.8935 | 0.0542 | 0.7148 |
| B_bart_mnli | test | global | 0.0960 | 0.7614 | 0.0829 | 0.5759 |
| B_bart_mnli | test | tuned | 0.0972 | 0.7872 | 0.0992 | 0.7499 |
| C_bert | validation | fixed | 0.5515 ± 0.0274 | 0.4291 ± 0.0017 | 0.6398 ± 0.0088 | 0.5316 ± 0.0014 |
| C_bert | validation | global | 0.4825 ± 0.0275 | 0.5303 ± 0.0268 | 0.5483 ± 0.0313 | 0.6313 ± 0.0309 |
| C_bert | validation | tuned | 0.5413 ± 0.0173 | 0.5434 ± 0.0038 | 0.5559 ± 0.0051 | 0.6502 ± 0.0073 |
| C_bert | test | fixed | 0.5578 ± 0.0255 | 0.4300 ± 0.0033 | 0.6421 ± 0.0062 | 0.5322 ± 0.0055 |
| C_bert | test | global | 0.4884 ± 0.0117 | 0.5275 ± 0.0231 | 0.5440 ± 0.0310 | 0.6345 ± 0.0286 |
| C_bert | test | tuned | 0.5207 ± 0.0115 | 0.5214 ± 0.0071 | 0.5446 ± 0.0038 | 0.6437 ± 0.0054 |
| C_roberta | validation | fixed | 0.5913 ± 0.0025 | 0.3685 ± 0.0105 | 0.7085 ± 0.0040 | 0.4863 ± 0.0066 |
| C_roberta | validation | global | 0.4461 ± 0.0290 | 0.5492 ± 0.0391 | 0.5252 ± 0.0433 | 0.7002 ± 0.0411 |
| C_roberta | validation | tuned | 0.5241 ± 0.0453 | 0.5230 ± 0.0057 | 0.5658 ± 0.0148 | 0.6726 ± 0.0103 |
| C_roberta | test | fixed | 0.5674 ± 0.0032 | 0.3740 ± 0.0070 | 0.7115 ± 0.0062 | 0.4901 ± 0.0041 |
| C_roberta | test | global | 0.4331 ± 0.0292 | 0.5466 ± 0.0323 | 0.5164 ± 0.0431 | 0.7041 ± 0.0373 |
| C_roberta | test | tuned | 0.4997 ± 0.0412 | 0.5139 ± 0.0043 | 0.5517 ± 0.0157 | 0.6697 ± 0.0095 |
| C_distilbert | validation | fixed | 0.5801 ± 0.0046 | 0.3455 ± 0.0051 | 0.7088 ± 0.0043 | 0.4783 ± 0.0037 |
| C_distilbert | validation | global | 0.4712 ± 0.0275 | 0.5517 ± 0.0273 | 0.5000 ± 0.0242 | 0.7120 ± 0.0257 |
| C_distilbert | validation | tuned | 0.5217 ± 0.0084 | 0.5308 ± 0.0116 | 0.5507 ± 0.0082 | 0.6637 ± 0.0070 |
| C_distilbert | test | fixed | 0.5912 ± 0.0050 | 0.3535 ± 0.0027 | 0.7124 ± 0.0025 | 0.4794 ± 0.0014 |
| C_distilbert | test | global | 0.4317 ± 0.0175 | 0.5505 ± 0.0240 | 0.4917 ± 0.0248 | 0.7198 ± 0.0225 |
| C_distilbert | test | tuned | 0.4990 ± 0.0082 | 0.5179 ± 0.0090 | 0.5385 ± 0.0103 | 0.6613 ± 0.0038 |

## Phần chưa có bằng chứng đầy đủ

Đủ A/B/C, ba seed mỗi kiến trúc, test sau khóa protocol và lựa chọn mô hình C.
