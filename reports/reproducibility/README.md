# Hồ sơ tái hiện A/B/C

Đã xuất **95 JSON**; **9/9** run C full hoàn tất được kiểm.

Mỗi JSON là bản copy nguyên byte; đối chiếu source/export SHA-256 trong `manifest.json`. Config, seed, môi trường, labels, data/model revision và history nằm trong metadata gốc.

Git HEAD/source hashes chỉ mô tả repo tại lúc export; trainer chưa ghi training commit nên không suy ra commit đã huấn luyện.

| Nhóm | Trạng thái | Checkpoint | Model revision | Seed |
|---|---|---|---|---|
| A/standard | complete | — | — | 42 |
| A/balanced | complete | — | — | 42 |
| B | complete | facebook/bart-large-mnli | d7645e127eaf1aefc7862fd59a17a5aa8558b8ce | None |
| C/bert/seed_42 | complete | google-bert/bert-base-cased | cd5ef92a9fb2f889e972770a36d4ed042daf221e | 42 |
| C/bert/seed_123 | complete | google-bert/bert-base-cased | cd5ef92a9fb2f889e972770a36d4ed042daf221e | 123 |
| C/bert/seed_2026 | complete | google-bert/bert-base-cased | cd5ef92a9fb2f889e972770a36d4ed042daf221e | 2026 |
| C/roberta/seed_42 | complete | FacebookAI/roberta-base | e2da8e2f811d1448a5b465c236feacd80ffbac7b | 42 |
| C/roberta/seed_123 | complete | FacebookAI/roberta-base | e2da8e2f811d1448a5b465c236feacd80ffbac7b | 123 |
| C/roberta/seed_2026 | complete | FacebookAI/roberta-base | e2da8e2f811d1448a5b465c236feacd80ffbac7b | 2026 |
| C/distilbert/seed_42 | complete | distilbert/distilbert-base-uncased | 12040accade4e8a0f71eabdb258fecc2e7e948be | 42 |
| C/distilbert/seed_123 | complete | distilbert/distilbert-base-uncased | 12040accade4e8a0f71eabdb258fecc2e7e948be | 123 |
| C/distilbert/seed_2026 | complete | distilbert/distilbert-base-uncased | 12040accade4e8a0f71eabdb258fecc2e7e948be | 2026 |

Các file có status khác exported được ghi rõ trong manifest; không tạo số liệu thay thế. Model/scores dùng để suy luận cần bản đầy đủ khớp hash trong metadata. Hướng dẫn: `docs/REPRODUCIBILITY.md` ở gốc repo.