# Hồ sơ tái hiện A/B/C

Đã xuất **20 JSON**; **2/9** run C full hoàn tất được kiểm.

Mỗi JSON là bản copy nguyên byte; đối chiếu source/export SHA-256 trong `manifest.json`. Config, seed, môi trường, labels, data/model revision và history nằm trong metadata gốc.

Git HEAD/source hashes chỉ mô tả repo tại lúc export; trainer chưa ghi training commit nên không suy ra commit đã huấn luyện.

| Nhóm | Trạng thái | Checkpoint | Model revision | Seed |
|---|---|---|---|---|
| A/standard | complete | — | — | 42 |
| A/balanced | complete | — | — | 42 |
| B | incomplete | — | — | — |
| C/bert/seed_42 | complete | google-bert/bert-base-cased | cd5ef92a9fb2f889e972770a36d4ed042daf221e | 42 |
| C/bert/seed_123 | complete | google-bert/bert-base-cased | cd5ef92a9fb2f889e972770a36d4ed042daf221e | 123 |
| C/bert/seed_2026 | incomplete | — | — | — |
| C/roberta/seed_42 | incomplete | — | — | — |
| C/roberta/seed_123 | incomplete | — | — | — |
| C/roberta/seed_2026 | incomplete | — | — | — |
| C/distilbert/seed_42 | incomplete | — | — | — |
| C/distilbert/seed_123 | incomplete | — | — | — |
| C/distilbert/seed_2026 | incomplete | — | — | — |

Các file có status khác exported được ghi rõ trong manifest; không tạo số liệu thay thế. Model/scores dùng để suy luận cần bản đầy đủ khớp hash trong metadata. Hướng dẫn: `docs/REPRODUCIBILITY.md` ở gốc repo.