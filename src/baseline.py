"""Các hàm nhỏ dùng chung cho phân tích baseline đa nhãn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def load_aligned_scores(path, expected_ids, expected_labels):
    """Đọc điểm dự đoán và ghép theo ID, không dựa vào thứ tự dòng của file."""
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        scores = saved["scores"].astype(float)
        labels = saved["label_names"].astype(str).tolist()
    wanted_ids = np.asarray(expected_ids, dtype=str)
    if labels != list(expected_labels):
        raise ValueError("Thứ tự 28 nhãn của file điểm không khớp dữ liệu")
    if scores.shape != (len(ids), len(labels)):
        raise ValueError("Ma trận điểm không có kích thước N × số nhãn")
    if len(set(ids)) != len(ids) or len(set(wanted_ids)) != len(wanted_ids):
        raise ValueError("ID bị trùng, không thể ghép an toàn")
    position = {sample_id: i for i, sample_id in enumerate(ids)}
    if set(position) != set(wanted_ids):
        raise ValueError("Tập ID trong file điểm khác tập ID của split cần đánh giá")
    aligned = scores[[position[sample_id] for sample_id in wanted_ids]]
    if not np.isfinite(aligned).all() or np.any((aligned < 0) | (aligned > 1)):
        raise ValueError("Điểm dự đoán phải hữu hạn trong [0, 1]")
    return aligned


def tune_thresholds(y_true, scores):
    """Chọn ngưỡng F1 cho từng nhãn trên validation, không xem test.

    Lưới cố định 0,05..0,95; khi hòa chọn ngưỡng gần 0,5 nhất. Đây là
    ước lượng trên cùng validation dùng để chọn ngưỡng, nên có thể lạc quan.
    """
    truth = np.asarray(y_true)
    probabilities = np.asarray(scores)
    if truth.ndim != 2 or probabilities.shape != truth.shape:
        raise ValueError("y_true và scores phải cùng kích thước N × số nhãn")
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    chosen = []
    for label_id in range(truth.shape[1]):
        candidates = []
        for threshold in grid:
            predicted = probabilities[:, label_id] >= threshold
            score = f1_score(truth[:, label_id], predicted, zero_division=0)
            candidates.append((score, -abs(threshold - 0.5), threshold))
        chosen.append(float(max(candidates)[2]))
    return np.asarray(chosen)
