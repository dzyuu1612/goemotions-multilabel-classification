"""Nhập một câu để xem baseline dự đoán 28 nhãn như thế nào.

Ví dụ: python -m scripts.predict_baseline --text "I am so grateful!"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc bằng baseline")
    parser.add_argument("--text", required=True, help="Một bình luận tiếng Anh")
    parser.add_argument("--variant", choices=("standard", "balanced"), default="standard")
    parser.add_argument("--threshold", choices=("fixed", "tuned"), default="fixed")
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("--text không được rỗng")
    folder = ROOT / "data" / "processed" / "baseline"
    if args.variant == "balanced":
        folder /= "balanced"
    folder /= "full"
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))
    # joblib chỉ được nạp từ file do chính nhóm tạo, không mở model từ nguồn lạ.
    model = joblib.load(folder / "model.joblib")
    thresholds = np.full(len(labels), 0.5)
    if args.threshold == "tuned":
        saved = json.loads((folder / "thresholds_validation.json").read_text(encoding="utf-8"))
        if saved["label_names"] != labels:
            raise ValueError("Thứ tự nhãn trong file ngưỡng không khớp")
        thresholds = np.asarray(saved["thresholds"], dtype=float)

    scores = model.predict_proba([args.text])[0]
    passed = np.flatnonzero(scores >= thresholds)
    print(f"Variant: {args.variant}; threshold mode: {args.threshold}")
    if len(passed):
        print("Predicted labels (score | threshold):")
        for j in sorted(passed, key=lambda j: -scores[j]):
            print(f"  {labels[j]:16s} {scores[j]:.3f} | {thresholds[j]:.2f}")
    else:
        print("No label passed its threshold. This is possible in multi-label prediction.")
    best = np.argsort(scores)[-5:][::-1]
    print("Top five scores, including scores below threshold:")
    for j in best:
        print(f"  {labels[j]:16s} {scores[j]:.3f}")


if __name__ == "__main__":
    main()
