"""Đánh giá test một lần SAU KHI nhóm đã khóa variant/ngưỡng bằng validation.

Ví dụ: python -m scripts.evaluate_baseline_test --variant standard \
            --threshold fixed --confirm-frozen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np

from src.data import REVISION, load_goemotions, multi_hot
from src.metrics import evaluate_multilabel


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Final test cho baseline đã khóa")
    parser.add_argument("--variant", choices=("standard", "balanced"), required=True)
    parser.add_argument("--threshold", choices=("fixed", "tuned"), required=True)
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="Xác nhận nhóm đã chọn variant/ngưỡng trước khi xem test")
    args = parser.parse_args()
    if not args.confirm_frozen:
        parser.error("Cần --confirm-frozen sau khi nhóm đã chốt cấu hình bằng validation")

    folder = ROOT / "data" / "processed" / "baseline"
    if args.variant == "balanced":
        folder /= "balanced"
    folder /= "full"
    final_root = ROOT / "data" / "processed" / "baseline" / "final"
    output = final_root / f"{args.variant}_{args.threshold}"
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError("Baseline đã có một lần đánh giá test. "
                              "Không chạy thêm cấu hình khác để chọn số đẹp.")

    model_path = folder / "model.joblib"
    config = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
    if config["data_revision"] != REVISION or config["smoke"] or config["variant"] != args.variant:
        raise ValueError("Model/config không phải bản full của variant đã chọn")
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))
    thresholds = np.full(len(labels), 0.5)
    threshold_hash = None
    if args.threshold == "tuned":
        threshold_path = folder / "thresholds_validation.json"
        saved = json.loads(threshold_path.read_text(encoding="utf-8"))
        if saved["label_names"] != labels or saved["data_revision"] != REVISION:
            raise ValueError("File ngưỡng không khớp nhãn/revision")
        thresholds = np.asarray(saved["thresholds"], dtype=float)
        threshold_hash = sha256(threshold_path)

    # Chỉ ở bước cuối này mới mở split test và nhãn test.
    frames, test_labels, _ = load_goemotions(ROOT, write_metadata=False, splits=("test",))
    if test_labels != labels:
        raise ValueError("Thứ tự nhãn test không khớp mapping đã huấn luyện")
    test = frames["test"]
    truth = multi_hot(test["labels"].tolist(), len(labels))
    model = joblib.load(model_path)  # Chỉ nạp model do nhóm tạo.
    scores = model.predict_proba(test["text"].tolist())
    result = evaluate_multilabel(truth, scores, labels, threshold=thresholds)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = None

    output.mkdir(parents=True)
    np.savez_compressed(output / "test_scores.npz", ids=test["id"].to_numpy(dtype=str),
                        scores=scores.astype(np.float32), label_names=np.asarray(labels, dtype=str))
    record = {
        "split": "test", "variant": args.variant, "threshold_mode": args.threshold,
        "data_revision": REVISION, "git_head": commit, "working_tree_dirty": dirty,
        "model_sha256": sha256(model_path), "threshold_file_sha256": threshold_hash,
        "metrics": result,
    }
    (output / "test_metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Test Macro-F1: {result['macro_f1']:.4f}; Micro-F1: {result['micro_f1']:.4f}")
    print(f"Final test artifacts: {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
