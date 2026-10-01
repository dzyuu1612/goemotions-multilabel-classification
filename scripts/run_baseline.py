"""Huấn luyện TF-IDF + Logistic Regression, chỉ đánh giá validation.

Chạy từ gốc repo: python -m scripts.run_baseline [--smoke] [--variant balanced]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.data import REVISION, load_goemotions, multi_hot
from src.metrics import evaluate_multilabel


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="GoEmotions TF-IDF + One-vs-Rest LR baseline")
    parser.add_argument("--smoke", action="store_true", help="Thử nhanh trên 5.000 train / 1.000 validation")
    parser.add_argument("--variant", choices=("standard", "balanced"), default="standard",
                        help="standard là baseline gốc; balanced thử trọng số lớp")
    args = parser.parse_args()

    # Chỉ mở train và validation; test dành cho lần đánh giá cuối sau khi khóa cấu hình.
    frames, labels, _ = load_goemotions(
        ROOT, write_metadata=False, splits=("train", "validation")
    )
    stored_labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))
    if labels != stored_labels:
        raise ValueError("data/labels.json không khớp mapping nhãn trong Parquet")
    train = frames["train"].iloc[:5000] if args.smoke else frames["train"]
    validation = frames["validation"].iloc[:1000] if args.smoke else frames["validation"]
    y_train = multi_hot(train["labels"].tolist(), len(labels))
    y_validation = multi_hot(validation["labels"].tolist(), len(labels))
    if train["id"].duplicated().any() or validation["id"].duplicated().any():
        raise ValueError("ID bị trùng trong train hoặc validation")

    class_weight = "balanced" if args.variant == "balanced" else None
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                                       class_weight=class_weight, random_state=42), n_jobs=1
                ),
            ),
        ]
    )
    started = perf_counter()
    model.fit(train["text"].tolist(), y_train)
    fit_seconds = perf_counter() - started

    predicted_at = perf_counter()
    scores = model.predict_proba(validation["text"].tolist())
    predict_seconds = perf_counter() - predicted_at
    if scores.shape != (len(validation), len(labels)):
        raise RuntimeError("Ma trận điểm không đúng N × 28")
    metrics = evaluate_multilabel(y_validation, scores, labels, threshold=0.5)

    output = ROOT / "data" / "processed" / "baseline"
    if args.variant == "balanced":
        output /= "balanced"
    output /= "smoke" if args.smoke else "full"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "validation_scores.npz",
        ids=validation["id"].to_numpy(dtype=str),
        scores=scores.astype(np.float32),
        label_names=np.asarray(labels, dtype=str),
    )
    joblib.dump(model, output / "model.joblib")
    result = {
        "method": "TF-IDF + OneVsRest LogisticRegression",
        "variant": args.variant,
        "data_revision": REVISION,
        "split_evaluated": "validation",
        "smoke": args.smoke,
        "n_train": len(train),
        "n_validation": len(validation),
        "fit_seconds": round(fit_seconds, 2),
        "predict_validation_seconds": round(predict_seconds, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "n_tfidf_features": len(model.named_steps["tfidf"].vocabulary_),
        "config": {"ngram_range": [1, 2], "min_df": 2, "max_features": 100_000,
                   "C": 1.0, "solver": "liblinear", "max_iter": 1000,
                   "class_weight": class_weight, "random_state": 42},
        "metrics": metrics,
    }
    (output / "validation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Trained {len(train):,} samples; evaluated {len(validation):,} validation samples")
    print(f"Macro-F1: {metrics['macro_f1']:.4f} | Micro-F1: {metrics['micro_f1']:.4f} | "
          f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Saved model and metrics in: {output.relative_to(ROOT).as_posix()}")
    print("Test metrics were not computed. Lock the configuration before using test labels.")


if __name__ == "__main__":
    main()
