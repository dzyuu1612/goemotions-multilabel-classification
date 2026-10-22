"""Kiểm trùng dữ liệu và TF-IDF train-only; không train lại classifier hay đổi split.

python -m scripts.verification.audit_data_boundaries --artifacts-root PATH
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.datasets.goemotions import load_goemotions, multi_hot, sha256
from src.models.baseline import load_run_metadata
from src.paths import ROOT


def audit(root):
    frames, names, manifest = load_goemotions(root, write_metadata=False)
    quality, overlaps, baseline = {}, {}, {}
    for split, frame in frames.items():
        encoded = multi_hot(frame.labels.tolist(), len(names))
        quality[split] = {
            "rows": len(frame), "label_matrix_shape": list(encoded.shape),
            "id_duplicates": int(frame.id.duplicated().sum()),
            "missing_or_blank_text": int(frame.text.fillna("").str.strip().eq("").sum()),
            "exact_duplicate_rows_within_split": int(frame.text.duplicated().sum()),
            "positive_label_assignments": int(encoded.sum()),
        }
    for left, right in combinations(frames, 2):
        a, b = frames[left], frames[right]
        shared = set(a.text) & set(b.text)
        overlaps[f"{left}__{right}"] = {
            "shared_ids": len(set(a.id) & set(b.id)),
            "shared_exact_texts": len(shared),
            "affected_rows_left": int(a.text.isin(shared).sum()),
            "affected_rows_right": int(b.text.isin(shared).sum()),
        }
    # Refit only the vectorizer on official train, then compare the saved vocabulary/IDF.
    # No Logistic Regression fit and no use of validation/test in this reconstruction.
    reference = None
    for variant in ("standard", "balanced"):
        base = root / "data/processed/baseline"
        folder = base / ("full" if variant == "standard" else "balanced/full")
        load_run_metadata(folder, variant, names)
        vectorizer = joblib.load(folder / "model.joblib").named_steps["tfidf"]
        if reference is None:
            reference = clone(vectorizer).fit(frames["train"].text.tolist())
        same_vocabulary = vectorizer.vocabulary_ == reference.vocabulary_
        difference = float(np.max(np.abs(vectorizer.idf_ - reference.idf_))) if same_vocabulary else None
        before = vectorizer.idf_.copy()
        shapes = {split: list(vectorizer.transform(frames[split].text.iloc[:3]).shape)
                  for split in ("validation", "test")}
        baseline[variant] = {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "same_as_train_only_vocabulary": same_vocabulary,
            "max_abs_idf_difference_from_train_only": difference,
            "validation_test_transform_preserved_idf": bool(np.array_equal(before, vectorizer.idf_)),
            "three_row_transform_shapes": shapes,
        }
        if not same_vocabulary or difference > 1e-12 or not np.array_equal(before, vectorizer.idf_):
            raise ValueError(f"TF-IDF {variant} không khớp fit train-only")
    history_path = ROOT / "results/report_2_1/learning_curves.csv"
    history = pd.read_csv(history_path)
    curves = []
    for (architecture, seed), group in history.groupby(["architecture", "seed"]):
        group = group.sort_values("epoch")
        first, last = group.iloc[0], group.iloc[-1]
        curves.append({
            "architecture": architecture, "seed": int(seed), "epochs": int(last.epoch),
            "train_loss_first": float(first.train_loss), "train_loss_last": float(last.train_loss),
            "validation_macro_f1_first": float(first.validation_macro_f1),
            "validation_macro_f1_last": float(last.validation_macro_f1),
            "validation_macro_f1_best_epoch": int(group.loc[group.validation_macro_f1.idxmax(), "epoch"]),
            "validation_f1_strictly_increased_each_epoch": bool(np.all(np.diff(group.validation_macro_f1) > 0)),
        })
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read official splits; reconstruct train-only TF-IDF; inspect saved learning curves. No model training or threshold tuning.",
        "dataset_revision": manifest["revision"], "dataset_files": manifest["files"],
        "quality": quality, "cross_split_overlap": overlaps, "baseline_train_only": baseline,
        "learning_curves_sha256": sha256(history_path), "learning_curves": curves,
        "conclusion": {
            "saved_tfidf_matches_train_only": True,
            "cross_split_exact_text_overlap_present": any(v["shared_exact_texts"] for v in overlaps.values()) > 0,
            "all_nine_runs_best_at_last_epoch": all(v["epochs"] == v["validation_macro_f1_best_epoch"] for v in curves),
            "caveats": [
                "Exact-text overlap is contamination risk even when IDs are disjoint; this is not a certificate of zero leakage.",
                "Near duplicates, shared conversations/authors and pretraining contamination are not ruled out.",
                "Test labels have already been inspected during EDA and earlier reporting.",
                "Logs lack train evaluation F1 and per-epoch validation loss; absence of overfitting/underfitting cannot be established.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/presentation_audit/data_boundary_audit.json")
    args = parser.parse_args()
    result = audit(args.artifacts_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline_train_only"],
                      "overlap": result["cross_split_overlap"],
                      "conclusion": result["conclusion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
