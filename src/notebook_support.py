"""Đường dẫn và kiểm chứng dùng trong notebook, không huấn luyện lại âm thầm.

Source luôn ở ROOT; artifacts đọc từ GOEMOTIONS_ARTIFACTS_ROOT hoặc bản chạy
cục bộ cạnh repo. Các phép suy luận mới chỉ ghi output vào reports/ của source.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.goemotions import load_goemotions, multi_hot
from src.evaluation.metrics import evaluate_multilabel
from src.evaluation.protocols import read_json, validate_protocol
from src.models.baseline import load_aligned_scores
from src.models.transformer import METRIC_NAMES, load_transformer_run
from src.paths import ROOT


def artifacts_root():
    """Biến môi trường có ưu tiên; không gán nhầm thư mục code thành dataset."""
    configured = os.environ.get("GOEMOTIONS_ARTIFACTS_ROOT")
    if configured:
        folder = Path(configured).expanduser().resolve()
        if not (folder / "data/labels.json").is_file():
            raise FileNotFoundError(f"Thiếu data/labels.json trong artifacts: {folder}")
        return folder
    sibling = ROOT.parent / "goemotions-multilabel-classification"
    if sibling != ROOT and (sibling / "data/processed/transformers/selected_model.json").is_file():
        return sibling.resolve()
    return ROOT


@lru_cache(maxsize=4)
def benchmark(root):
    """Kiểm 72 dòng/36 nhóm từ scores gốc, dùng lại kết quả trong cùng kernel."""
    from scripts.verification.reproduce_report import verify_scores
    return verify_scores(Path(root))


def replay_test(root, system, seed=42):
    """Suy luận TEST thật từ model đã fit; kiểm cả scores lẫn ba ngưỡng đã khóa.

    Trả bảng metric mới và sai lệch với artifact báo cáo. Không dùng test tìm
    ngưỡng/checkpoint và không ghi đè weights hoặc scores lịch sử.
    """
    return replay_split(root, system, seed, split="test")


def replay_split(root, system, seed=42, split="test"):
    """Replay validation/test from the same checked benchmark without retraining."""
    if split not in ("validation", "test"):
        raise ValueError("Replay chỉ dùng validation hoặc test")
    root = Path(root)
    result, frames, labels = benchmark(str(root.resolve()))
    frame = frames[split]
    if system.startswith("A_"):
        import joblib
        variant = system.removeprefix("A_")
        if variant not in ("standard", "balanced"):
            raise ValueError("Baseline phải là standard hoặc balanced")
        base = root / "data/processed/baseline"
        folder = base / ("full" if variant == "standard" else "balanced/full")
        model = joblib.load(folder / "model.joblib")
        scores = model.predict_proba(frame.text.tolist())
        reference = (base / f"final/{variant}_test_scores.npz" if split == "test"
                     else folder / "validation_scores.npz")
        choices = [c for c in read_json(base / "final_protocol.json")["configurations"] if c["variant"] == variant]
    elif system.startswith("C_"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from scripts.transformers.train_transformer import encoded_dataset, predict_scores, trim_padding_collate
        from src.models.transformer import resolve_device
        torch.set_num_threads(4)
        architecture = system.removeprefix("C_")
        folder = root / f"data/processed/transformers/{architecture}/seed_{seed}/full/standard"
        metadata = load_transformer_run(folder, require_full=True)
        protocol = read_json(folder / "final_protocol.json")
        validate_protocol(folder, protocol, labels)
        device = resolve_device("auto")
        tokenizer = AutoTokenizer.from_pretrained(folder / "checkpoint", local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(folder / "checkpoint", local_files_only=True).to(device)
        data, _ = encoded_dataset(frame, tokenizer, labels, metadata["config"]["max_length"])
        loader = torch.utils.data.DataLoader(data, batch_size=metadata["config"]["batch_size"],
                                             shuffle=False, collate_fn=trim_padding_collate)
        scores = predict_scores(model, loader, device)
        reference, choices = folder / f"{split}_scores.npz", protocol["configurations"]
        del model, loader, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        raise ValueError("replay_test dành cho hai baseline A và các checkpoint C")
    previous = load_aligned_scores(reference, frame.id.tolist(), labels)
    truth = multi_hot(frame.labels.tolist(), len(labels))
    rows = []
    for choice in choices:
        metric = evaluate_multilabel(truth, scores, labels, choice["thresholds"])
        old = next(r for r in result["records"] if r["system"] == system and r["seed"] == seed
                   and r["split"] == split and r["threshold_mode"] == choice["threshold_mode"])
        difference = max(abs(metric[k] - old[k]) for k in METRIC_NAMES)
        if difference > 1e-12:
            raise ValueError(f"Suy luận mới khác metric báo cáo: {system}/{seed}/{choice['threshold_mode']}")
        rows.append({"system": system, "seed": seed, "split": split,
                     "threshold_mode": choice["threshold_mode"], "n_samples": len(frame),
                     **{k: metric[k] for k in METRIC_NAMES},
                     "metric_max_abs_diff": difference,
                     "score_max_abs_diff": float(np.max(np.abs(scores - previous)))})
    output = ROOT / "reports/notebook_replay"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{system}_{seed}_{split}.npz", ids=frame.id.to_numpy(dtype=str),
                        scores=scores, label_names=np.asarray(labels, dtype=str))
    return pd.DataFrame(rows)


def metrics_from_confusion(counts, n_samples):
    """Tính lại 7 metric từ TP/FP/FN/TN, không dựng lại scores/checkpoint bị thiếu."""
    values = counts[["tp", "fp", "fn", "tn"]].to_numpy(dtype=float)
    if (not len(values) or not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
            or not np.all(values.sum(axis=1) == n_samples)):
        raise ValueError("Confusion counts không hợp lệ hoặc khác số mẫu")
    tp, fp, fn, _ = values.T

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) != 0)

    return {"macro_precision": float(divide(tp, tp + fp).mean()),
            "macro_recall": float(divide(tp, tp + fn).mean()),
            "macro_f1": float(divide(2 * tp, 2 * tp + fp + fn).mean()),
            "micro_precision": float(divide(tp.sum(), (tp + fp).sum())),
            "micro_recall": float(divide(tp.sum(), (tp + fn).sum())),
            "micro_f1": float(divide(2 * tp.sum(), (2 * tp + fp + fn).sum())),
            "hamming_loss": float((fp + fn).sum() / (n_samples * len(values)))}
