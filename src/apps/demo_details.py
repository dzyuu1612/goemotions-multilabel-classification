"""Thông tin giao diện lấy từ run và ngưỡng đã lưu, không tạo số thực nghiệm mới."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from src.datasets.goemotions import REVISION, sha256
from src.evaluation.metrics import validate_thresholds
from src.models.baseline import load_run_metadata, load_thresholds
from src.paths import ROOT

MODES = [("Cố định 0,5", "fixed"), ("Ngưỡng chung từ validation", "global"),
         ("Ngưỡng riêng từng nhãn từ validation", "tuned")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_details(root, selection, key):
    root = Path(root).resolve()
    labels = read(root / "data/labels.json")
    if key.startswith("baseline_"):
        variant = key.removeprefix("baseline_")
        if variant not in ("standard", "balanced"):
            raise ValueError("Baseline không hợp lệ")
        folder = root / "data/processed/baseline" / ("full" if variant == "standard" else "balanced/full")
        meta = load_run_metadata(folder, variant, labels)
        limits = {mode: load_thresholds(folder, meta, mode) for _, mode in MODES}
        return dict(folder=folder, metadata=meta, limits=limits, labels=labels,
                    seed=42, system="A_" + variant)
    if key == "zero_shot":
        folder = root / "data/processed/zero_shot/full"
        meta = read(folder / "run_metadata.json")
        metadata_file = "run_metadata.json"
        system, seed = "B_bart_mnli", None
    elif key == "distilbert":
        folder = Path(os.environ.get("GOEMOTIONS_C3_RUN_ROOT", ROOT / "reports/c3_current_environment_2026-10-21"))
        chosen = read(folder / "selection.json")
        if chosen.get("selection_split") != "validation" or chosen.get("test_used") is not False:
            raise ValueError("C3 phải được chọn bằng validation")
        seed = chosen["representative_seed"]
        folder = folder / f"seed_{seed}"
        metadata_file, system = "run.json", "C_distilbert"
        meta = read(folder / metadata_file)
    elif key in ("bert", "roberta"):
        chosen = read(selection)
        if key == chosen["architecture"]:
            folder = (root / chosen["run_dir"]).resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Checkpoint nằm ngoài thư mục artifact")
            if sha256(folder / "run_metadata.json") != chosen["run_metadata_sha256"]:
                raise ValueError("Checkpoint mặc định đã thay đổi sau lựa chọn")
        else:
            candidates = []
            for candidate in (root / "data/processed/transformers" / key).glob("seed_*/full/standard"):
                score = read(candidate / "validation_metrics.json")["macro_f1"]
                seed = int(candidate.parents[1].name.removeprefix("seed_"))
                candidates.append((score, -seed, candidate))
            if not candidates:
                raise FileNotFoundError(f"Chưa có checkpoint {key}")
            folder = max(candidates)[2]
        metadata_file, system = "run_metadata.json", "C_" + key
        meta = read(folder / metadata_file)
        if meta.get("architecture") != key:
            raise ValueError("Sai kiến trúc checkpoint")
        seed = meta["seed"]
    else:
        raise ValueError("Mô hình không hợp lệ")
    protocol = read(folder / "final_protocol.json")
    if (meta.get("data_revision") != REVISION or meta.get("label_names") != labels
            or protocol.get("data_revision") != REVISION or protocol.get("label_names") != labels
            or protocol.get("run_metadata_sha256") != sha256(folder / metadata_file)
            or protocol.get("validation_scores_sha256") != sha256(folder / "validation_scores.npz")):
        raise ValueError("Ngưỡng không khớp run, dữ liệu hoặc thứ tự nhãn")
    if key in ("bert", "roberta", "distilbert"):
        published = ROOT / "results/report_2_1"
        record = f"receipts/{key}/seed_{seed}/final_protocol.json"
        if read(published / "manifest.json")["files_sha256"].get(record) != sha256(folder / "final_protocol.json"):
            raise ValueError("Checkpoint/ngưỡng không cùng bộ kết quả đang hiển thị")
    choices = protocol["configurations"]
    if sorted(c["threshold_mode"] for c in choices) != ["fixed", "global", "tuned"]:
        raise ValueError("Run chưa có đủ ba chế độ ngưỡng")
    limits = {c["threshold_mode"]: validate_thresholds(c["thresholds"], 28) for c in choices}
    if not np.all(np.asarray(limits["fixed"]) == 0.5):
        raise ValueError("Ngưỡng fixed không phải 0,5")
    return dict(folder=folder, metadata=meta, limits=limits, labels=labels, seed=seed, system=system)


def presentation(details, mode):
    if mode not in dict((key, label) for label, key in MODES):
        raise ValueError("Chế độ ngưỡng không hợp lệ")
    meta, seed = details["metadata"], details["seed"]
    cfg = meta.get("config", {})
    limits = np.broadcast_to(details["limits"][mode], (28,))
    threshold_text = (f"{limits[0]:.2f}" if np.all(limits == limits[0]) else
                      f"28 ngưỡng riêng, từ {limits.min():.2f} đến {limits.max():.2f}")
    title = (f"**Checkpoint đang dùng: seed {seed}**" if seed is not None else
             "**BART-MNLI pretrained — không fine-tune trên GoEmotions**")
    title += f" · Ngưỡng: **{threshold_text}**."
    params = [["Số nhãn đầu ra", "28"], ["Chế độ ngưỡng", dict((k,n) for n,k in MODES)[mode]]]
    if details["system"].startswith("C_"):
        params += [["Checkpoint pretrained", cfg.get("checkpoint", cfg.get("model_id", ""))],
                   ["Seed", str(seed)], ["Epoch đã chọn", str(meta.get("selected_epoch", meta.get("best_epoch")))],
                   ["Số epoch tối đa", str(cfg["epochs"])], ["Learning rate", f"{cfg['learning_rate']:.0e}"],
                   ["Batch size", str(cfg["batch_size"])], ["Giới hạn token", str(cfg["max_length"])],
                   ["Optimizer", "AdamW"], ["Weight decay", str(cfg["weight_decay"])],
                   ["Warmup ratio", str(cfg["warmup_ratio"])],
                   ["Số tham số", f"{meta.get('parameter_count', meta.get('trainable_parameters')):,}"],
                   ["Loss khi fine-tune", "BCEWithLogitsLoss không trọng số"]]
    elif details["system"].startswith("A_"):
        params += [["Biểu diễn", "TF-IDF unigram + bigram"], ["Số đặc trưng đã học", str(meta["n_tfidf_features"])],
                   ["min_df", str(cfg["min_df"])], ["max_features", str(cfg["max_features"])],
                   ["C của Logistic Regression", str(cfg["C"])], ["Solver", cfg["solver"]],
                   ["max_iter", str(cfg["max_iter"])], ["Class weight", str(cfg["class_weight"])]]
    else:
        params += [["Checkpoint", meta["checkpoint"]], ["Hypothesis template", meta["hypothesis_template"]],
                   ["Multi-label", "True"], ["Fine-tune GoEmotions", "Không"],
                   ["Giới hạn cặp NLI", "1.024 token"]]
    # Small published tables have raw-byte hashes; never substitute hard-coded scores.
    published = ROOT / "results/report_2_1"
    manifest = read(published / "manifest.json")
    records = {}
    for name in ("all_runs.csv", "mean_std.csv"):
        if sha256(published / name) != manifest["files_sha256"][name]:
            raise ValueError("Bảng kết quả công bố đã thay đổi")
        with (published / name).open(encoding="utf-8-sig", newline="") as stream:
            records[name] = list(csv.DictReader(stream))
    matching = [r for r in records["all_runs.csv"] if r["system"] == details["system"]
                and r["split"] == "test" and r["threshold_mode"] == mode
                and (seed is None or float(r["seed"]) == seed)]
    averages = [r for r in records["mean_std.csv"] if r["system"] == details["system"]
                and r["split"] == "test" and r["threshold_mode"] == mode]
    if len(matching) != 1 or len(averages) != 1:
        raise ValueError("Chưa có kết quả test tương ứng mô hình/seed/ngưỡng")
    row, avg = matching[0], averages[0]
    metrics = []
    for label, key in [("Macro-F1", "macro_f1"), ("Micro-F1", "micro_f1"),
                       ("Micro Precision", "micro_precision"), ("Micro Recall", "micro_recall"),
                       ("Hamming Loss ↓", "hamming_loss")]:
        summary = f"{float(avg[key+'_mean']):.4f}"
        if avg[key+"_std"]:
            summary += f" ± {float(avg[key+'_std']):.4f}"
        metrics.append([label, f"{float(row[key]):.4f}", summary])
    return title, params, metrics
