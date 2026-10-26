"""Tính lại số báo cáo, không huấn luyện lại hoặc tìm ngưỡng bằng test.

python -m scripts.verification.reproduce_report --artifacts-root ../goemotions-multilabel-classification

Các bước: kiểm dữ liệu/hash -> ghép scores theo ID -> áp ngưỡng đã khóa ->
tính metric -> đối chiếu 72 dòng/36 nhóm -> kiểm bảng Word nếu được chỉ định.
--inference-c chạy suy luận test thật cho cả 9 checkpoint; không trả cache.
Output mới luôn nằm trong reports/reproduction, không sửa artifacts gốc.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from scripts.analysis.summarize_project import SEEDS, summarize
from src.datasets.goemotions import EXPECTED_ROWS, REVISION, load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel
from src.evaluation.protocols import read_json, validate_protocol
from src.models.baseline import load_aligned_scores, load_run_metadata
from src.models.transformer import (ARCHITECTURES, METRIC_NAMES, load_demo_selection,
                                    load_transformer_run, summarize_architectures)
from src.paths import ROOT

TOLERANCE = 1e-12


def check(condition, message):
    """Không dùng assert: python -O cũng phải chặn kết quả sai."""
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def record_key(row):
    seed = row.get("seed")
    seed = None if seed in (None, "") else int(float(seed))
    return row["system"], seed, row["split"], row["threshold_mode"]


def compare_metrics(actual, expected, context):
    differences = {key: abs(actual[key] - float(expected[key])) for key in METRIC_NAMES}
    check(all(np.isfinite(d) and d <= TOLERANCE for d in differences.values()),
          f"Metric khác báo cáo: {context}: {differences}")
    return max(differences.values())


def compare_averages(records, expected):
    """Gộp lại từ từng seed thật, dùng sample std ddof=1; A/B không có std."""
    actual = summarize(records)
    key = lambda row: (row["system"], row["split"], row["threshold_mode"])
    lookup = {key(row): row for row in expected}
    check(len(lookup) == len(expected) == len(actual) == 36, "Bảng tổng hợp phải có đúng 36 nhóm")
    for row in actual:
        other = lookup.get(key(row))
        check(other is not None and row["n_runs"] == int(other["n_runs"]), "Sai nhóm/số seed tổng hợp")
        for metric in METRIC_NAMES:
            for suffix in ("_mean", "_std"):
                value, saved = row[metric + suffix], other[metric + suffix]
                if value is None:
                    check(saved in (None, ""), "Không tạo std cho một run A/B")
                else:
                    check(np.isfinite(float(saved)) and abs(value - float(saved)) <= TOLERANCE,
                          f"Sai tổng hợp {key(row)} {metric + suffix}")
    return actual


def verify_scores(artifact_root):
    """Chỉ đọc cache có sẵn; thiếu split thì dừng trước khi loader có thể tải mạng."""
    artifact_root = Path(artifact_root).resolve()
    for split in EXPECTED_ROWS:
        path = artifact_root / f"data/raw/{split}-00000-of-00001.parquet"
        check(path.is_file(), f"Thiếu dữ liệu cục bộ: {path}")
    frames, labels, manifest = load_goemotions(artifact_root, write_metadata=False)
    check(read_json(artifact_root / "data/labels.json") == labels, "Mapping khác dữ liệu gốc")
    truth = {split: multi_hot(frame.labels.tolist(), len(labels)) for split, frame in frames.items()}
    expected = read_csv(artifact_root / "reports/project_results/all_runs.csv")
    lookup = {record_key(row): row for row in expected}
    check(len(lookup) == len(expected) == 72, "Bảng gốc phải có 72 dòng duy nhất")
    records, histories, input_hashes, differences = [], [], {}, []

    def measure(system, seed, split, path, choices):
        scores = load_aligned_scores(path, frames[split].id.tolist(), labels)
        input_hashes[path.relative_to(artifact_root).as_posix()] = sha256(path)
        for choice in choices:
            result = evaluate_multilabel(truth[split], scores, labels, choice["thresholds"])
            row = {"system": system, "seed": seed, "split": split,
                   "threshold_mode": choice["threshold_mode"], "n_samples": len(frames[split]),
                   **{key: result[key] for key in METRIC_NAMES}}
            saved = lookup.get(record_key(row))
            check(saved is not None and int(saved["n_samples"]) == row["n_samples"], "Sai dòng/split báo cáo")
            differences.append(compare_metrics(result, saved, record_key(row)))
            if "validation_metrics" in choice and split == "validation":
                compare_metrics(result, choice["validation_metrics"], f"protocol {system} {seed}")
            records.append(row)

    # Bước 1: A có hai model; cả sáu cấu hình phải thuộc đúng protocol đã khóa.
    baseline = artifact_root / "data/processed/baseline"
    protocol = read_json(baseline / "final_protocol.json")
    check(protocol["data_revision"] == REVISION and protocol["label_names"] == labels, "Sai protocol A")
    final = read_json(baseline / "final/final_results.json")
    check(final["protocol_sha256"] == sha256(baseline / "final_protocol.json"), "A test thuộc protocol khác")
    for variant in ("standard", "balanced"):
        folder = baseline / ("full" if variant == "standard" else "balanced/full")
        metadata = load_run_metadata(folder, variant, labels)
        frozen = protocol["runs"][variant]
        check(metadata["artifact_sha256"] == frozen["artifact_sha256"] and
              sha256(folder / "thresholds_validation.json") == frozen["threshold_file_sha256"], "A đã đổi sau khóa")
        choices = [row for row in protocol["configurations"] if row["variant"] == variant]
        check({row["threshold_mode"] for row in choices} == {"fixed", "global", "tuned"} and len(choices) == 3,
              "Thiếu/trùng cấu hình A")
        for split in ("validation", "test"):
            path = folder / "validation_scores.npz" if split == "validation" else baseline / f"final/{variant}_test_scores.npz"
            measure("A_" + variant, 42, split, path, choices)
    print("PASS A: 12 dòng từ scores thật", flush=True)

    # Bước 2: B không fine-tune; giữ nguyên scores NLI và ba ngưỡng validation.
    folder = artifact_root / "data/processed/zero_shot/full"
    protocol = read_json(folder / "final_protocol.json")
    validate_protocol(folder, protocol, labels)
    test = read_json(folder / "test_run_metadata.json")
    check(test["validation_run_metadata_sha256"] == protocol["run_metadata_sha256"], "Sai run B test")
    check(test["protocol_sha256"] == sha256(folder / "final_protocol.json"), "B test thuộc protocol khác")
    for name, expected_hash in test["artifact_sha256"].items():
        path = (folder / name).resolve()
        check(path.is_relative_to(folder.resolve()) and sha256(path) == expected_hash, f"B test sai hash {name}")
    for split in ("validation", "test"):
        measure("B_bart_mnli", None, split, folder / f"{split}_scores.npz", protocol["configurations"])
    print("PASS B: 6 dòng từ scores thật", flush=True)

    # Bước 3: C phải đủ 3 kiến trúc × 3 seed, đúng weights/tokenizer/scores/hash.
    runs_by_architecture = {architecture: [] for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            folder = artifact_root / f"data/processed/transformers/{architecture}/seed_{seed}/full/standard"
            metadata = load_transformer_run(folder, require_full=True)
            runs_by_architecture[architecture].append(metadata)
            protocol = read_json(folder / "final_protocol.json")
            validate_protocol(folder, protocol, labels)
            check(metadata["architecture"] == architecture and metadata["seed"] == seed, "Sai thư mục run C")
            saved = read_json(folder / "test_results.json")
            check(saved["protocol_sha256"] == sha256(folder / "final_protocol.json") and
                  saved["scores_sha256"] == sha256(folder / "test_scores.npz"), "Sai hash C test")
            for split in ("validation", "test"):
                measure("C_" + architecture, seed, split, folder / f"{split}_scores.npz", protocol["configurations"])
            histories.append({"architecture": architecture, "seed": seed,
                              "selected_epoch": metadata["selected_epoch"], "config": metadata["config"],
                              "model_revision": metadata["model_revision"], "history": metadata["history"]})
            print(f"PASS C: {architecture} seed {seed}", flush=True)
    check({record_key(row) for row in records} == set(lookup) and len(records) == 72, "Bảng kết quả chưa đủ")
    averages = compare_averages(records, read_csv(artifact_root / "reports/project_results/mean_std.csv"))
    selection_path = artifact_root / "data/processed/transformers/selected_model.json"
    _, selected, _ = load_demo_selection(artifact_root, selection_path)
    # Chọn lại bằng validation để kiểm selection; không dùng test đổi người thắng.
    ranking = summarize_architectures(runs_by_architecture)
    winner = "C_" + ranking[0]["architecture"]
    best_seed = max((row for row in records if row["system"] == winner and
                     row["split"] == "validation" and row["threshold_mode"] == "fixed"),
                    key=lambda row: (row["macro_f1"], -row["seed"]))["seed"]
    check("C_" + selected["architecture"] == winner and selected["seed"] == best_seed, "Selection khác validation")
    return {"records": records, "averages": averages, "histories": histories,
            "data_manifest": manifest, "score_hashes": input_hashes,
            "metric_max_absolute_difference": max(differences),
            "selection": {"architecture": selected["architecture"], "seed": selected["seed"]}}, frames, labels


def verify_word_tables(path, result):
    """Đối chiếu trực tiếp 9 bảng kết quả trong báo cáo cuối kỳ hiện có."""
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in document.findall(".//w:tbl", namespace):
        tables.append([["".join(t.text or "" for t in cell.findall(".//w:t", namespace))
                        for cell in row.findall("w:tc", namespace)] for row in table.findall("w:tr", namespace)])
    records = {record_key(row): row for row in result["records"]}
    averages = {(r["system"], r["split"], r["threshold_mode"]): r for r in result["averages"]}
    architectures = {"C1": "bert", "C2": "roberta", "C3": "distilbert"}
    a_split, sections, cells = iter(("validation", "test")), [], 0

    def compare(cell, value):
        nonlocal cells
        check(cell.strip() == value, f"Bảng Word khác số tính lại: {cell!r} != {value!r}")
        cells += 1

    def averaged(system, split, mode, metric):
        row = averages[system, split, mode]
        return f"{row[metric + '_mean']:.4f} ± {row[metric + '_std']:.4f}"

    for table in tables:
        if not table:
            continue
        header = table[0]
        if header == ["Cấu hình A", "Macro F1", "Micro F1", "Hamming"]:
            split = next(a_split)
            for row, (variant, mode) in zip(table[1:], ((v, m) for v in ("standard", "balanced") for m in ("fixed", "global", "tuned"))):
                for cell, metric in zip(row[1:], ("macro_f1", "micro_f1", "hamming_loss")):
                    compare(cell, f"{records['A_' + variant, 42, split, mode][metric]:.4f}")
        elif header in (["Luật ngưỡng", "Macro F1", "Micro F1", "Hamming"], ["B", "Macro F1", "Micro F1", "Hamming"]):
            split = "validation" if header[0] == "Luật ngưỡng" else "test"
            for row, mode in zip(table[1:], ("fixed", "global", "tuned")):
                for cell, metric in zip(row[1:], ("macro_f1", "micro_f1", "hamming_loss")):
                    compare(cell, f"{records['B_bart_mnli', None, split, mode][metric]:.4f}")
        elif header == ["C", "Seed", "Epoch", "Val F", "Test F", "Test T"]:
            for row in table[1:]:
                system, seed = "C_" + architectures[row[0]], int(row[1])
                history = next(r for r in result["histories"] if r["architecture"] == architectures[row[0]] and r["seed"] == seed)
                check(int(row[2]) == history["selected_epoch"], "Word sai epoch")
                for cell, (split, mode) in zip(row[3:], (("validation", "fixed"), ("test", "fixed"), ("test", "tuned"))):
                    compare(cell, f"{records[system, seed, split, mode]['macro_f1']:.4f}")
        elif header == ["C", "Val Macro F1", "Test Macro F1", "Test Micro F1"]:
            for row in table[1:]:
                for cell, (split, metric) in zip(row[1:], (("validation", "macro_f1"), ("test", "macro_f1"), ("test", "micro_f1"))):
                    compare(cell, averaged("C_" + architectures[row[0]], split, "fixed", metric))
        elif header in (["C", "Macro F1", "Micro F1", "Hamming"], ["C", "P fixed", "R fixed"], ["C", "P tuned", "R tuned"]):
            mode = "fixed" if header[1] == "P fixed" else "tuned"
            metrics = ("macro_f1", "micro_f1", "hamming_loss") if header[1] == "Macro F1" else ("micro_precision", "micro_recall")
            for row in table[1:]:
                for cell, metric in zip(row[1:], metrics):
                    compare(cell, averaged("C_" + architectures[row[0]], "test", mode, metric))
        else:
            continue
        sections.append(header)
    check(len(sections) == 9 and cells == 111, f"Không nhận đủ 9 bảng/111 ô của báo cáo: {len(sections)}/{cells}")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "tables": len(sections), "metric_cells": cells}


def inference_test(artifact_root, output, frames, labels, records, *, all_c=False, baseline=False, device="auto"):
    """Suy luận lại vào output mới; metric phải khớp, điểm float được báo sai lệch."""
    results = []
    lookup = {record_key(row): row for row in records}

    def compare(system, seed, scores, saved_path, choices):
        saved = load_aligned_scores(saved_path, frames["test"].id.tolist(), labels)
        truth = multi_hot(frames["test"].labels.tolist(), len(labels))
        for choice in choices:
            metric = evaluate_multilabel(truth, scores, labels, choice["thresholds"])
            compare_metrics(metric, lookup[system, seed, "test", choice["threshold_mode"]], f"inference {system} {seed}")
        destination = output / f"{system}_seed_{seed}_test_scores.npz"
        np.savez_compressed(destination, ids=frames["test"].id.to_numpy(dtype=str),
                            label_names=np.asarray(labels, dtype=str), scores=scores)
        results.append({"system": system, "seed": seed, "n_samples": len(scores),
                        "score_max_absolute_difference": float(np.max(np.abs(scores - saved))),
                        "metric_max_absolute_difference": max(compare_metrics(
                            evaluate_multilabel(truth, scores, labels, c["thresholds"]),
                            lookup[system, seed, "test", c["threshold_mode"]], system) for c in choices),
                        "scores_sha256": sha256(destination)})
        print(f"PASS inference test {system} seed {seed}", flush=True)

    if baseline:
        import joblib
        base = artifact_root / "data/processed/baseline"
        protocol = read_json(base / "final_protocol.json")
        for variant in ("standard", "balanced"):
            folder = base / ("full" if variant == "standard" else "balanced/full")
            model = joblib.load(folder / "model.joblib")
            scores = model.predict_proba(frames["test"].text.tolist())
            compare("A_" + variant, 42, scores, base / f"final/{variant}_test_scores.npz",
                    [c for c in protocol["configurations"] if c["variant"] == variant])
    if all_c:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from scripts.transformers.train_transformer import encoded_dataset, predict_scores, trim_padding_collate
        from src.models.transformer import resolve_device
        torch.set_num_threads(4)
        target = resolve_device(device)
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                folder = artifact_root / f"data/processed/transformers/{architecture}/seed_{seed}/full/standard"
                meta = read_json(folder / "run_metadata.json")
                tokenizer = AutoTokenizer.from_pretrained(folder / "checkpoint", local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(folder / "checkpoint", local_files_only=True).to(target)
                dataset, _ = encoded_dataset(frames["test"], tokenizer, labels, meta["config"]["max_length"])
                loader = torch.utils.data.DataLoader(dataset, batch_size=meta["config"]["batch_size"],
                                                     shuffle=False, collate_fn=trim_padding_collate)
                scores = predict_scores(model, loader, target)
                compare("C_" + architecture, seed, scores, folder / "test_scores.npz",
                        read_json(folder / "final_protocol.json")["configurations"])
                del model, loader, dataset
                if target.type == "cuda":
                    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/reproduction")
    parser.add_argument("--report", type=Path, help="Báo cáo cuối kỳ DOCX để kiểm 9 bảng kết quả")
    parser.add_argument("--inference-c", action="store_true", help="Suy luận test thật cho 9 checkpoint C")
    parser.add_argument("--inference-baseline", action="store_true", help="Suy luận test thật cho 2 model A")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.artifacts_root is None:
        from src.notebook_support import artifacts_root
        args.artifacts_root = artifacts_root()
    root, output = args.artifacts_root.resolve(), args.output.resolve()
    # Output phải riêng: không cho ghi đè dataset, scores hay bảng báo cáo cũ.
    for reserved in (root / "data", root / "reports/project_results"):
        check(not output.is_relative_to(reserved), f"Chọn output ngoài {reserved}")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "verification.json"
    marker.unlink(missing_ok=True)  # Lượt lỗi không được để lại chứng nhận PASS của lượt trước.
    result, frames, labels = verify_scores(root)
    result["report"] = verify_word_tables(args.report, result) if args.report else None
    result["inference"] = inference_test(root, output, frames, labels, result["records"],
                                          all_c=args.inference_c, baseline=args.inference_baseline, device=args.device)
    import pandas as pd
    pd.DataFrame(result["records"]).to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["averages"]).to_csv(output / "mean_std.csv", index=False, encoding="utf-8-sig")
    result.update(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  artifacts_root=str(root), source_root=str(ROOT), verified_rows=len(result["records"]),
                  verified_averages=len(result["averages"]), tolerance=TOLERANCE,
                  training_repeated=False, threshold_search_repeated=False,
                  zero_shot_inference_repeated=False)
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"PASS: 72 dòng, 36 nhóm, {len(result['inference'])} model suy luận lại. {marker}")


if __name__ == "__main__":
    main()
