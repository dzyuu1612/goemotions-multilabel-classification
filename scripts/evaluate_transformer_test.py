"""Suy luận test bằng checkpoint C và ba ngưỡng đã khóa từ validation."""
import argparse
from pathlib import Path
import numpy as np

from src.baseline import load_aligned_scores
from src.data import load_goemotions, multi_hot, sha256
from src.experiment import read_json, save_json, validate_protocol
from src.metrics import evaluate_multilabel

ROOT = Path(__file__).resolve().parents[1]


def evaluate_run(folder, protocol_path, *, device="auto", batch_size=16):
    folder, protocol_path = Path(folder), Path(protocol_path)
    labels = read_json(ROOT / "data/labels.json")
    protocol = read_json(protocol_path)
    metadata = validate_protocol(folder, protocol, labels)
    fingerprint = sha256(protocol_path)
    result_path = folder / "test_results.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved["protocol_sha256"] != fingerprint or saved["scores_sha256"] != sha256(folder / "test_scores.npz"):
            raise ValueError("Kết quả test bị đổi hoặc thuộc protocol khác")
        return saved
    scores_path = folder / "test_scores.npz"
    marker = folder / "test_protocol.sha256"
    if scores_path.exists() and not marker.exists():
        raise ValueError("Scores test có sẵn nhưng không có dấu vết protocol; không tái sử dụng")
    if marker.exists() and marker.read_text().strip() != fingerprint:
        raise ValueError("Đã bắt đầu test với protocol khác")
    marker.write_text(fingerprint, encoding="ascii")
    # Nhãn test chỉ được mở sau khi kiểm toàn bộ checkpoint và protocol.
    frames, names, _ = load_goemotions(ROOT, write_metadata=False, splits=("test",))
    if names != labels:
        raise ValueError("Mapping test sai")
    frame = frames["test"]
    if scores_path.exists():
        scores = load_aligned_scores(scores_path, frame["id"], labels)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from src.neural import resolve_device
        from scripts.train_transformer import encoded_dataset, predict_scores, trim_padding_collate
        torch.set_num_threads(4)
        target = resolve_device(device)
        checkpoint = folder / "checkpoint"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(target)
        dataset, _ = encoded_dataset(frame, tokenizer, labels, metadata["config"]["max_length"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=trim_padding_collate)
        scores = predict_scores(model, loader, target)
        np.savez_compressed(scores_path, ids=frame["id"].to_numpy(dtype=str),
                            scores=scores, label_names=np.asarray(labels, dtype=str))
    truth = multi_hot(frame["labels"].tolist(), len(labels))
    rows = [{"threshold_mode": cfg["threshold_mode"],
             "metrics": evaluate_multilabel(truth, scores, labels, cfg["thresholds"])}
            for cfg in protocol["configurations"]]
    saved = {"split": "test", "architecture": metadata["architecture"],
             "seed": metadata["seed"], "method": "C", "protocol_sha256": fingerprint,
             "scores_sha256": sha256(scores_path), "results": rows}
    save_json(result_path, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    protocol = args.protocol or args.run_dir / "final_protocol.json"
    result = evaluate_run(args.run_dir, protocol, device=args.device, batch_size=args.batch_size)
    for row in result["results"]:
        print(row["threshold_mode"], row["metrics"]["macro_f1"], row["metrics"]["micro_f1"])


if __name__ == "__main__":
    main()
