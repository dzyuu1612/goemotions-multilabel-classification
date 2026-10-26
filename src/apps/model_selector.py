"""Các bộ dự đoán A/B/C cho giao diện; chỉ đọc artifacts đã hoàn thành."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from threading import Lock

import numpy as np

from src.datasets.goemotions import REVISION
from src.models.baseline import load_run_metadata
from src.models.transformer import load_transformer_run, resolve_device
from src.models.zero_shot import CHECKPOINT, HYPOTHESIS_TEMPLATE, map_pipeline_scores
from src.paths import ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_scores(labels, scores, threshold=0.5):
    scores = np.asarray(scores, dtype=float)
    limits = np.broadcast_to(np.asarray(threshold), (len(labels),))
    if (scores.shape != (28,) or len(labels) != 28 or len(set(labels)) != 28
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Điểm dự đoán phải đủ 28 nhãn và nằm trong [0, 1]")
    chosen = [label for label, score, limit in zip(labels, scores, limits) if score >= limit]
    message = ("**Cảm xúc dự đoán:** " + ", ".join(chosen)) if chosen else (
        "**Chưa có nhãn nào đạt ngưỡng.** Không tự suy ra neutral trong trường hợp này.")
    rows = [[labels[i], round(float(scores[i]), 4), round(float(limits[i]), 4),
             "Có" if scores[i] >= limits[i] else ""] for i in np.argsort(-scores)]
    return message, rows


class BaselineDemo:
    def __init__(self, root, variant):
        import joblib

        self.labels = read_json(root / "data/labels.json")
        base = root / "data/processed/baseline"
        folder = base / ("full" if variant == "standard" else "balanced/full")
        load_run_metadata(folder, variant, self.labels)
        self.model = joblib.load(folder / "model.joblib")

    def predict(self, text):
        return format_scores(self.labels, self.model.predict_proba([text])[0])


class ZeroShotDemo:
    def __init__(self, root, device):
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from transformers import pipeline

        metadata = read_json(root / "data/processed/zero_shot/full/run_metadata.json")
        self.labels = read_json(root / "data/labels.json")
        if (metadata.get("status") != "complete" or metadata.get("smoke") is not False
                or metadata.get("checkpoint") != CHECKPOINT
                or metadata.get("data_revision") != REVISION
                or metadata.get("label_names") != self.labels
                or metadata.get("hypothesis_template") != HYPOTHESIS_TEMPLATE
                or metadata.get("multi_label") is not True):
            raise ValueError("Metadata BART-MNLI không khớp lượt zero-shot đã báo cáo")
        self.template = metadata["hypothesis_template"]
        resolved = resolve_device(device)
        self.max_length = 1024
        download_args = dict(
            repo_id=CHECKPOINT, revision=metadata["model_revision"],
            cache_dir=str(ROOT / "data/cache/huggingface/hub"),
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                            "tokenizer_config.json", "vocab.json", "merges.txt",
                            "special_tokens_map.json"],
        )
        try:
            folder = snapshot_download(**download_args, local_files_only=True)
            if not all((Path(folder) / name).is_file() for name in
                       ("config.json", "model.safetensors", "vocab.json", "merges.txt")):
                raise FileNotFoundError("Cache BART-MNLI chưa đủ trọng số/tokenizer")
        except (LocalEntryNotFoundError, FileNotFoundError):
            folder = snapshot_download(**download_args)
        self.classifier = pipeline(
            "zero-shot-classification", model=folder, tokenizer=folder,
            device=0 if resolved.type == "cuda" else -1, framework="pt",
            trust_remote_code=False,
            dtype=torch.float16 if resolved.type == "cuda" and metadata.get("dtype") == "float16"
            else torch.float32,
        )
        self.classifier.model.eval()
        if self.classifier.entailment_id < 0:
            raise ValueError("Checkpoint B không có mapping entailment")

    def predict(self, text):
        result = self.classifier(text, candidate_labels=self.labels, multi_label=True,
                                 hypothesis_template=self.template, batch_size=8)
        scores = map_pipeline_scores(result, self.labels)[0]
        message, rows = format_scores(self.labels, scores)
        # Pipeline NLI cắt phần premise khi cặp premise/hypothesis quá dài.
        tokenizer = self.classifier.tokenizer
        lengths = [len(tokenizer(text, self.template.format(label), truncation=False,
                                 verbose=False)["input_ids"]) for label in self.labels]
        if max(lengths) > self.max_length:
            message += "\n\nVăn bản đã được cắt để vừa giới hạn 1.024 token của cặp NLI."
        return message, rows


class ModelSelector:
    """Nạp khi cần và chỉ giữ một model để tránh chiếm RAM khi đổi lựa chọn."""
    choices = [
        ("C1 · BERT", "bert"),
        ("C2 · RoBERTa", "roberta"),
        ("C3 · DistilBERT", "distilbert"),
        ("A · TF-IDF + Logistic Regression — thường", "baseline_standard"),
        ("A · TF-IDF + Logistic Regression — cân bằng lớp", "baseline_balanced"),
        ("B · Zero-shot BART-MNLI", "zero_shot"),
    ]

    def __init__(self, root, selection, device="auto", thresholds=None, c3_source="benchmark"):
        self.root, self.selection = Path(root).resolve(), Path(selection)
        self.device, self.thresholds = device, thresholds
        self.default = read_json(self.selection)["architecture"]
        if self.default not in ("bert", "roberta", "distilbert"):
            raise ValueError("Mô hình C mặc định không thuộc ba kiến trúc")
        if c3_source not in ("benchmark", "report-2-1"):
            raise ValueError("Nguồn C3 phải là benchmark hoặc report-2-1")
        self.c3_source = c3_source
        self.choices = [(name + (" — mô hình mặc định" if key == self.default else ""), key)
                        for name, key in type(self).choices]
        self._key, self._predictor = None, None
        self._lock = Lock()

    def describe(self, key):
        names = dict((key, name) for name, key in self.choices)
        if key not in names:
            raise ValueError("Lựa chọn mô hình không hợp lệ")
        details = {
            "bert": "Dùng seed có Macro-F1 validation cao nhất trong ba seed BERT.",
            "roberta": "Dùng seed có Macro-F1 validation cao nhất trong ba seed RoBERTa.",
            "distilbert": ("Dùng seed đại diện của bộ DistilBERT trong báo cáo 2.1."
                           if self.c3_source == "report-2-1" else
                           "Dùng seed có Macro-F1 validation cao nhất trong ba seed DistilBERT của lượt chạy chung."),
            "baseline_standard": "TF-IDF và 28 bộ phân loại Logistic Regression, không cân bằng lớp.",
            "baseline_balanced": "TF-IDF và Logistic Regression có trọng số cân bằng lớp.",
            "zero_shot": "BART-MNLI suy luận trực tiếp, không fine-tune trên GoEmotions; có thể chậm hơn.",
        }
        threshold_note = ("Dùng ngưỡng riêng đã cung cấp cho checkpoint mặc định."
                          if key == self.default and self.thresholds else "Ngưỡng dự đoán: 0,5.")
        return f"**{names[key]}**. {details[key]} {threshold_note}"

    def _load(self, key):
        from src.apps.demo import TransformerDemo

        if key == self.default:
            return TransformerDemo(self.selection, self.device, self.thresholds, self.root)
        if key in ("bert", "roberta") or (key == "distilbert" and self.c3_source == "benchmark"):
            base = self.root / "data/processed/transformers" / key
            candidates = []
            for folder in sorted(base.glob("seed_*/full/standard")):
                metrics = read_json(folder / "validation_metrics.json")
                seed = int(folder.parents[1].name.removeprefix("seed_"))
                candidates.append((metrics["macro_f1"], -seed, folder))
            if not candidates:
                raise FileNotFoundError(f"Chưa có checkpoint full cho {key}")
            folder = max(candidates)[2]
            metadata = load_transformer_run(folder, require_full=True)
            if metadata["architecture"] != key:
                raise ValueError("Kiến trúc checkpoint khác lựa chọn")
            return TransformerDemo.from_checked_checkpoint(folder / "checkpoint", metadata, self.device)
        if key == "distilbert":
            from src.current_report import checked_c3, current_c3_root

            folder = current_c3_root()
            selected = read_json(folder / "selection.json")
            if selected.get("selection_split") != "validation" or selected.get("test_used") is not False:
                raise ValueError("C3 phải được chọn bằng validation")
            seed = selected["representative_seed"]
            metadata, _ = checked_c3(folder, seed)
            return TransformerDemo.from_checked_checkpoint(
                folder / f"seed_{seed}" / "best", metadata, self.device)
        if key in ("baseline_standard", "baseline_balanced"):
            return BaselineDemo(self.root, key.removeprefix("baseline_"))
        if key == "zero_shot":
            return ZeroShotDemo(self.root, self.device)
        raise ValueError("Lựa chọn mô hình không hợp lệ")

    def predict(self, text, key):
        self.describe(key)
        text = (text or "").strip()
        if not text:
            return "Vui lòng nhập một câu hoặc bình luận tiếng Anh.", []
        if len(text) > 20000:
            return "Văn bản quá dài. Vui lòng dùng một bình luận dưới 20.000 ký tự.", []
        with self._lock:
            if self._key != key:
                self._predictor, self._key = None, None
                gc.collect()
                import torch

                torch.set_num_threads(4)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._predictor = self._load(key)
                self._key = key
            message, rows = self._predictor.predict(text)
            metadata = getattr(self._predictor, "metadata", {})
            seed = metadata.get("seed", metadata.get("config", {}).get("seed"))
            seed_note = f" · seed {seed}" if seed is not None else ""
            return f"**Mô hình:** {dict((k, n) for n, k in self.choices)[key]}{seed_note}\n\n{message}", rows
