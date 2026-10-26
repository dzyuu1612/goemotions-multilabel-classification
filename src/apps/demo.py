"""Demo D mặc định dùng C đã chọn; có thể thử A, B và ba kiến trúc C.

Chạy: python app.py --device auto
Không có checkpoint đầy đủ thì báo cách tạo; không trả kết quả giả.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from src.models.transformer import load_demo_selection, resolve_device, configure_console

from src.paths import ROOT


class TransformerDemo:
    def __init__(self, selection, device="auto", thresholds=None, artifact_root=ROOT):
        folder, self.metadata, self.thresholds = load_demo_selection(
            artifact_root, selection, thresholds_path=thresholds
        )
        self._load_checkpoint(folder / "checkpoint", device)

    def _load_checkpoint(self, checkpoint, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        ).to(self.device).eval()
        self.labels = self.metadata["label_names"]
        self.max_length = self.metadata["config"]["max_length"]
        if (self.model.config.num_labels != 28 or
                [self.model.config.id2label[i] for i in range(28)] != self.labels or
                self.model.config.problem_type != "multi_label_classification"):
            raise ValueError("Checkpoint demo không khớp head/mapping đa nhãn")
        self.torch = torch

    @classmethod
    def from_checked_checkpoint(cls, checkpoint, metadata, device):
        """Caller kiểm biên nhận của run trước khi gọi hàm nạp dùng chung."""
        predictor = cls.__new__(cls)
        predictor.metadata, predictor.thresholds = metadata, 0.5
        predictor._load_checkpoint(checkpoint, device)
        return predictor

    def predict(self, text):
        """Cùng tokenizer/mapping/sigmoid/ngưỡng như lúc đánh giá checkpoint."""
        text = (text or "").strip()
        if not text:
            return "Vui lòng nhập một câu hoặc bình luận tiếng Anh.", []
        if len(text) > 20000:
            return "Văn bản quá dài. Vui lòng dùng một bình luận dưới 20.000 ký tự.", []
        # Chỉ đếm để báo cắt token; chuỗi dài này không được đưa vào encoder.
        raw_tokens = self.tokenizer(text, truncation=False, add_special_tokens=True,
                                    verbose=False)["input_ids"]
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            context = self.torch.autocast("cuda", dtype=self.torch.float16) if self.device.type == "cuda" else nullcontext()
            with context:
                logits = self.model(**inputs).logits
            scores = self.torch.sigmoid(logits.float())[0].cpu().numpy()
        if scores.shape != (28,) or not np.isfinite(scores).all():
            raise ValueError("Mô hình trả scores sai")
        limits = np.broadcast_to(np.asarray(self.thresholds), (28,))
        selected = [name for name, score, limit in zip(self.labels, scores, limits) if score >= limit]
        message = ("**Cảm xúc dự đoán:** " + ", ".join(selected)) if selected else (
            "**Chưa có nhãn nào đạt ngưỡng.** Không tự suy ra neutral trong trường hợp này."
        )
        if len(raw_tokens) > self.max_length:
            message += f"\n\nVăn bản đã được cắt còn {self.max_length} token để dự đoán."
        rows = [[self.labels[i], round(float(scores[i]), 4), round(float(limits[i]), 4),
                 "Có" if scores[i] >= limits[i] else ""] for i in np.argsort(-scores)]
        return message, rows


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path,
                        help="Thư mục chứa data/ của lượt chạy đã báo cáo")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--thresholds", type=Path, help="Ngưỡng validation của đúng run C được chọn")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--c3-source", choices=("benchmark", "report-2-1"), default="benchmark",
                        help="benchmark: cùng bộ 9 run của pipeline; report-2-1: bundle C3 riêng của báo cáo cũ")
    args = parser.parse_args()
    if args.artifacts_root is None:
        from src.notebook_support import artifacts_root
        args.artifacts_root = artifacts_root()
    args.artifacts_root = args.artifacts_root.resolve()
    if args.selection is None:
        args.selection = args.artifacts_root / "data/processed/transformers/selected_model.json"
    import gradio as gr
    from src.apps.model_selector import ModelSelector

    selector = ModelSelector(args.artifacts_root, args.selection, args.device, args.thresholds, args.c3_source)

    def predict(text, model_key):
        try:
            return selector.predict(text, model_key)
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(f"Không thể dự đoán bằng mô hình đã chọn: {error}") from error

    with gr.Blocks(title="GoEmotions — nhận diện cảm xúc") as demo:
        gr.Markdown("# Nhận diện cảm xúc trong bình luận\n"
                    "Nhập văn bản **tiếng Anh**. Một câu có thể có nhiều cảm xúc trong 28 nhãn GoEmotions.")
        model_choice = gr.Dropdown(choices=selector.choices, value=selector.default,
                                   label="Chọn mô hình", interactive=True)
        model_info = gr.Markdown(selector.describe(selector.default))
        gr.Markdown("Điểm là đầu ra của từng mô hình, chưa được hiệu chuẩn thành xác suất "
                    "đáng tin cậy. Lần chọn mô hình đầu tiên có thể cần thời gian nạp.")
        text = gr.Textbox(label="Bình luận tiếng Anh", lines=4,
                          placeholder="Thank you so much! I am really happy with your help.")
        button = gr.Button("Nhận diện cảm xúc", variant="primary")
        result = gr.Markdown()
        table = gr.Dataframe(headers=["Cảm xúc", "Điểm", "Ngưỡng", "Được chọn"],
                             datatype=["str", "number", "number", "str"], interactive=False,
                             label="Điểm của 28 nhãn")
        model_choice.change(lambda key: (selector.describe(key), "", []),
                            inputs=model_choice, outputs=[model_info, result, table],
                            queue=False, api_name=False)
        button.click(predict, inputs=[text, model_choice], outputs=[result, table],
                     concurrency_limit=1, api_name="predict")
        gr.Examples(examples=[["Thank you so much! I am really happy with your help."],
                              ["I am disappointed and worried about what will happen next."],
                              ["The meeting starts at nine tomorrow."]], inputs=text)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
