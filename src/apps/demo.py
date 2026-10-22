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
    parser.add_argument("--artifacts-root", type=Path, default=ROOT,
                        help="Thư mục chứa data/ của lượt chạy đã báo cáo")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--thresholds", type=Path, help="Ngưỡng validation của đúng run C được chọn")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    args.artifacts_root = args.artifacts_root.resolve()
    if args.selection is None:
        args.selection = args.artifacts_root / "data/processed/transformers/selected_model.json"
    import gradio as gr
    from src.apps.model_selector import ModelSelector
    from src.apps.demo_details import MODES

    selector = ModelSelector(args.artifacts_root, args.selection, args.device, args.thresholds)

    def predict(text, model_key, threshold_mode):
        try:
            return selector.predict(text, model_key, threshold_mode)
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(f"Không thể dự đoán bằng mô hình đã chọn: {error}") from error

    modes = list(MODES)
    initial_mode = "fixed"
    if args.thresholds:
        modes.append(("Ngưỡng từ tệp được cung cấp", "custom"))
        initial_mode = "custom"

    def update_details(key, mode):
        try:
            info, params, metrics = selector.presentation(key, mode)
            return info, params, metrics, "", []
        except (ValueError, OSError, KeyError) as error:
            raise gr.Error(f"Chưa đọc được thông tin mô hình: {error}") from error

    initial_info, initial_params, initial_metrics = selector.presentation(selector.default, initial_mode)
    with gr.Blocks(title="GoEmotions — nhận diện cảm xúc", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Nhận diện cảm xúc trong bình luận\n"
                    "Nhập văn bản **tiếng Anh**. Một câu có thể có nhiều cảm xúc trong 28 nhãn GoEmotions.")
        with gr.Row():
            model_choice = gr.Dropdown(choices=selector.choices, value=selector.default,
                                       label="Chọn mô hình", interactive=True, scale=3)
            threshold_choice = gr.Dropdown(choices=modes, value=initial_mode,
                                           label="Luật chọn nhãn", interactive=True, scale=2)
        model_info = gr.Markdown(initial_info)
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Bình luận tiếng Anh", lines=4,
                                  placeholder="Thank you so much! I am really happy with your help.")
                button = gr.Button("Nhận diện cảm xúc", variant="primary")
                gr.Examples(examples=[["Thank you so much! I am really happy with your help."],
                                      ["I am disappointed and worried about what will happen next."],
                                      ["The meeting starts at nine tomorrow."]], inputs=text, label="Ví dụ để thử")
            with gr.Column(scale=2):
                gr.Markdown("### Kết quả test đã đánh giá\n"
                            "Cột checkpoint thuộc lượt đang chọn; cột cuối là mean ± std của ba seed C. "
                            "A/B chỉ có một lượt. Các chỉ số tính trên 5.427 câu test, không tính từ câu vừa nhập.")
                metrics_table = gr.Dataframe(value=initial_metrics,
                    headers=["Chỉ số", "Lượt chọn", "Báo cáo nhóm"],
                    datatype=["str", "str", "str"], interactive=False, wrap=True,
                    column_widths=["30%", "25%", "45%"])
        with gr.Accordion("Thông số đã chạy của mô hình đang chọn", open=False):
            parameter_table = gr.Dataframe(value=initial_params, headers=["Thông số", "Giá trị đã chạy"],
                                           datatype=["str", "str"], interactive=False, wrap=True, max_height=650)
        gr.Markdown("Điểm dự đoán chưa được hiệu chuẩn thành xác suất đáng tin cậy. "
                    "Ngưỡng chung và ngưỡng riêng được lấy từ validation, giữ nguyên trọng số đã học.")
        result = gr.Markdown()
        table = gr.Dataframe(headers=["Cảm xúc", "Điểm", "Ngưỡng", "Được chọn"],
                             datatype=["str", "number", "number", "str"], interactive=False,
                             label="Điểm của 28 nhãn")
        detail_outputs = [model_info, parameter_table, metrics_table, result, table]
        model_choice.change(update_details, inputs=[model_choice, threshold_choice],
                            outputs=detail_outputs, concurrency_limit=1, concurrency_id="details",
                            trigger_mode="always_last", api_name="model_details")
        threshold_choice.change(update_details, inputs=[model_choice, threshold_choice],
                                outputs=detail_outputs, concurrency_limit=1, concurrency_id="details",
                                trigger_mode="always_last", api_name=False)
        button.click(predict, inputs=[text, model_choice, threshold_choice], outputs=[result, table],
                     concurrency_limit=1, api_name="predict")
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
