"""Gradio demo using validation-selected checkpoints and their frozen thresholds.

Run: python app_best_epochs.py --device auto
Verify the three current C checkpoints: python app_best_epochs.py --verify
No model training or threshold fitting happens when this application runs.
"""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
from src.apps.demo import TransformerDemo, DEMO_SENTENCES
from src.apps.model_selector import ModelSelector, ZeroShotDemo, format_scores
from src.apps import model_selector as legacy_demo
from src.models.transformer import load_transformer_run, configure_console
from src.evaluation.protocols import validate_protocol
from src.datasets.goemotions import sha256

OLD=REPO/"artifacts/final_submission"
MAIN=ROOT/"main_results"
legacy_demo.ROOT=REPO


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


class AmpDemo(TransformerDemo):
    def predict(self,text,threshold=0.5):
        text=(text or "").strip()
        if not text:return "Vui lòng nhập một câu tiếng Anh.",[]
        raw=self.tokenizer(text,truncation=False,verbose=False)["input_ids"]
        encoded=self.tokenizer(text,truncation=True,max_length=self.max_length,return_tensors="pt")
        inputs={key:value.to(self.device) for key,value in encoded.items()}
        context=self.torch.autocast("cuda",dtype=self.torch.float16) if self.device.type=="cuda" else nullcontext()
        with self.torch.inference_mode(),context:
            logits=self.model(**inputs).logits
        points=self.torch.sigmoid(logits.float())[0].cpu().numpy()
        message,rows=format_scores(self.labels,points,threshold)
        if len(raw)>self.max_length:message+=f"\n\nVăn bản đã được cắt còn {self.max_length} token."
        return message,rows


class BestEpochSelector(ModelSelector):
    def __init__(self,device="auto"):
        self.lock=read(MAIN/"LOCK_BEFORE_TEST.json")
        if self.lock["selection_sha256"]!=sha256(MAIN/"selection.json"):
            raise ValueError("Lựa chọn mô hình khác biên bản đã khóa")
        super().__init__(OLD,MAIN/"selection.json",device=device)
        self.current_selection=read(MAIN/"selection.json")

    def _load(self,key):
        if key in ("bert","roberta","distilbert"):
            seed=self.current_selection["representatives"][key]
            row=next(r for r in self.lock["runs"] if r["architecture"]==key and r["seed"]==seed)
            protocol_path=MAIN/Path(row["protocol"].replace("\\","/"))
            if sha256(protocol_path)!=row["protocol_sha256"]:raise ValueError("Ngưỡng đã thay đổi sau khi khóa")
            run=Path(row["run_dir"])
            if not run.exists():run=ROOT/"checkpoints"/key/f"seed_{seed}"
            metadata=load_transformer_run(run,require_full=True)
            protocol=read(protocol_path)
            validate_protocol(run,protocol,metadata["label_names"])
            predictor=AmpDemo.from_checked_checkpoint(run/"checkpoint",metadata,self.device)
            predictor.frozen_protocol=protocol
            return predictor
        if key=="zero_shot":return ZeroShotDemo(OLD,"cpu")
        return super()._load(key)

    def _limits(self,key,mode):
        if key not in ("bert","roberta","distilbert"):return super()._limits(key,mode)
        if mode not in ("fixed","tuned"):raise ValueError("Chế độ ngưỡng không hợp lệ")
        options=[r for r in self._predictor.frozen_protocol["configurations"] if r["threshold_mode"]==mode]
        if len(options)!=1:raise ValueError("Thiếu ngưỡng đã khóa của đúng checkpoint")
        return options[0]["thresholds"],""


def verify():
    selector=BestEpochSelector("cuda")
    if selector.predict("","bert")[1] or selector.predict("x"*20001,"bert")[1]:
        raise ValueError("Invalid input was not rejected")
    expected=read(MAIN/"report_data/demo_predictions.json")
    checked=[]
    for key in ("bert","roberta","distilbert"):
        seed=selector.current_selection["representatives"][key]
        rows=[r for r in expected if r["architecture"]==key and r["seed"]==seed]
        for record in rows:
            message,points=selector.predict(record["text"],key,record["mode"])
            actual={r[0]:r[1] for r in points}
            chosen={r[0] for r in points if r[3]=="Có"}
            if chosen!=set(record["labels"]):raise ValueError("Demo decisions differ from measured example")
            for name,value in record["scores"].items():
                if abs(actual[name]-round(value,4))>0.00011:raise ValueError("Demo points differ from saved example")
            checked.append({"architecture":key,"seed":seed,"mode":record["mode"],"text":record["text"]})
    receipt={"status":"passed","input_validation_passed":True,"checked_examples":checked,
             "selection_sha256":sha256(MAIN/"selection.json"),"training":False,"threshold_fitting":False}
    (MAIN/"demo_verification.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Verified",len(checked),"current C/demo combinations; no training.")


def main():
    configure_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--port",type=int,default=7861)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.verify:return verify()
    import gradio as gr
    selector=BestEpochSelector(args.device)
    def predict(text,key,mode):
        try:return selector.predict(text,key,mode)
        except (ValueError,OSError,RuntimeError) as error:raise gr.Error(str(error)) from error
    with gr.Blocks(title="GoEmotions — nhận diện cảm xúc") as demo:
        gr.Markdown("# Nhận diện cảm xúc trong bình luận\nNhập văn bản **tiếng Anh**. Một câu có thể có nhiều nhãn cảm xúc.")
        choice=gr.Dropdown(choices=selector.choices,value=selector.default,label="Chọn mô hình")
        mode=gr.Radio(choices=[("Ngưỡng riêng","tuned"),("Ngưỡng 0,5","fixed")],value="tuned",label="Chế độ ngưỡng")
        text=gr.Textbox(label="Bình luận",lines=4)
        button=gr.Button("Nhận diện cảm xúc",variant="primary")
        result=gr.Markdown()
        table=gr.Dataframe(headers=["Cảm xúc","Điểm","Ngưỡng áp dụng","Đạt ngưỡng"],datatype=["str","number","number","str"],interactive=False)
        button.click(predict,inputs=[text,choice,mode],outputs=[result,table],concurrency_limit=1,api_name="predict")
        gr.Examples(examples=[[t] for t in DEMO_SENTENCES],inputs=text)
    demo.launch(server_name="127.0.0.1",server_port=args.port,share=False)


if __name__=="__main__":main()
