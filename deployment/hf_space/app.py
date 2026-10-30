"""Minimal CPU Space for the frozen BERT/123 checkpoint; no training."""
from pathlib import Path
from functools import lru_cache
import os,json,hashlib,urllib.request,zipfile,threading
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer,AutoModelForSequenceClassification

HERE=Path(__file__).resolve().parent
META=json.loads((HERE/'model_manifest.json').read_text(encoding='utf-8'))
LOCK=threading.Lock()
EXAMPLES=["You stayed late to help me. I really appreciate it!","I was worried, but your help made me feel so relieved.","I am not angry with you, just disappointed in the result.","Wonderful, another three hours stuck in traffic.","The package will arrive on Tuesday afternoon."]
def digest(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def valid(folder):
 return all((folder/name).is_file() and digest(folder/name)==sha for name,sha in META['checkpoint_sha256'].items())
@lru_cache(maxsize=1)
def load():
 folder=Path(os.environ.get('GOEMOTIONS_CHECKPOINT',HERE/'.cache/checkpoint'))
 if not valid(folder):
  if os.environ.get('GOEMOTIONS_CHECKPOINT'):raise ValueError('Checkpoint local không khớp manifest')
  folder.mkdir(parents=True,exist_ok=True);archive=folder.parent/'bert_123.zip'
  if not archive.is_file() or digest(archive)!=META['archive_sha256']:
   part=archive.with_suffix('.part')
   req=urllib.request.Request(META['download_url'],headers={'User-Agent':'GoEmotions-Gradio-Space'})
   with urllib.request.urlopen(req,timeout=120) as response,part.open('wb') as out:
    while chunk:=response.read(2**20):out.write(chunk)
   if digest(part)!=META['archive_sha256']:raise ValueError('ZIP checkpoint sai SHA-256')
   part.replace(archive)
  with zipfile.ZipFile(archive) as z:
   for name in META['checkpoint_sha256']:
    if Path(name).name!=name:raise ValueError('Tên file manifest không hợp lệ')
    (folder/name).write_bytes(z.read(META['archive_prefix']+name))
  if not valid(folder):raise ValueError('Checkpoint giải nén khác manifest')
 torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
 model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True,trust_remote_code=False).float().eval()
 if model.config.num_labels!=28 or [model.config.id2label[i] for i in range(28)]!=META['label_names']:raise ValueError('Mapping checkpoint không đúng')
 return tok,model
def predict(text,mode='tuned'):
 text=(text or '').strip()
 if not text:return 'Vui lòng nhập một bình luận tiếng Anh.',[]
 if len(text)>20000:return 'Văn bản quá dài; dùng dưới 20.000 ký tự.',[]
 if mode not in ('fixed','tuned'):raise ValueError('Chế độ ngưỡng không hợp lệ')
 with LOCK:
  tok,model=load();raw=tok(text,truncation=False,verbose=False)['input_ids']
  encoded=tok(text,truncation=True,max_length=128,return_tensors='pt')
  with torch.inference_mode():scores=torch.sigmoid(model(**encoded).logits)[0].numpy()
 limits=np.asarray(META['thresholds'] if mode=='tuned' else [.5]*28)
 labels=META['label_names'];chosen=[l for l,s,t in zip(labels,scores,limits) if s>=t]
 message='**Cảm xúc dự đoán:** '+', '.join(chosen) if chosen else '**Chưa có nhãn nào đạt ngưỡng.** Không tự gán neutral.'
 if len(raw)>128:message+='\n\nVăn bản được cắt còn 128 token.'
 rows=[[labels[i],round(float(scores[i]),4),round(float(limits[i]),4),'Có' if scores[i]>=limits[i] else ''] for i in np.argsort(-scores)]
 return message,rows
def build():
 with gr.Blocks(title='GoEmotions BERT') as demo:
  gr.Markdown('# Nhận diện cảm xúc GoEmotions\nBERT seed 123. Nhập bình luận **tiếng Anh**; một câu có thể mang nhiều nhãn. Điểm chưa được hiệu chuẩn thành xác suất đáng tin cậy.')
  mode=gr.Radio([('Ngưỡng riêng (mặc định)','tuned'),('Ngưỡng 0,5','fixed')],value='tuned',label='Chế độ ngưỡng')
  entry=gr.Textbox(label='Bình luận',lines=3);button=gr.Button('Nhận diện cảm xúc',variant='primary');result=gr.Markdown()
  grid=gr.Dataframe(headers=['Cảm xúc','Điểm','Ngưỡng áp dụng','Đạt ngưỡng'],datatype=['str','number','number','str'],interactive=False)
  button.click(predict,[entry,mode],[result,grid],api_name='predict',concurrency_limit=1)
  mode.change(lambda:('',[]),outputs=[result,grid],api_name=False)
  gr.Examples([[x] for x in EXAMPLES],inputs=entry)
 return demo
if __name__=='__main__':build().launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),share=False)
