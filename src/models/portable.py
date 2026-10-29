"""A single .goemotions ZIP stores weights, tokenizer, labels and frozen thresholds.

No dataset, training loop or pickle is needed for inference.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def pack_model(run, destination):
    from src.models.transformer import load_transformer_run
    run, destination = Path(run).resolve(), Path(destination).resolve()
    metadata = load_transformer_run(run, require_full=True)
    protocol=json.loads((run/'final_protocol.json').read_text(encoding='utf-8'))
    if (protocol['run_metadata_sha256'] != digest(run/'run_metadata.json') or
            protocol['label_names'] != metadata['label_names'] or protocol['threshold_mode']!='tuned'):
        raise ValueError('Frozen thresholds do not match this checkpoint')
    files={f'checkpoint/{p.name}':p for p in sorted((run/'checkpoint').iterdir()) if p.is_file()}
    if not any(x.endswith('.safetensors') for x in files):
        raise ValueError('A safetensors checkpoint is required')
    manifest={'format_version':1,'architecture':metadata['architecture'],'seed':metadata['seed'],
              'label_names':metadata['label_names'],'thresholds':protocol['thresholds'],
              'max_length':metadata['config']['max_length'],
              'checkpoint_revision':metadata['model_revision'],
              'source_run_metadata_sha256':digest(run/'run_metadata.json'),
              'threshold_source':'frozen original validation protocol',
              'files_sha256':{name:digest(path) for name,path in files.items()}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    temporary=destination.with_suffix(destination.suffix+'.part')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in files.items():z.write(path,name)
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    temporary.replace(destination)
    return manifest


def unpack_model(archive, cache=None):
    archive=Path(archive).resolve()
    key=digest(archive)
    root=Path(cache or Path(tempfile.gettempdir())/'goemotions-models')/key
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
        m=json.loads(z.read('manifest.json'))
        if m.get('format_version')!=1:raise ValueError('Unsupported model bundle version')
        if set(names)!=set(m['files_sha256'])|{'manifest.json'}:raise ValueError('Unexpected model files')
        labels,thresholds=m['label_names'],m['thresholds']
        if (len(labels)!=28 or len(set(labels))!=28 or len(thresholds)!=28 or
                any(not isinstance(t,(int,float)) or not 0<=t<=1 for t in thresholds)):
            raise ValueError('Invalid label mapping or thresholds')
        for name,expected in m['files_sha256'].items():
            target=(root/name).resolve()
            if not target.is_relative_to(root.resolve()) or not name.startswith('checkpoint/'):
                raise ValueError('Invalid model member path')
            if target.is_file() and digest(target)==expected:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as source,target.with_suffix(target.suffix+'.part').open('wb') as out:
                while chunk:=source.read(2**20):out.write(chunk)
            part=target.with_suffix(target.suffix+'.part')
            if digest(part)!=expected:
                part.unlink()
                raise ValueError('Model bundle checksum mismatch')
            part.replace(target)
    return root,m


class PortablePredictor:
    def __init__(self,archive,device='cpu',cache=None):
        os.environ.setdefault('HF_HUB_OFFLINE','1')
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        root,self.manifest=unpack_model(archive,cache)
        self.torch=torch
        torch.set_num_threads(4)
        self.device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else device if device!='auto' else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(root/'checkpoint',local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(root/'checkpoint',local_files_only=True,
            trust_remote_code=False,use_safetensors=True).to(self.device).eval()
        if (self.model.config.problem_type!='multi_label_classification' or
                [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]!=self.manifest['label_names']):
            raise ValueError('Classifier mapping differs from bundle')

    def predict(self,text,mode='tuned'):
        if not isinstance(text,str) or not text.strip() or len(text)>20000:
            raise ValueError('Nhập một bình luận tiếng Anh từ 1 đến 20.000 ký tự.')
        if mode not in ('fixed','tuned'):raise ValueError('Unknown threshold mode')
        text=text.strip()
        raw=self.tokenizer(text,add_special_tokens=True,truncation=False,verbose=False)
        batch=self.tokenizer(text,max_length=self.manifest['max_length'],truncation=True,return_tensors='pt')
        with self.torch.inference_mode():
            logits=self.model(**{k:v.to(self.device) for k,v in batch.items()}).logits
            scores=self.torch.sigmoid(logits.float())[0].cpu().tolist()
        limits=self.manifest['thresholds'] if mode=='tuned' else [0.5]*28
        rows=[{'label':name,'score':score,'threshold':limit,'selected':score>=limit}
              for name,score,limit in zip(self.manifest['label_names'],scores,limits)]
        rows.sort(key=lambda x:-x['score'])
        return {'text':text,'architecture':self.manifest['architecture'],'seed':self.manifest['seed'],
                'threshold_mode':mode,'labels':[r['label'] for r in rows if r['selected']],
                'rows':rows,'tokens':self.tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist()),
                'original_token_count':len(raw['input_ids']),
                'truncated':len(raw['input_ids'])>self.manifest['max_length']}
