from pathlib import Path
import os,json
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import joblib
from transformers import AutoTokenizer
from src.datasets.goemotions import load_goemotions,multi_hot,sha256
from scripts.verification.audit_data_boundaries import audit

import argparse
from src.paths import ROOT
from src.notebook_support import artifacts_root
p=argparse.ArgumentParser(description='Audit every split with the actual frozen tokenizers; no classifier training.')
p.add_argument('--artifacts-root',type=Path);p.add_argument('--output',type=Path,default=ROOT/'results/final_submission')
args=p.parse_args();W=args.output;W.mkdir(parents=True,exist_ok=True);A=args.artifacts_root or artifacts_root()
write=lambda p,v:p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
base=audit(A);write(W/'data_boundary_audit.json',base)
frames,names,_=load_goemotions(A,write_metadata=False)
model=joblib.load(A/'data/processed/baseline/full/model.joblib');vec=model.named_steps['tfidf']
records=[]
for split,f in frames.items():
 x=vec.transform(f.text.tolist());y=multi_hot(f.labels.tolist(),28)
 records.append(dict(model='TF-IDF',split=split,n=len(f),feature_shape=list(x.shape),label_shape=list(y.shape),
   empty_feature_rows=int((x.getnnz(axis=1)==0).sum()),positive_labels=int(y.sum()),
   multilabel_rows=int((y.sum(axis=1)>1).sum()),neutral_with_other=int(((y[:,27]==1)&(y.sum(axis=1)>1)).sum())))
 print('TF-IDF',split,records[-1],flush=True)
folders={'bert':A/'data/processed/transformers/bert/seed_123/full/standard/checkpoint',
 'roberta':A/'data/processed/transformers/roberta/seed_2026/full/standard/checkpoint',
 'distilbert':A/'data/processed/transformers/distilbert/seed_123/full/standard/checkpoint'}
architecture={}
for model,folder in folders.items():
 tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
 conf=json.loads((folder/'config.json').read_text(encoding='utf8'))
 architecture[model]=dict(hidden=conf.get('hidden_size',conf.get('dim')),layers=conf.get('num_hidden_layers',conf.get('n_layers')),
                         heads=conf.get('num_attention_heads',conf.get('n_heads')),tokenizer=type(tok).__name__,vocab_size=len(tok))
 for split,f in frames.items():
  counts=[];unks=0
  for start in range(0,len(f),1000):
   encoded=tok(f.text.iloc[start:start+1000].tolist(),truncation=False,add_special_tokens=True,verbose=False)['input_ids']
   counts.extend(map(len,encoded));unks+=sum(tok.unk_token_id in ids for ids in encoded)
  counts=np.array(counts)
  records.append(dict(model=model,split=split,n=len(f),max_length=128,truncated_rows=int((counts>128).sum()),
        max_original_tokens=int(counts.max()),p95_tokens=float(np.percentile(counts,95)),
        rows_with_unknown_token=unks,model_config_sha256=sha256(folder/'config.json')))
  print(model,split,records[-1],flush=True)
write(W/'encoding_audit.json',dict(records=records,architectures=architecture,
    tokenizer_scope='Actual cased BERT, RoBERTa BPE and uncased DistilBERT tokenizers from representative checkpoints. No tokenizer fitting.',
    tfidf_scope='Transform with saved train-only vectorizer; no classifier training.'))
