"""Khôi phục bộ kết quả A/B/C1/C2/C3 và checkpoint đã công bố.

Chuẩn bị demo không huấn luyện hoặc tìm ngưỡng; mọi file được kiểm SHA-256.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib,json,os,urllib.request,zipfile
from pathlib import Path
from src.paths import ROOT

PUBLISHED=ROOT/'results/final_submission'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def safe_target(root,name):
    target=(Path(root)/name).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('Đường dẫn ZIP vượt thư mục artifacts')
    return target

@lru_cache(maxsize=3)
def prepare_runtime(destination=None):
    """Khôi phục scores/biên nhận; không thay file hiện có khác mã băm."""
    manifest=read_json(PUBLISHED/'manifest.json')
    configured=destination or os.environ.get('GOEMOTIONS_FINAL_ARTIFACTS_ROOT')
    root=Path(configured).expanduser().resolve() if configured else ROOT/'artifacts/final_submission'
    archive=PUBLISHED/'runtime_bundle.zip'
    if digest(archive)!=manifest['runtime_bundle_sha256']:
        raise ValueError('Runtime bundle khác manifest của bộ nộp')
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if (len(bundle.namelist()) != len(set(bundle.namelist())) or
                set(bundle.namelist()) != set(manifest['files_sha256'])):
            raise ValueError('Nội dung runtime ZIP khác manifest')
        for name,expected in manifest['files_sha256'].items():
            path=safe_target(root,name)
            if path.exists():
                if digest(path)!=expected:
                    raise ValueError(f'Artifacts hiện có khác bộ nộp: {name}. Dùng thư mục đích mới.')
                continue
            payload=bundle.read(name)
            if hashlib.sha256(payload).hexdigest()!=expected:
                raise ValueError(f'File trong bundle khác mã băm: {name}')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return root

def restore_checkpoints(destination=None,selected_only=False):
    """Tải weights đã huấn luyện và kiểm ZIP; không chạy train."""
    root=prepare_runtime(destination);catalog=read_json(PUBLISHED/'checkpoint_assets.json')
    cache=ROOT/'artifacts/downloads';cache.mkdir(parents=True,exist_ok=True)
    for asset in catalog['assets']:
        if selected_only and asset.get('architecture') in ('bert','roberta','distilbert'):
            scores = [(read_json(root/f'data/processed/transformers/{asset["architecture"]}/seed_{seed}/full/standard/validation_metrics.json')['macro_f1'], -seed)
                      for seed in (42,123,2026)]
            representative = -max(scores)[1]
            if asset['seed']!=representative:
                continue
        expected=asset['weights']
        if all((root/name).is_file() and digest(root/name)==value for name,value in expected.items()):
            continue
        target=cache/asset['name']
        if not target.is_file() or digest(target)!=asset['sha256']:
            temporary=target.with_suffix('.download')
            request=urllib.request.Request(asset['download_url'],headers={'User-Agent':'GoEmotions-final-submission'})
            with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as out:
                while chunk:=response.read(2**20):out.write(chunk)
            if digest(temporary)!=asset['sha256']:
                raise ValueError('ZIP checkpoint sai mã băm: '+asset['name'])
            temporary.replace(target)
        with zipfile.ZipFile(target) as bundle:
            for item in bundle.infolist():
                if item.is_dir():continue
                path=safe_target(root,item.filename)
                if path.exists():continue
                path.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(item) as source,path.open('wb') as output:
                    while chunk:=source.read(2**20):output.write(chunk)
        for name,value in expected.items():
            if not (root/name).is_file() or digest(root/name)!=value:
                raise ValueError('Weights không khớp bộ nộp: '+name)
        print('Đã kiểm checkpoint:',asset['name'],flush=True)
    return root
