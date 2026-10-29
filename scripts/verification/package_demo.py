"""Package the validation-selected BERT checkpoint; never train or retune."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from src.models.portable import pack_model,digest
from src.paths import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root',type=Path,default=ROOT/'artifacts/final_submission')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selection=json.loads((a.artifacts_root/'data/processed/transformers/selected_model.json').read_text(encoding='utf-8'))
    run=(a.artifacts_root/selection['run_dir']).resolve()
    if not run.is_relative_to(a.artifacts_root.resolve()):raise ValueError('Invalid selected run path')
    if digest(run/'run_metadata.json')!=selection['run_metadata_sha256']:raise ValueError('Selection hash differs')
    staging=a.output/'GoEmotions_Web'
    staging.mkdir(parents=True,exist_ok=True)
    model=staging/'GoEmotions_BERT.goemotions'
    pack_model(run,model)
    for name in ['web_demo.py','index.html','START_DEMO.cmd','requirements.txt','README.txt']:
        shutil.copy2(ROOT/'deployment'/name,staging/name)
    shutil.copy2(ROOT/'src/models/portable.py',staging/'portable.py')
    with zipfile.ZipFile(a.output/'GoEmotions_Web.zip','x',compression=zipfile.ZIP_STORED) as z:
        for file in sorted(staging.iterdir()):z.write(file,'GoEmotions_Web/'+file.name)
    result={'model':str(model),'model_sha256':digest(model),'model_bytes':model.stat().st_size,
            'web_zip':str(a.output/'GoEmotions_Web.zip'),'selection':selection,
            'inference_only':True,'runtime_included':False}
    (a.output/'package_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='selection'},ensure_ascii=True))


if __name__=='__main__':main()
