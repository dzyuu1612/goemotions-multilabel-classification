"""Check published source, executed notebooks and local README links without downloads."""
from __future__ import annotations
import ast,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.notebook_publication import publication_errors

def tracked_paths():
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    return sorted({n for n in raw.decode().split('\0') if n and (ROOT/n).is_file()})

def check_source(paths):
    errors=[];counts=dict(files=len(paths),python=0,notebooks=0,local_links=0)
    forbidden={'.docx','.doc','.pdf','.pptx','.xlsx','.joblib','.safetensors','.pt','.pth','.html'}
    retired=('assets/','deployment/','reports/','artifacts/','outputs/','private/','docs/','results/archive_','results/c3_rerun/','results/report_2_1/','results/training_20261025/','notebooks/executed/')
    secrets=[re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    manifest=ROOT/'notebooks/execution_manifest.json'
    try:
        value=json.loads(manifest.read_text(encoding='utf-8'))
        errors.extend(publication_errors(ROOT,value))
        published={e['source'] for e in value['notebooks']}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(f'Notebook manifest: {exc}');published=set()
    for name in paths:
        path=ROOT/name
        if name.startswith(retired) or path.suffix.lower() in forbidden:
            errors.append(f'{name}: generated/retired file in source tree')
        if path.suffix=='.zip' and name!='results/final_submission/runtime_bundle.zip':
            errors.append(f'{name}: unexpected archive')
        if path.suffix=='.py':
            counts['python']+=1
            try:
                tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
                for node in ast.walk(tree):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        module=node.module
                        if module.split('.')[0] in {'src','scripts','examples','tools','experiments'}:
                            target=ROOT.joinpath(*module.split('.'))
                            if not target.with_suffix('.py').is_file() and not (target/'__init__.py').is_file():
                                errors.append(f'{name}: missing import {module}')
            except (SyntaxError,UnicodeError) as exc:errors.append(f'{name}: {exc}')
        if path.suffix=='.ipynb':
            counts['notebooks']+=1
            if name not in published:errors.append(f'{name}: no execution receipt')
            if name.endswith('.executed.ipynb'):errors.append(f'{name}: duplicate output notebook')
        if path.suffix=='.md':
            for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                target=match.group(1).strip().strip('<>')
                if target.startswith(('#','http:','https:','mailto:','data:')):continue
                target=unquote(target.split('#',1)[0])
                if target:
                    counts['local_links']+=1
                    if not (path.parent/target).exists():errors.append(f'{name}: broken link {target}')
        if path.suffix in {'.py','.md','.toml','.yml','.txt','.json','.ipynb'}:
            text=path.read_text(encoding='utf-8-sig')
            if any(p.search(text) for p in secrets):errors.append(f'{name}: possible credential')
    if (ROOT/'results/final_submission/manifest.json').is_file():
        from src.final_submission import digest
        original=json.loads((ROOT/'results/final_submission/manifest.json').read_text(encoding='utf-8'))
        if digest(ROOT/'results/final_submission/runtime_bundle.zip')!=original['runtime_bundle_sha256']:
            errors.append('A/B runtime checksum mismatch')
    return counts,errors

def main():
    counts,errors=check_source(tracked_paths())
    print(json.dumps({'passed':not errors,**counts,'errors':errors},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
