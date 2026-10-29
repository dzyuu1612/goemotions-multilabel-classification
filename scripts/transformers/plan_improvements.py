"""Create a reviewable C1/C2/C3 validation-only search; execution is opt-in.

Pilot 3 learning rates per architecture at seed 42. Confirm one selected
configuration per architecture with seeds 42/123/2026, then freeze thresholds.
Previously viewed test remains a descriptive benchmark, not a new blind test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.models.transformer import ARCHITECTURES, write_json, configure_console, load_transformer_run, run_folder
from src.paths import ROOT


def main():
    configure_console()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--smoke', action='store_true', help='64/32 rows per architecture, no benchmark claim')
    p.add_argument('--stage', choices=['pilot','confirm'], default='pilot')
    p.add_argument('--pilot-root', type=Path, help='Completed full pilot directory when stage=confirm')
    a = p.parse_args()
    target = a.output_root.resolve()
    protected = [ROOT/'artifacts/final_submission', ROOT/'results', ROOT/'data/processed']
    if any(target == x.resolve() or target.is_relative_to(x.resolve()) for x in protected):
        p.error('Choose a new experiments directory outside frozen results')
    entries=[]
    chosen={}
    if a.stage=='confirm':
        if a.smoke or a.pilot_root is None:p.error('confirm requires --pilot-root and no --smoke')
        for architecture in ARCHITECTURES:
            candidates=[]
            for lr in [2e-5,3e-5,5e-5]:
                folder=run_folder(a.pilot_root/f'lr_{lr:.0e}',architecture,42)
                if not (folder/'run_metadata.json').is_file():
                    p.error(f'Missing completed FULL pilot: {folder}. Smoke runs cannot select a final configuration.')
                meta=load_transformer_run(folder,require_full=True)
                if meta['config']['learning_rate']!=lr:raise ValueError('Pilot LR differs')
                candidates.append((meta['metrics']['macro_f1'],-meta['selected_epoch'],-lr,meta))
            chosen[architecture]=max(candidates,key=lambda row:row[:3])[3]
    for architecture in ARCHITECTURES:
        settings=([(chosen[architecture]['config']['learning_rate'],seed) for seed in [42,123,2026]]
                  if a.stage=='confirm' else [(lr,42) for lr in ([2e-5] if a.smoke else [2e-5,3e-5,5e-5])])
        for lr,seed in settings:
            candidate = target / f'lr_{lr:.0e}'
            command=[sys.executable,'-m','scripts.transformers.train_transformer',
                     '--architecture',architecture,'--seed',str(seed),'--learning-rate',str(lr),
                     '--epochs','8','--patience','2','--min-delta','0.001',
                     '--diagnostics','--output-root',str(candidate),'--device',a.device,'--resume']
            if a.smoke:command.append('--smoke')
            if a.stage=='confirm':command.extend(['--revision',chosen[architecture]['model_revision']])
            entries.append({'architecture':architecture,'learning_rate':lr,'seed':seed,
                            'status':'planned','command':command})
    plan={'phase':'smoke' if a.smoke else a.stage+'_validation_only',
          'selection':'best validation Macro-F1 @0.5; ties: fewer selected epochs, lower LR',
          'confirmation_seeds':[42,123,2026],
          'test_policy':'Do not open test in search. Existing observed test is descriptive.',
          'claim':'Planned configurations are not measured improvements.', 'runs':entries}
    path=target/'plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if [x['command'] for x in old['runs']] != [x['command'] for x in entries]:
            raise ValueError('Existing plan differs; choose a new output directory')
        plan=old
    write_json(path,plan)
    print(path)
    if a.execute:
        for row in plan['runs']:
            subprocess.run(row['command'],cwd=ROOT,check=True)
            row['status']='completed_smoke' if a.smoke else 'completed_pilot'
            write_json(path,plan)


if __name__=='__main__':main()
