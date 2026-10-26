"""Verify all six demo choices against real validation scores from one benchmark."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.apps.model_selector import ModelSelector
from src.datasets.goemotions import sha256
from src.evaluation.protocols import save_json
from src.models.baseline import load_aligned_scores
from src.models.transformer import load_transformer_run, run_folder
from src.paths import ROOT
from scripts.verification.reproduce_report import verify_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=ROOT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    root = args.artifacts_root.resolve()
    verified, frames, labels = verify_scores(root)
    selection = root/'data/processed/transformers/selected_model.json'
    selector = ModelSelector(root, selection, args.device, c3_source='benchmark')
    frame = frames['validation']
    references, seeds, source_hashes = {}, {}, {}
    for architecture in ('bert', 'roberta', 'distilbert'):
        candidates = [load_transformer_run(run_folder(root, architecture, seed), require_full=True)
                      for seed in (42, 123, 2026)]
        best = max(candidates, key=lambda m:(m['metrics']['macro_f1'], -m['seed']))
        seeds[architecture] = best['seed']
        path = run_folder(root, architecture, best['seed'])/'validation_scores.npz'
        references[architecture] = load_aligned_scores(path, frame.id, labels)
        source_hashes[path.relative_to(root).as_posix()] = sha256(path)
    for key, relative in (
        ('baseline_standard', 'data/processed/baseline/full/validation_scores.npz'),
        ('baseline_balanced', 'data/processed/baseline/balanced/full/validation_scores.npz'),
        ('zero_shot', 'data/processed/zero_shot/full/validation_scores.npz'),
    ):
        path = root/relative
        references[key] = load_aligned_scores(path, frame.id, labels)
        source_hashes[relative] = sha256(path)
    # Use the same three real validation comments for every demo choice. Avoid
    # threshold-adjacent examples because singleton FP16 padding can alter scores.
    clear = np.ones(len(frame), dtype=bool)
    for scores in references.values():
        clear &= np.min(np.abs(scores - 0.5), axis=1) > 0.02
    clear &= frame.text.eq(frame.text.str.strip()).to_numpy()
    indices = np.flatnonzero(clear)[:3]
    if len(indices) != 3:
        raise ValueError('Need three common validation cases away from thresholds')
    receipts = []
    for _, key in selector.choices:
        cases = []
        for index in indices:
            text = frame.iloc[index].text
            message, rows = selector.predict(text, key)
            if len(rows) != 28 or {r[0] for r in rows} != set(labels):
                raise ValueError(f'{key}: missing/duplicated label scores')
            by_label = {r[0]: r for r in rows}
            actual = np.array([by_label[label][1] for label in labels])
            expected = references[key][index]
            delta = float(np.max(np.abs(actual - expected)))
            predicted = [label for label in labels if by_label[label][3] == 'Có']
            expected_labels = [labels[j] for j in np.flatnonzero(expected >= 0.5)]
            if delta > 0.005 or predicted != expected_labels:
                raise ValueError(f'{key}/{frame.iloc[index].id}: demo differs from validation scores ({delta})')
            cases.append(dict(id=frame.iloc[index].id, text=text, predicted_labels=predicted,
                              max_score_difference=delta, message=message,
                              scores={label:float(by_label[label][1]) for label in labels}))
        empty = selector.predict(' ', key)
        overlong = selector.predict('a'*20001, key)
        if empty[1] or overlong[1]:
            raise ValueError(f'{key}: invalid input produced scores')
        receipts.append(dict(model=key, seed=seeds.get(key), status='PASS', cases=cases,
                             empty_input_rejected=True, overlong_input_rejected=True))
        print(f'PASS demo choice {key}: three validation cases and input checks', flush=True)
    save_json(ROOT/'reports/demo_all_models_verification.json', dict(
        checked_at_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
        scope='Real inference for all six choices; does not certify browser UI',
        selection=verified['selection'], selection_sha256=sha256(selection),
        score_tolerance=0.005, validation_score_hashes=source_hashes, models=receipts))


if __name__ == '__main__':
    main()
