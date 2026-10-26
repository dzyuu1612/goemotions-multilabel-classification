"""Check that validation replay never silently uses test rows or scores."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_multilabel
from src.notebook_support import replay_split


class ReplaySplitTest(TestCase):
    def test_train_is_rejected_before_reading_benchmark(self):
        with patch('src.notebook_support.benchmark') as checked:
            with self.assertRaises(ValueError):
                replay_split(Path('.'), 'A_standard', split='train')
            checked.assert_not_called()

    def test_validation_replay_uses_validation_ids_and_reference(self):
        labels = [str(i) for i in range(28)]
        frame = pd.DataFrame({'id':['validation-id'], 'text':['validation text'], 'labels':[[0]]})
        scores = np.zeros((1, 28)); scores[0, 0] = 0.8
        truth = np.zeros((1, 28), dtype=int); truth[0, 0] = 1
        metric = evaluate_multilabel(truth, scores, labels, 0.5)
        record = dict(metric, system='A_standard', seed=42, split='validation', threshold_mode='fixed')
        protocol = {'configurations':[dict(variant='standard', threshold_mode='fixed', thresholds=0.5)]}
        model = Mock(); model.predict_proba.return_value = scores
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (patch('src.notebook_support.benchmark', return_value=(
                      {'records':[record]}, {'validation':frame}, labels)),
                  patch('src.notebook_support.read_json', return_value=protocol),
                  patch('src.notebook_support.load_aligned_scores', return_value=scores) as load_scores,
                  patch('src.notebook_support.ROOT', root),
                  patch('joblib.load', return_value=model)):
                actual = replay_split(root, 'A_standard', split='validation')
                model.predict_proba.assert_called_once_with(['validation text'])
                self.assertEqual(load_scores.call_args.args[0], root/'data/processed/baseline/full/validation_scores.npz')
                self.assertEqual(actual.iloc[0]['split'], 'validation')
                with np.load(root/'reports/notebook_replay/A_standard_42_validation.npz') as saved:
                    self.assertEqual(saved['ids'].tolist(), ['validation-id'])
