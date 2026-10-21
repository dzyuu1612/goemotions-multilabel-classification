"""Chặn việc tráo bộ C3, trùng seed và lấy bundle cũ khi bundle mới thiếu."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.current_report import compare_report_records, current_c3_root
from src.models.transformer import METRIC_NAMES


class CurrentReportTest(unittest.TestCase):
    def records(self):
        rows = []
        for system in ('A_standard', 'A_balanced', 'B', 'C_bert', 'C_roberta', 'C_distilbert'):
            seeds = (42, 123, 2026) if system.startswith('C_') else ((None,) if system == 'B' else (42,))
            for seed in seeds:
                for split in ('validation', 'test'):
                    for mode in ('fixed', 'global', 'tuned'):
                        rows.append(dict(system=system, seed=seed, split=split, threshold_mode=mode,
                            n_samples=5426 if split == 'validation' else 5427,
                            **{metric: 0.25 for metric in METRIC_NAMES}))
        return rows

    def test_a_different_c3_score_cannot_be_published_as_the_report_snapshot(self):
        original = self.records()
        compare_report_records(original, original)
        changed = deepcopy(original)
        changed[-1]['macro_f1'] = 0.35
        with self.assertRaisesRegex(ValueError, 'Metric khác báo cáo'):
            compare_report_records(changed, original)

    def test_duplicate_or_missing_seed_rows_cannot_replace_a_full_report(self):
        rows = self.records()
        for changed in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.subTest(n=len(changed)), self.assertRaises(ValueError):
                compare_report_records(changed, rows)

    def test_missing_current_bundle_stops_instead_of_using_historical_c3(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'GOEMOTIONS_C3_RUN_ROOT': str(Path(folder)/'missing')}):
                with self.assertRaisesRegex(FileNotFoundError, 'không dùng bộ C3 cũ thay thế'):
                    current_c3_root()


if __name__ == '__main__':
    unittest.main()
