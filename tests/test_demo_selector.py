"""Kiểm chuyển mô hình, quyết định nhãn và đầu vào mà không tải model mạng."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.apps.model_selector import ModelSelector, format_scores


class DemoSelectorTest(unittest.TestCase):
    def test_default_label_follows_validation_winner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selected_model.json'
            path.write_text(json.dumps({'architecture': 'roberta'}), encoding='utf-8')
            selector = ModelSelector(folder, path, 'cpu')
            defaults = [key for name, key in selector.choices if 'mặc định' in name]
            self.assertEqual(defaults, ['roberta'])
            self.assertNotIn('mặc định của báo cáo', selector.describe('bert'))

    def test_missing_benchmark_c3_does_not_use_historical_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            selector = self.make_selector(folder)
            with patch('src.current_report.current_c3_root') as historical:
                with self.assertRaises(FileNotFoundError):
                    selector._load('distilbert')
                historical.assert_not_called()

    def test_c3_source_must_be_explicit_and_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selected_model.json'
            path.write_text(json.dumps({'architecture': 'bert'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                ModelSelector(folder, path, c3_source='unknown')

    def make_selector(self, folder):
        path = Path(folder) / "selected_model.json"
        path.write_text(json.dumps({"architecture": "bert"}), encoding="utf-8")
        return ModelSelector(folder, path, "cpu")

    def test_format_scores_does_not_force_neutral_and_uses_unrounded_threshold(self):
        names = [f"label_{i}" for i in range(27)] + ["neutral"]
        points = np.zeros(28)
        points[0] = 0.49999  # Hiển thị 0,5000 nhưng chưa đạt ngưỡng.
        message, rows = format_scores(names, points)
        self.assertIn("Chưa có nhãn", message)
        self.assertEqual(rows[0], ["label_0", 0.5, 0.5, ""])
        points[1] = 0.5
        message, rows = format_scores(names, points)
        self.assertIn("label_1", message)
        self.assertNotIn("neutral", message)
        self.assertEqual(len(rows), 28)

    def test_invalid_scores_are_rejected(self):
        names = [str(i) for i in range(28)]
        for points in (np.zeros(27), np.full(28, np.nan), np.full(28, 1.1)):
            with self.assertRaises(ValueError):
                format_scores(names, points)

    def test_input_validation_does_not_load_models(self):
        with tempfile.TemporaryDirectory() as folder:
            selector = self.make_selector(folder)
            with patch.object(selector, "_load") as load:
                self.assertEqual(selector.predict(" ", "bert")[1], [])
                self.assertEqual(selector.predict("a" * 20001, "zero_shot")[1], [])
                with self.assertRaises(ValueError):
                    selector.predict("hello", "unknown")
                load.assert_not_called()

    def test_switch_uses_new_model_and_failed_load_never_reuses_old_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            selector = self.make_selector(folder)
            one = SimpleNamespace(predict=lambda text, threshold: ("BERT result", []))
            two = SimpleNamespace(predict=lambda text, threshold: ("RoBERTa result", []))
            with patch.object(selector, "_load", side_effect=[one, two, OSError("missing model")]) as load:
                selector._limits = lambda key, mode: (0.5, "")
                with patch.dict("sys.modules", {"torch": SimpleNamespace(
                        set_num_threads=lambda n: None,
                        cuda=SimpleNamespace(is_available=lambda: False))}):
                    self.assertIn("BERT result", selector.predict("hello", "bert")[0])
                    selector.predict("another", "bert")
                    self.assertEqual(load.call_count, 1)
                    self.assertIn("RoBERTa result", selector.predict("hello", "roberta")[0])
                    with self.assertRaises(OSError):
                        selector.predict("hello", "zero_shot")
                    self.assertIsNone(selector._key)
                    self.assertIsNone(selector._predictor)


if __name__ == "__main__":
    unittest.main()
