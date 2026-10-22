"""Kiểm chuyển mô hình, quyết định nhãn và đầu vào mà không tải model mạng."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.apps.model_selector import BaselineDemo, ModelSelector, format_scores


class DemoSelectorTest(unittest.TestCase):
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
            one = SimpleNamespace(predict=lambda text: ("BERT result", []))
            two = SimpleNamespace(predict=lambda text: ("RoBERTa result", []))
            with patch.object(selector, "_load", side_effect=[one, two, OSError("missing model")]) as load:
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

    def test_changing_thresholds_uses_same_weights_and_correct_label_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            selector = self.make_selector(folder)
            predictor = BaselineDemo.__new__(BaselineDemo)
            predictor.labels = [f"label_{i}" for i in range(28)]
            predictor.model = SimpleNamespace(predict_proba=lambda texts: np.full((1, 28), 0.6))
            per_label = np.full(28, 0.8)
            per_label[0] = 0.4
            with patch.object(selector, "_load", return_value=predictor) as load:
                with patch.object(selector, "details", return_value={"limits": {"global": 0.8, "tuned": per_label}}):
                    with patch.dict("sys.modules", {"torch": SimpleNamespace(
                            set_num_threads=lambda n: None,
                            cuda=SimpleNamespace(is_available=lambda: False))}):
                        _, fixed = selector.predict("hello", "baseline_standard", "fixed")
                        _, global_rows = selector.predict("hello", "baseline_standard", "global")
                        _, tuned = selector.predict("hello", "baseline_standard", "tuned")
                        self.assertEqual(sum(bool(r[3]) for r in fixed), 28)
                        self.assertEqual(sum(bool(r[3]) for r in global_rows), 0)
                        self.assertEqual([r[0] for r in tuned if r[3]], ["label_0"])
                        load.assert_called_once()
                        with self.assertRaises(ValueError):
                            selector.predict("hello", "baseline_standard", "unsupported")

    def test_custom_thresholds_cannot_be_applied_to_another_model(self):
        with tempfile.TemporaryDirectory() as folder:
            selector = self.make_selector(folder)
            selector.thresholds = Path(folder) / "thresholds.json"
            with self.assertRaises(ValueError):
                selector.limits("roberta", "custom")


if __name__ == "__main__":
    unittest.main()
