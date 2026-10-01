"""Kiểm các lỗi có thể làm sai kết quả khi ghép ID và chọn ngưỡng."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baseline import load_aligned_scores, tune_thresholds


class BaselineTest(unittest.TestCase):
    def test_scores_are_joined_by_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.npz"
            np.savez(path, ids=np.array(["b", "a"]),
                     scores=np.array([[0.2, 0.8], [0.9, 0.1]]),
                     label_names=np.array(["x", "y"]))
            aligned = load_aligned_scores(path, ["a", "b"], ["x", "y"])
            np.testing.assert_allclose(aligned, [[0.9, 0.1], [0.2, 0.8]])
            with self.assertRaises(ValueError):
                load_aligned_scores(path, ["a", "c"], ["x", "y"])

    def test_tuning_uses_each_label_separately(self):
        truth = np.array([[1, 0], [0, 1], [0, 0]])
        scores = np.array([[0.4, 0.2], [0.2, 0.8], [0.1, 0.1]])
        thresholds = tune_thresholds(truth, scores)
        self.assertEqual(thresholds.shape, (2,))
        self.assertLessEqual(thresholds[0], 0.4)
        self.assertGreater(thresholds[0], 0.2)
        self.assertLessEqual(thresholds[1], 0.8)
        self.assertGreater(thresholds[1], 0.2)


if __name__ == "__main__":
    unittest.main()
