"""Hand-calculated checks for the shared multi-label metric contract."""

import unittest

import numpy as np

from src.metrics import evaluate_multilabel


class MetricsTest(unittest.TestCase):
    def test_hand_calculated_micro_macro_and_hamming(self):
        truth = np.array([[1, 0], [1, 1], [0, 1]])
        scores = np.array([[0.9, 0.4], [0.2, 0.8], [0.7, 0.8]])
        result = evaluate_multilabel(truth, scores, ["a", "b"])
        self.assertAlmostEqual(result["macro_f1"], 0.75)
        self.assertAlmostEqual(result["micro_f1"], 0.75)
        self.assertAlmostEqual(result["hamming_loss"], 2 / 6)
        self.assertEqual([x["support"] for x in result["per_label"]], [2, 2])

    def test_no_positive_predictions_is_defined(self):
        result = evaluate_multilabel(np.array([[1, 0]]), np.array([[0.1, 0.1]]), ["a", "b"])
        self.assertEqual(result["macro_f1"], 0)
        self.assertEqual(result["zero_division"], 0)

    def test_each_label_can_have_its_own_threshold(self):
        truth = np.array([[1, 0], [0, 1]])
        scores = np.array([[0.4, 0.3], [0.2, 0.7]])
        fixed = evaluate_multilabel(truth, scores, ["a", "b"], threshold=0.5)
        separate = evaluate_multilabel(truth, scores, ["a", "b"], threshold=[0.4, 0.8])
        self.assertLess(fixed["macro_f1"], 1)
        self.assertEqual(separate["threshold"], [0.4, 0.8])
        self.assertEqual(separate["macro_f1"], 0.5)

    def test_wrong_number_of_thresholds_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_multilabel(np.array([[1, 0]]), np.array([[0.9, 0.1]]),
                                ["a", "b"], threshold=[0.5])


if __name__ == "__main__":
    unittest.main()
