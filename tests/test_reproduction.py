"""Kiểm công cụ tái lập dừng khi số bị sửa; không mở dữ liệu/model thật."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.analysis.summarize_project import summarize
from scripts.verification.reproduce_report import (compare_averages, compare_metrics,
                                                    record_key, verify_scores, verify_word_tables)
from src.models.transformer import METRIC_NAMES


def synthetic_records():
    rows = []
    systems = {"A_standard": [42], "A_balanced": [42], "B_bart_mnli": [None],
               "C_bert": [42, 123, 2026], "C_roberta": [42, 123, 2026],
               "C_distilbert": [42, 123, 2026]}
    for system, seeds in systems.items():
        for index, seed in enumerate(seeds):
            for split in ("validation", "test"):
                for mode in ("fixed", "global", "tuned"):
                    rows.append({"system": system, "seed": seed, "split": split,
                                 "threshold_mode": mode,
                                 **{key: 0.2 + 0.1 * index for key in METRIC_NAMES}})
    return rows


class ReproductionTest(unittest.TestCase):
    def test_unchanged_seeds_reproduce_mean_and_sample_std(self):
        rows = synthetic_records()
        result = compare_averages(rows, summarize(rows))
        self.assertEqual(len(result), 36)
        bert = next(row for row in result if row["system"] == "C_bert")
        self.assertAlmostEqual(bert["macro_f1_std"], 0.1)

    def test_altered_mean_or_population_std_is_rejected(self):
        rows = synthetic_records()
        for field, value in (("macro_f1_mean", 0.9), ("macro_f1_std", 0.081649658)):
            with self.subTest(field=field):
                saved = summarize(rows)
                next(row for row in saved if row["system"] == "C_bert")[field] = value
                with self.assertRaisesRegex(ValueError, "Sai tổng hợp"):
                    compare_averages(rows, saved)

    def test_fabricated_std_for_single_run_is_rejected(self):
        rows = synthetic_records()
        saved = summarize(rows)
        next(row for row in saved if row["system"] == "A_standard")["macro_f1_std"] = 0.0
        with self.assertRaisesRegex(ValueError, "Không tạo std"):
            compare_averages(rows, saved)

    def test_duplicate_or_missing_aggregate_is_rejected(self):
        rows = synthetic_records()
        saved = summarize(rows)
        for altered in (saved[:-1], saved[:-1] + [copy.deepcopy(saved[0])]):
            with self.assertRaisesRegex(ValueError, "36 nhóm"):
                compare_averages(rows, altered)

    def test_changed_metric_or_nan_cannot_pass(self):
        expected = {key: 0.5 for key in METRIC_NAMES}
        self.assertEqual(compare_metrics(expected, expected, "unchanged"), 0.0)
        for bad in (0.6, float("nan")):
            with self.assertRaisesRegex(ValueError, "Metric khác"):
                compare_metrics({**expected, "macro_f1": bad}, expected, "changed")

    def test_missing_dataset_stops_before_any_possible_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("scripts.verification.reproduce_report.load_goemotions") as loader:
                with self.assertRaisesRegex(ValueError, "Thiếu dữ liệu cục bộ"):
                    verify_scores(Path(temporary))
                loader.assert_not_called()

    def test_report_without_expected_tables_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>')
            with self.assertRaisesRegex(ValueError, "9 bảng/111 ô"):
                verify_word_tables(path, {"records": [], "averages": []})

    def test_csv_seed_normalization_keeps_zero_shot_seed_empty(self):
        row = {"system": "C_bert", "seed": "123.0", "split": "test", "threshold_mode": "fixed"}
        self.assertEqual(record_key(row), ("C_bert", 123, "test", "fixed"))
        self.assertIsNone(record_key({**row, "seed": ""})[1])


if __name__ == "__main__":
    unittest.main()
