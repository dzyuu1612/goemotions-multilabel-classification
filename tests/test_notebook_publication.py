"""Bản notebook công bố phải khớp source và chạy thật đủ cell, qua cả Git newline."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.notebook_publication import notebook_digest, publication_errors
from src.notebook_support import metrics_from_confusion
from src.evaluation.metrics import evaluate_multilabel


class NotebookPublicationTest(unittest.TestCase):
    def fixture(self, root):
        source = root / "notebooks/00_example.ipynb"
        executed = root / "notebooks/executed/00_example.executed.ipynb"
        executed.parent.mkdir(parents=True)
        original = {"cells": [{"cell_type": "code", "source": ["print(1)"],
                               "execution_count": None, "outputs": []}]}
        ready = {"cells": [{**original["cells"][0], "execution_count": 1,
                            "outputs": [{"output_type": "stream", "text": ["1\n"]}]}]}
        source.write_text(json.dumps(original), encoding="utf-8")
        executed.write_text(json.dumps(ready), encoding="utf-8")
        entry = {"source": source.relative_to(root).as_posix(), "source_sha256": notebook_digest(source),
                 "executed": executed.relative_to(root).as_posix(), "executed_sha256": notebook_digest(executed),
                 "code_cells": 1, "executed_code_cells": 1, "error_cells": 0}
        return source, executed, ready, {"schema_version": 1, "notebooks": [entry]}

    def test_matching_fully_executed_notebook_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, _, manifest = self.fixture(root)
            self.assertEqual(publication_errors(root, manifest), [])

    def test_changed_source_cannot_keep_old_output_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, _, manifest = self.fixture(root)
            source.write_text(source.read_text().replace("print(1)", "print(2)"))
            errors = publication_errors(root, manifest)
            self.assertTrue(any("Hash" in error for error in errors))
            self.assertTrue(any("source hiện tại" in error for error in errors))

    def test_missing_execution_count_is_rejected_even_with_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, path, ready, manifest = self.fixture(root)
            ready["cells"][0]["execution_count"] = None
            path.write_text(json.dumps(ready))
            manifest["notebooks"][0]["executed_sha256"] = notebook_digest(path)
            self.assertTrue(any("chạy đủ cell" in error for error in publication_errors(root, manifest)))

    def test_error_output_is_rejected_even_with_complete_execution_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, path, ready, manifest = self.fixture(root)
            ready["cells"][0]["outputs"] = [{"output_type": "error", "ename": "ValueError"}]
            path.write_text(json.dumps(ready))
            manifest["notebooks"][0]["executed_sha256"] = notebook_digest(path)
            self.assertTrue(any("cell lỗi" in error for error in publication_errors(root, manifest)))

    def test_digest_survives_windows_to_git_lf_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notebook.ipynb"
            path.write_bytes(b'{\r\n  "cells": []\r\n}\r\n')
            windows = notebook_digest(path)
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            self.assertEqual(notebook_digest(path), windows)

    def test_each_source_must_have_an_executed_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, _, manifest = self.fixture(root)
            (root / "notebooks/01_missing.ipynb").write_bytes(source.read_bytes())
            self.assertTrue(any("01_missing" in error for error in publication_errors(root, manifest)))

    def test_confusion_receipts_reconstruct_all_seven_metrics(self):
        truth = [[1, 0], [1, 1], [0, 1], [0, 0]]
        scores = [[0.9, 0.2], [0.1, 0.8], [0.7, 0.4], [0.1, 0.1]]
        original = evaluate_multilabel(truth, scores, ["a", "b"])
        recovered = metrics_from_confusion(pd.DataFrame(original["per_label"]), 4)
        for metric, value in recovered.items():
            self.assertAlmostEqual(value, original[metric])

    def test_bad_confusion_receipt_is_rejected(self):
        for values in ((1, 1, 1, 0), (1, -1, 1, 3), (0.5, 0.5, 1, 2)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    metrics_from_confusion(pd.DataFrame([values], columns=["tp", "fp", "fn", "tn"]), 4)


if __name__ == "__main__":
    unittest.main()
