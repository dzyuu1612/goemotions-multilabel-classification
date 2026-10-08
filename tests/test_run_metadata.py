"""Exporter chỉ kiểm/copy JSON GIẢ; không đọc dataset hay tải model/GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_run_metadata import (clear_previous_generated_files, copy_json, export_metadata)
from src.data import EXPECTED_ROWS, REVISION, sha256
from src.experiment import save_json
from src.neural import run_folder


class RunMetadataTest(unittest.TestCase):
    def test_copy_keeps_original_json_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            output = Path(temporary) / "export"
            payload = b'{ "seed" : 42, "revision": "abc" }\n'
            (root / "run.json").write_bytes(payload)
            record = copy_json(root, output, "run.json", "synthetic")
            self.assertEqual(record["status"], "exported")
            target = output / record["export_path"]
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(record["source_sha256"], record["export_sha256"])

    def test_raw_text_credentials_large_json_and_non_json_are_not_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary), Path(temporary) / "export"
            for name, payload, limit in (("raw.json", {"history": [{"text": "private example"}]}, 1024),
                                         ("secret.json", {"hf_token": "synthetic-secret"}, 1024),
                                         ("large.json", {"seed": 42}, 1),
                                         ("weights.txt", {"seed": 42}, 1024)):
                save_json(root / name, payload)
                record = copy_json(root, output, name, "synthetic", max_bytes=limit)
                self.assertEqual(record["status"], "invalid")
                self.assertIsNone(record["export_path"])
            self.assertFalse((output / "artifacts").exists())

    def test_outside_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            with self.assertRaises(ValueError):
                copy_json(root, Path(temporary) / "output", "../outside.json", "synthetic")

    def test_incomplete_existing_metadata_and_missing_files_do_not_create_placeholders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary), Path(temporary) / "export"
            save_json(root / "run.json", {"completed": False, "history": [{"epoch": 1}]})
            incomplete = copy_json(root, output, "run.json", "synthetic", permitted=False, reason="still training")
            missing = copy_json(root, output, "missing.json", "synthetic")
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertIsNotNone(incomplete["source_sha256"])
            self.assertEqual(missing["status"], "missing")
            self.assertFalse((output / "artifacts").exists())

    def test_partial_export_copies_only_completed_c_and_reports_missing_expected_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = [f"label_{i}" for i in range(28)]
            save_json(root / "data/labels.json", labels)
            completed = {"architecture": "bert", "seed": 42, "checkpoint": "google-bert/bert-base-cased",
                         "model_revision": "a" * 40, "data_revision": REVISION, "label_names": labels,
                         "smoke": False, "completed": True, "method": "C",
                         "sizes": {"train": EXPECTED_ROWS["train"], "validation": EXPECTED_ROWS["validation"]},
                         "history": [{"epoch": 1, "macro_f1": 0.5, "micro_f1": 0.6}]}
            done_folder = run_folder(root, "bert", 42)
            pending_folder = run_folder(root, "bert", 123)
            save_json(done_folder / "run_metadata.json", completed)
            save_json(done_folder / "checkpoint/config.json", {"problem_type": "multi_label_classification"})
            save_json(pending_folder / "run_metadata.json", dict(completed, completed=False, seed=123))

            def checked_run(folder, require_full=False):
                self.assertTrue(require_full)
                if Path(folder) == done_folder:
                    return completed
                raise ValueError("synthetic missing/incomplete run")

            with (patch("scripts.export_run_metadata.load_transformer_run", side_effect=checked_run),
                  patch("scripts.export_run_metadata.load_run_metadata", side_effect=FileNotFoundError("synthetic A missing")),
                  patch("scripts.export_run_metadata.validate_b_full", side_effect=FileNotFoundError("synthetic B missing"))):
                manifest = export_metadata(root)
            self.assertEqual(manifest["c_full_completed"], 1)
            self.assertEqual(manifest["c_full_expected"], 9)
            exported = [row["source_path"] for row in manifest["artifacts"] if row["status"] == "exported"]
            self.assertIn(done_folder.relative_to(root).as_posix() + "/run_metadata.json", exported)
            self.assertNotIn(pending_folder.relative_to(root).as_posix() + "/run_metadata.json", exported)
            self.assertTrue((root / "reports/reproducibility/manifest.json").is_file())
            self.assertTrue((root / "reports/reproducibility/README.md").is_file())

    def test_cleanup_removes_only_known_generated_copies_and_keeps_edited_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "export"
            root.mkdir()
            save_json(root / "first.json", {"seed": 42})
            save_json(root / "second.json", {"seed": 123})
            first = copy_json(root, output, "first.json", "synthetic")
            second = copy_json(root, output, "second.json", "synthetic")
            save_json(output / "manifest.json", {"artifacts": [first, second]})
            edited = output / second["export_path"]
            edited.write_text('{"human_note": "keep"}', encoding="utf-8")
            conflicts = clear_previous_generated_files(output)
            self.assertFalse((output / first["export_path"]).exists())
            self.assertTrue(edited.exists())
            self.assertEqual(conflicts, [second["export_path"]])
            save_json(output / "manifest.json", {"artifacts": [dict(second, status="conflict")]})
            self.assertEqual(clear_previous_generated_files(output), [second["export_path"]])
            self.assertTrue(edited.exists())
            self.assertTrue((root / "first.json").exists())


if __name__ == "__main__":
    unittest.main()
