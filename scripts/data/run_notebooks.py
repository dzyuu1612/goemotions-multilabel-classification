"""Chạy mọi cell bằng Python hiện tại, giữ source sạch, có thể xuất bản output.

Ví dụ: python -m scripts.data.run_notebooks --notebook 00_quick_start
       python -m scripts.data.run_notebooks --notebook all --publish --artifacts-root ../goemotions-multilabel-classification
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from src.paths import ROOT
from src.notebook_publication import notebook_digest


def main():
    names = [path.stem for path in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    parser = argparse.ArgumentParser(description="Run notebooks and keep executed copies local")
    parser.add_argument("--notebook", nargs="+", choices=["all", *names], default=["00_quick_start"])
    parser.add_argument("--publish", action="store_true", help="Lưu bản có output trong notebooks/executed/ để commit")
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout mỗi cell, giây")
    args = parser.parse_args()
    selected = names if "all" in args.notebook else args.notebook
    if len(set(selected)) != len(selected):
        parser.error("Không lặp tên notebook")
    if args.artifacts_root:
        os.environ["GOEMOTIONS_ARTIFACTS_ROOT"] = str(args.artifacts_root.resolve())
    output_dir = ROOT / ("notebooks/executed" if args.publish else "reports/notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema_version": 1, "notebooks": []}
    manifest["hash_policy"] = "sha256:utf8-lf"

    for name in selected:
        source = ROOT / "notebooks" / (name + ".ipynb")
        notebook = nbformat.read(source, as_version=4)
        started = datetime.now(timezone.utc).isoformat()
        client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        # Tránh chạy nhầm kernel của một Python khác trên máy.
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.on_cell_start = lambda cell, cell_index: print(
            f"{name} cell {cell_index + 1}/{len(notebook.cells)} ({cell.cell_type})", flush=True)
        try:
            client.execute()
        except Exception:
            failed = ROOT / "reports/notebooks" / (name + ".failed.ipynb")
            failed.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, failed)
            raise
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == "code"]
        if any(c.execution_count is None or any(o.output_type == "error" for o in c.outputs) for c in code):
            raise ValueError(f"{name}: thiếu cell đã chạy hoặc có lỗi; không xuất bản")
        output = output_dir / (name + ".executed.ipynb")
        nbformat.write(notebook, output)
        count = len(code)
        entry = {"source": source.relative_to(ROOT).as_posix(), "source_sha256": notebook_digest(source),
                 "executed": output.relative_to(ROOT).as_posix(), "executed_sha256": notebook_digest(output),
                 "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                 "code_cells": count, "executed_code_cells": count, "error_cells": 0,
                 "scope": notebook.metadata.get("goemotions", {}).get("scope", "learning example"),
                 "benchmark_training_repeated": False}
        manifest["notebooks"] = [e for e in manifest["notebooks"] if e["source"] != entry["source"]] + [entry]
        manifest["notebooks"].sort(key=lambda e: e["source"])
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        print(f"PASS {name}: {count} code cells; {output.relative_to(ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
