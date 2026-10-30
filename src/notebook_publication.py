"""Kiểm bản notebook công bố khớp source, đã chạy đủ cell và không có lỗi.

Digest chuẩn hóa newline LF để Windows và GitHub Actions Linux kiểm giống nhau.
Không suy đoán một cell đã chạy từ việc có output: phải có execution count tuần tự.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def notebook_digest(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def publication_errors(root, manifest):
    root = Path(root).resolve()
    errors, seen = [], set()
    sources = {p.relative_to(root).as_posix() for p in (root / "notebooks").glob("*.ipynb")}
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("notebooks"), list):
        return ["Manifest notebook sai schema"]
    for entry in manifest["notebooks"]:
        source_name, executed_name = entry.get("source", ""), entry.get("executed", "")
        source, executed = (root / source_name).resolve(), (root / executed_name).resolve()
        expected_name = f"notebooks/executed/{Path(source_name).stem}.executed.ipynb"
        if (source_name not in sources or source_name in seen or executed_name != expected_name
                or not source.is_relative_to(root) or not executed.is_relative_to(root)):
            errors.append(f"Manifest sai/trùng đường dẫn: {source_name}")
            continue
        seen.add(source_name)
        if not source.is_file() or not executed.is_file():
            errors.append(f"Thiếu source/output notebook: {source_name}")
            continue
        if (notebook_digest(source) != entry.get("source_sha256") or
                notebook_digest(executed) != entry.get("executed_sha256")):
            errors.append(f"Hash notebook không khớp: {source_name}")
        original = json.loads(source.read_text(encoding="utf-8"))
        ready = json.loads(executed.read_text(encoding="utf-8"))
        content = lambda nb: [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
        if content(original) != content(ready):
            errors.append(f"Output không chạy từ source hiện tại: {source_name}")
        if any(c.get("outputs") for c in original["cells"] if c["cell_type"] == "code") and original["cells"] != ready["cells"]:
            errors.append(f"Output trong notebook chính khác bản đã thực thi: {source_name}")
        code = [c for c in ready["cells"] if c["cell_type"] == "code"]
        if (not code or [c.get("execution_count") for c in code] != list(range(1, len(code) + 1))
                or entry.get("code_cells") != len(code) or entry.get("executed_code_cells") != len(code)
                or entry.get("error_cells") != 0):
            errors.append(f"Notebook chưa chạy đủ cell tuần tự: {source_name}")
        if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
            errors.append(f"Notebook có cell lỗi: {source_name}")
    if seen != sources:
        errors.append("Chưa có bản chạy cho mọi notebook source: " + ", ".join(sorted(sources - seen)))
    return errors
