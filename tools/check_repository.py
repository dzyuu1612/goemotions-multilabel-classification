"""Kiểm source sạch và bản notebook đã chạy trong notebooks/executed/.

Chạy từ gốc repo: python tools/check_repository.py
Chỉ đọc tệp/Git index; không tải dataset hoặc model.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

# Chạy trực tiếp python tools/check_repository.py cũng tìm được package src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.notebook_publication import publication_errors

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".docx", ".doc", ".pdf", ".tex", ".html", ".pptx", ".xlsx", ".zip",
                      ".joblib", ".safetensors", ".pt", ".pth"}
GENERATED_PREFIXES = ("reports/", "data/raw/", "data/processed/", "data/cache/",
                      "data/tokenizers/", "artifacts/", "outputs/", "private/", "docs/")
PUBLISHED_REPORT_PREFIXES = (
    "reports/project_results/", "reports/errors_test_standard_fixed/",
    "reports/errors_validation_standard_fixed/", "reports/tables/", "reports/figures/",
    "reports/reproduction_report_20261026/",
)
PUBLISHED_REPORT_FILES = {
    "reports/demo_all_models_verification.json", "reports/demo_http_verification.json",
    "reports/demo_verification.json", "reports/environment.json", "reports/summary.json",
    "reports/verification.json",
}
RETIRED_GENERATORS = {"build_report_docx.py", "build_ieee_paper.py", "build_progress_reports.py",
                      "update_report_results.py", "verify_final_reports.py", "report_c3_full.py"}
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
SECRETS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def tracked_paths():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                            check=True, stdout=subprocess.PIPE)
    # Kiểm cây đang làm việc, gồm code mới và đổi chỗ chưa stage; không cần git add.
    return sorted({path for path in result.stdout.decode("utf-8").split("\0")
                   if path and (ROOT / path).is_file()})


def check_source(paths):
    errors = []
    counts = {"files": len(paths), "python": 0, "notebooks": 0, "local_links": 0}
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"{name}: tệp đã đổi chỗ nhưng chưa git add")
            continue
        published_report = name.startswith(PUBLISHED_REPORT_PREFIXES) or name in PUBLISHED_REPORT_FILES
        runtime_zip = name == "results/final_submission/runtime_bundle.zip"
        if ((name.startswith(GENERATED_PREFIXES) and not published_report)
                or (path.suffix.lower() in FORBIDDEN_SUFFIXES and not runtime_zip)
                or path.name in RETIRED_GENERATORS):
            errors.append(f"{name}: tệp ngoài phạm vi source")
        if path.suffix.lower() in {".md", ".markdown"} and name != "README.md":
            errors.append(f"{name}: Markdown chỉ được giữ README.md")
        if path.suffix == ".py":
            counts["python"] += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        if module.split(".")[0] in {"src", "scripts", "examples", "tools"}:
                            target = ROOT.joinpath(*module.split("."))
                            if not target.with_suffix(".py").is_file() and not (target/"__init__.py").is_file():
                                errors.append(f"{name}: import local không tồn tại: {module}")
            except (SyntaxError, UnicodeError) as error:
                errors.append(f"{name}: {error}")
        if path.suffix == ".ipynb":
            counts["notebooks"] += 1
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                errors.append(f"{name}: sai định dạng notebook")
            published = name.startswith("notebooks/executed/")
            for index, cell in enumerate(notebook.get("cells", []), 1):
                if cell.get("cell_type") == "code" and (
                    cell.get("outputs") or cell.get("execution_count") is not None
                ) and not published:
                    errors.append(f"{name}: cell {index} có output/execution count")
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            # Chỉ kiểm link Markdown; đường dẫn output trong inline code không phải link.
            for match in LINK.finditer(text):
                target = match.group(1).strip().strip("<>")
                if target.startswith(("#", "http:", "https:", "mailto:", "data:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                counts["local_links"] += 1
                if not (path.parent / target).exists():
                    errors.append(f"{name}: link local thiếu: {target}")
        if path.suffix in {".py", ".md", ".toml", ".yml", ".txt", ".json", ".ipynb"}:
            text = path.read_text(encoding="utf-8-sig")
            if any(pattern.search(text) for pattern in SECRETS):
                errors.append(f"{name}: chuỗi giống secret cần kiểm")
    published_paths = [name for name in paths if name.startswith("notebooks/executed/")]
    if published_paths:
        manifest = ROOT / "notebooks/executed/execution_manifest.json"
        if not manifest.is_file():
            errors.append("Thiếu execution_manifest.json cho notebook công bố")
        else:
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                errors.extend(publication_errors(ROOT, value))
                allowed = {entry["executed"] for entry in value["notebooks"]} | {
                    "notebooks/executed/execution_manifest.json"}
                errors.extend(f"Output notebook ngoài manifest: {name}" for name in published_paths if name not in allowed)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"Manifest/output notebook không hợp lệ: {error}")
    result_paths = {name for name in paths if name.startswith('results/report_2_1/')}
    if result_paths:
        try:
            base = ROOT / 'results/report_2_1'
            manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
            expected_paths = {'results/report_2_1/' + name for name in manifest['files_sha256']}
            expected_paths.add('results/report_2_1/manifest.json')
            if result_paths != expected_paths:
                errors.append('Bộ kết quả báo cáo 2.1 thiếu hoặc thừa tệp so với manifest')
            for name, digest in manifest['files_sha256'].items():
                target = (base / name).resolve()
                if not target.is_relative_to(base.resolve()) or not target.is_file():
                    errors.append(f'Tệp kết quả không hợp lệ: {name}')
                elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    errors.append(f'Hash kết quả báo cáo 2.1 không khớp: {name}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'Manifest kết quả báo cáo không hợp lệ: {error}')
    if (ROOT / "results/final_submission/manifest.json").is_file():
        try:
            from tempfile import TemporaryDirectory
            from src.final_submission import prepare_runtime
            with TemporaryDirectory() as folder:
                prepare_runtime(folder)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Runtime bộ nộp không hợp lệ: {error}")
    return counts, errors


def main():
    counts, errors = check_source(tracked_paths())
    print(json.dumps({"passed": not errors, **counts, "errors": errors},
                     ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
