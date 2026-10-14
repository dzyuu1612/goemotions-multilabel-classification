"""Chạy notebook bằng Python hiện tại; giữ output ở máy, không sửa source.

Ví dụ: python -m scripts.data.run_notebooks --notebook 00_quick_start
"""
import argparse
import sys

import nbformat
from nbclient import NotebookClient

from src.paths import ROOT


def main():
    names = [path.stem for path in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    parser = argparse.ArgumentParser(description="Run notebooks and keep executed copies local")
    parser.add_argument("--notebook", nargs="+", choices=names, default=["00_quick_start"])
    args = parser.parse_args()

    for name in args.notebook:
        source = ROOT / "notebooks" / (name + ".ipynb")
        notebook = nbformat.read(source, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        # Tránh chạy nhầm kernel của một Python khác trên máy.
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.execute()
        nbformat.validate(notebook)
        output = ROOT / "reports" / "notebooks" / (name + ".executed.ipynb")
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        count = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"PASS {name}: {count} code cells; {output.relative_to(ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
