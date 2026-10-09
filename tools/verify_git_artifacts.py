"""Đối chiếu byte metadata gốc, bản bàn giao và blob Git.

Chạy sau git add: python tools/verify_git_artifacts.py
Chạy sau commit: python tools/verify_git_artifacts.py --revision HEAD
Không sửa artifacts, không đọc token hoặc chạy mô hình.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(path, revision):
    # Dùng argv riêng; path không được nội suy vào shell.
    reference = f"{revision}:{path}" if revision else f":{path}"
    return subprocess.run(["git", "show", reference], cwd=ROOT, check=True,
                          capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="", help="Mặc định kiểm index; HEAD kiểm commit")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reports/reproducibility/manifest.json").read_text(encoding="utf-8"))
    entries = [row for row in manifest["artifacts"] if row["status"] == "exported"]
    if len(entries) != manifest["n_exported"] or manifest["n_not_exported"]:
        raise ValueError("Manifest còn thiếu hoặc số artifact không khớp")
    checks = []
    for row in entries:
        path = "reports/reproducibility/" + row["export_path"]
        source_hash = sha((ROOT / row["source_path"]).read_bytes())
        copy_hash = sha((ROOT / path).read_bytes())
        blob_hash = sha(git_bytes(path, args.revision))
        checks.append({"path": path, "source_path": row["source_path"],
                       "expected_sha256": row["source_sha256"], "original_source_sha256": source_hash,
                       "source_copy_sha256": copy_hash, "git_blob_sha256": blob_hash,
                       "exact_bytes_equal": source_hash == copy_hash == blob_hash == row["source_sha256"]})
    paired = json.loads((ROOT / "reports/baseline_validation/paired_examples_manifest.json").read_text(encoding="utf-8"))
    for row in paired["outputs"]:
        path = row["path"]
        copy_hash, blob_hash = sha((ROOT / path).read_bytes()), sha(git_bytes(path, args.revision))
        checks.append({"path": path, "expected_sha256": row["sha256"],
                       "source_copy_sha256": copy_hash, "git_blob_sha256": blob_hash,
                       "exact_bytes_equal": copy_hash == blob_hash == row["sha256"]})
    proof = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "stage": f"Git revision {args.revision}" if args.revision else "Git index before commit",
             "revision_sha": subprocess.run(["git", "rev-parse", args.revision], cwd=ROOT,
                                            check=True, capture_output=True, text=True).stdout.strip()
                             if args.revision else None,
             "metadata_files_checked": len(entries), "paired_csv_files_checked": len(paired["outputs"]),
             "files_checked": len(checks), "passed": all(row["exact_bytes_equal"] for row in checks),
             "attribute": "reports/reproducibility/artifacts/** -text; paired_examples.csv -text",
             "checks": checks,
             "scope": "Exact byte/hash transport of exported metadata and teaching CSV; not training commit provenance"}
    name = "git_artifact_bytes_committed.json" if args.revision else "git_artifact_bytes.json"
    (ROOT / "reports/execution" / name).write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Git byte verification: {sum(row['exact_bytes_equal'] for row in checks)}/{len(checks)}")
    if not proof["passed"]:
        raise ValueError("Git blob khác byte đã ghi; kiểm .gitattributes trước bàn giao")


if __name__ == "__main__":
    main()
