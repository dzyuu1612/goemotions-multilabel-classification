"""Run the fixed C3 experiment once for three seeds using the C1/C2 trainer.

python -m scripts.transformers.rerun_c3
Archives the previous submission and C3 checkpoints before starting training.
Evaluation/report export is a separate step after all three runs complete.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from src.paths import ROOT
from src.models.transformer import run_folder, training_config, write_json

SEEDS = (42, 123, 2026)
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
OUTPUT = ROOT / "results/c3_rerun"


def hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def main():
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("C3 requires the available CUDA environment used by C1/C2")
    if OUTPUT.exists():
        raise FileExistsError("This experiment already started. Inspect the log; do not rerun completed seeds.")
    current = ROOT / "results/final_submission"
    backup = ROOT / "results/archive_before_c3_rerun"
    archive = ROOT / "results/archive_c3_old"
    if backup.exists() or archive.exists():
        raise FileExistsError("Archive already exists; inspect it before starting any training")
    before = hashes(current)
    shutil.copytree(current, backup)
    assert hashes(backup) == before
    runtime = ROOT / "artifacts/final_submission"
    for seed in SEEDS:
        source = run_folder(runtime, "distilbert", seed)
        shutil.copytree(source, archive / f"seed_{seed}")
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "before_sha256.json", before)
    write_json(archive / "metadata.json", {
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": {str(s): json.loads((archive / f"seed_{s}/run_metadata.json").read_text(encoding="utf-8"))
                 for s in SEEDS}, "files_sha256": hashes(archive),
    })
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "amp_fp16": True,
        "gradient_checkpointing": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "trainer_sha256": hashlib.sha256((ROOT / "scripts/transformers/train_transformer.py").read_bytes()).hexdigest(),
        "config": training_config("distilbert", device="cuda", revision=REVISION),
    }
    write_json(OUTPUT / "ENV.json", environment)
    (OUTPUT / "ENV.txt").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "DECISION.md").write_text(
        f"# Fixed C3 experiment\n\nRecorded before training: {datetime.now(timezone.utc).isoformat()}\n\n"
        "One run for each seed {42, 123, 2026}. These three runs are the main C3 results, "
        "regardless of whether their scores are higher or lower than the archived runs. "
        "No hyperparameter search. Validation selects checkpoints and thresholds; test does not.\n\n"
        "Technical failures may restart only the affected seed with the same configuration, "
        "with a written incident log.\n", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
    for seed in SEEDS:
        command = [sys.executable, "-u", "-m", "scripts.transformers.train_transformer",
                   "--architecture", "distilbert", "--seed", str(seed), "--device", "cuda",
                   "--revision", REVISION, "--output-root", str(OUTPUT)]
        print(f"START C3 seed={seed}", flush=True)
        with (OUTPUT / f"train_seed_{seed}.log").open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        print(f"COMPLETE C3 seed={seed}", flush=True)
    write_json(OUTPUT / "training_complete.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
        "original_submission_unchanged": hashes(current) == before,
    })


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
