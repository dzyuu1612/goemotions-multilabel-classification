"""Đo suy luận thật trên cùng mẫu validation, một tiến trình cho mỗi hệ."""
import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/inference_v8'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from experiments.best_epochs_20261103.app_best_epochs import BestEpochSelector, MAIN, OLD
from src.apps.model_selector import ZeroShotDemo
from src.datasets.goemotions import sha256
from src.models.zero_shot import map_pipeline_scores

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=("baseline_balanced", "zero_shot", "bert", "roberta", "distilbert"))
parser.add_argument("validation", type=Path)
args = parser.parse_args()
if os.name != "nt":
    parser.error("Phép đo working set này dùng Windows.")
key = args.model
torch.set_num_threads(4)
frame = pd.read_parquet(args.validation)
indices = np.sort(np.random.default_rng(20261122).choice(len(frame), 512, replace=False))
texts = frame.iloc[indices]["text"].tolist()
ids = frame.iloc[indices]["id"].tolist()
gpu = key != "baseline_balanced"
selector = BestEpochSelector("cuda" if gpu else "cpu")
predictor = ZeroShotDemo(OLD, "cuda") if key == "zero_shot" else selector._load(key)
selector._predictor = predictor
thresholds = np.asarray(selector._limits(key, "tuned")[0], dtype=np.float32)
cpu = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())

if key == "baseline_balanced":
    files = [OLD / "data/processed/baseline/balanced/full/model.joblib"]
    truncated = None
elif key == "zero_shot":
    classifier = predictor.classifier
    tokenizer = classifier.tokenizer
    revision = json.loads((OLD / "data/processed/zero_shot/full/run_metadata.json").read_text(encoding="utf-8"))["model_revision"]
    folder = ROOT / "data/cache/huggingface/hub/models--facebook--bart-large-mnli/snapshots" / revision
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix in (".safetensors", ".json", ".txt")]
    max_pair = max(len(tokenizer(" ", predictor.template.format(label), verbose=False)["input_ids"])
                   for label in predictor.labels)
    lengths = tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
    truncated = sum(len(tokens) + max_pair > predictor.max_length for tokens in lengths)
else:
    seed = selector.current_selection["representatives"][key]
    receipt = next(row for row in selector.lock["runs"] if row["architecture"] == key and row["seed"] == seed)
    run = Path(receipt["run_dir"])
    if not run.exists():
        run = MAIN.parent / "checkpoints" / key / f"seed_{seed}"
    files = [f for f in (run / "checkpoint").iterdir() if f.is_file()
             and f.suffix in (".safetensors", ".json", ".txt", ".bin")]
    lengths = predictor.tokenizer(texts, truncation=False, verbose=False)["input_ids"]
    truncated = sum(len(tokens) > predictor.max_length for tokens in lengths)
assert files
print(key, "loaded", flush=True)

def predict(batch):
    if key == "baseline_balanced":
        points = predictor.model.predict_proba(batch)
    elif key == "zero_shot":
        results = classifier(batch, candidate_labels=predictor.labels, multi_label=True,
                             hypothesis_template=predictor.template, batch_size=16)
        points = map_pipeline_scores(results, predictor.labels)
    else:
        inputs = predictor.tokenizer(batch, truncation=True, padding=True, max_length=predictor.max_length,
                                     return_tensors="pt").to(predictor.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = predictor.model(**inputs).logits
        points = torch.sigmoid(logits.float()).cpu().numpy()
    points = np.asarray(points)
    assert points.shape == (len(batch), 28) and np.isfinite(points).all()
    return [[name for name, ok in zip(predictor.labels, row) if ok] for row in points >= thresholds]

def sync():
    if gpu:
        torch.cuda.synchronize()

raw, summaries = [], []
for batch_size in (1, 16):
    for j in range(50):
        start = (j * batch_size) % (len(texts) - batch_size + 1)
        predict(texts[start:start + batch_size])
    sync()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for j in range(0, len(texts), batch_size):
        batch = texts[j:j + batch_size]
        sync()
        start = perf_counter()
        predict(batch)
        sync()
        seconds = perf_counter() - start
        times.append(seconds)
        raw.append([key, batch_size, j // batch_size + 1, len(batch), seconds])
    q = 1024 ** 2
    stats = dict(batch=batch_size, n=len(texts), calls=len(times), mean_ms=1000*np.mean(times),
                 p50_ms=1000*np.median(times), p95_ms=1000*np.percentile(times, 95),
                 sentences_per_sec=len(texts)/sum(times),
                 vram_alloc_mib=torch.cuda.max_memory_allocated()/q if gpu else None,
                 vram_reserved_mib=torch.cuda.max_memory_reserved()/q if gpu else None,
                 ram_peak_mib=psutil.Process().memory_info().peak_wset/q)
    summaries.append(stats)
    print(key, stats, flush=True)

data = dict(model=key, device="CUDA" if gpu else "CPU", gpu=torch.cuda.get_device_name() if gpu else None,
            cpu=cpu, os=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
            cuda=torch.version.cuda, python=sys.version.split()[0], torch_threads=4,
            validation_sha256=sha256(args.validation), sample_ids=ids, sample_seed=20261122,
            warmup_batches=50, repeats=1, truncated=truncated,
            dtype="AMP fp16" if gpu and key != "zero_shot" else ("fp16" if gpu else "float64"),
            file_bytes={f.name:f.stat().st_size for f in files},
            bytes=sum(f.stat().st_size for f in files), summary=summaries)
(OUT / f"benchmark_{key}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUT / f"timings_{key}.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["model", "batch_size", "batch_index", "sentences", "seconds"])
    writer.writerows(raw)
