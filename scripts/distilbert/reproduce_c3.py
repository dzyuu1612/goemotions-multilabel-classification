"""Train C3 using Huy's design, freeze validation choices, then evaluate test.

python -m scripts.distilbert.reproduce_c3 --output reports/c3_current_environment

Each seed is trained in a separate process. Existing completed runs are checked
before reuse; incomplete runs require a new output directory. Historical results
are never overwritten. Test labels are loaded only after all choices are frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from src.paths import ROOT
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.models.distilbert_study import (
    EmotionDataset, load_bundle, make_loader, read_json, score_loader, write_json,
)
from src.models.baseline import load_aligned_scores, tune_global_threshold, tune_thresholds
from src.evaluation.metrics import evaluate_multilabel

SEEDS = (42, 123, 2026)
METRICS = ("macro_f1", "micro_f1", "macro_precision", "macro_recall",
           "micro_precision", "micro_recall", "hamming_loss")
SOURCES = ("src/models/distilbert_study.py", "src/datasets/goemotions.py",
           "src/evaluation/metrics.py", "scripts/distilbert/train_distilbert.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_run(folder, config, source_hashes):
    meta = read_json(folder / "run.json")
    require(meta["status"] == "complete" and meta["mode"] == "full", "Incomplete full run")
    require(meta["config"] == config and meta["source_sha256"] == source_hashes,
            "Existing run differs from the frozen configuration/source")
    require(meta["n_train"] == 43410 and meta["n_validation"] == 5426 and not meta["test_used"],
            "Training must use only full official train and validation")
    for name, expected in meta["artifact_sha256"].items():
        require(sha256(folder / name) == expected, f"Changed training artifact: {name}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/distilbert_full.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    require(config["mode"] == "full", "This runner requires full training")
    hashes = {name: sha256(ROOT / name) for name in SOURCES}
    plan = {"seeds": list(SEEDS), "config": config, "data_revision": REVISION,
            "source_sha256": hashes, "config_sha256": sha256(args.config),
            "checkpoint_selection": "max validation Macro-F1 at 0.5; earliest epoch on tie",
            "representative_selection": "max validation Macro-F1 at 0.5; smaller seed on tie",
            "threshold_grid": [round(i / 20, 2) for i in range(1, 20)],
            "threshold_tie_rule": "closest to 0.5, then larger threshold",
            "test_used_for_selection": False}
    plan_file = output / "training_plan.json"
    if plan_file.exists():
        require(read_json(plan_file)["plan"] == plan, "The frozen training plan changed")
    else:
        write_json(plan_file, {"created_at_utc": now(), "plan": plan})
    torch.set_num_threads(config.get("cpu_threads", 4))
    metadata = {}
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        expected = dict(config, seed=seed)
        if not (folder / "run.json").exists():
            require(not folder.exists() or not any(folder.iterdir()),
                    f"Incomplete run at {folder}; use a new output directory")
            log = output / f"train_seed_{seed}.log"
            require(not log.exists(), f"Refusing to overwrite {log}")
            command = [sys.executable, "-u", "-m", "scripts.distilbert.train_distilbert",
                       "--config", str(args.config.resolve()), "--seed", str(seed),
                       "--output", str(folder)]
            print(f"START full training seed {seed}", flush=True)
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                           encoding="utf-8", stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT)
                try:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        print(line, end="", flush=True)
                    require(process.wait() == 0, f"Training failed; inspect {log}")
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
        metadata[seed] = checked_run(folder, expected, hashes)

    # Only train/validation have been opened at this point.
    frames, names, _ = load_goemotions(ROOT, write_metadata=False, splits=("train", "validation"))
    val_truth = multi_hot(frames["validation"].labels.tolist())
    train_support = multi_hot(frames["train"].labels.tolist()).sum(0)
    rare_ids = np.argsort(train_support)[:5]
    frozen, records, labels, histories, val_metrics = {}, [], [], [], {}
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        scores = load_aligned_scores(folder / "validation_scores.npz", frames["validation"].id, names)
        fixed = evaluate_multilabel(val_truth, scores, names, 0.5)
        require(fixed == read_json(folder / "validation_metrics.json"), "Saved validation metrics differ")
        global_value, curve = tune_global_threshold(val_truth, scores)
        per_label = tune_thresholds(val_truth, scores)
        threshold_artifact = {
            "run_metadata_sha256": sha256(folder / "run.json"), "seed": seed,
            "data_revision": REVISION, "label_names": names,
            "model_sha256": metadata[seed]["artifact_sha256"]["best/model.safetensors"],
            "scores_sha256": sha256(folder / "validation_scores.npz"),
            "thresholds": per_label.tolist(), "global_threshold": global_value,
            "selection_split": "validation", "test_used": False,
        }
        protocol_file = folder / "final_protocol.json"
        choices = [{"threshold_mode": "fixed", "thresholds": 0.5},
                   {"threshold_mode": "global", "thresholds": global_value},
                   {"threshold_mode": "tuned", "thresholds": per_label.tolist()}]
        protocol = {"seed": seed, "data_revision": REVISION, "label_names": names,
                    "run_metadata_sha256": sha256(folder / "run.json"),
                    "model_sha256": threshold_artifact["model_sha256"],
                    "validation_scores_sha256": sha256(folder / "validation_scores.npz"),
                    "configurations": choices, "test_used_for_selection": False}
        if protocol_file.exists():
            saved = read_json(protocol_file)
            require({k: v for k, v in saved.items() if k != "frozen_at_utc"} == protocol,
                    "Frozen protocol changed")
        else:
            write_json(folder / "thresholds_validation.json", threshold_artifact)
            write_json(folder / "global_validation_curve.json", curve)
            write_json(protocol_file, dict(protocol, frozen_at_utc=now()))
        frozen[seed] = read_json(protocol_file)
        val_metrics[seed] = fixed["macro_f1"]
        for h in read_json(folder / "history.json"):
            histories.append({"architecture": "distilbert", "seed": seed, **h})
    representative = max(SEEDS, key=lambda seed: (val_metrics[seed], -seed))
    write_json(output / "selection.json", {"representative_seed": representative,
               "selection_split": "validation", "test_used": False,
               "validation_macro_f1_fixed": val_metrics,
               "protocol_sha256": {str(s): sha256(output / f"seed_{s}/final_protocol.json") for s in SEEDS},
               "created_at_utc": now()})
    print(f"FROZEN all 3 checkpoints and thresholds; representative seed {representative}", flush=True)

    # Test is loaded after checkpoint, threshold and representative selection.
    test_frames, test_names, _ = load_goemotions(ROOT, write_metadata=False, splits=("test",))
    require(test_names == names, "Test label mapping differs")
    frames.update(test_frames)
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        bundle = load_bundle(folder, config["device"])
        for split in ("validation", "test"):
            dataset = EmotionDataset(frames[split], bundle["tokenizer"], config["max_length"])
            loader = make_loader(dataset, bundle["tokenizer"], config["eval_batch_size"])
            scores = score_loader(bundle["model"], loader, config["device"])
            truth = dataset.targets
            if split == "validation":
                reference = load_aligned_scores(folder / "validation_scores.npz", frames[split].id, names)
                np.testing.assert_allclose(scores, reference, rtol=0, atol=1e-6)
                replay_diff = float(np.max(np.abs(scores - reference)))
                # Thresholds were fitted on these exact stored scores.
                require(np.array_equal(scores >= 0.5, reference >= 0.5), "Reload changed fixed decisions")
                scores = reference
            else:
                destination = folder / "test_scores.npz"
                if destination.exists():
                    np.testing.assert_allclose(scores, load_aligned_scores(destination, frames[split].id, names),
                                               rtol=0, atol=1e-6)
                else:
                    np.savez_compressed(destination, ids=frames[split].id.to_numpy(dtype=str),
                                        scores=scores, label_names=np.asarray(names, dtype=str))
            results = []
            for choice in frozen[seed]["configurations"]:
                mode = choice["threshold_mode"]
                metric = evaluate_multilabel(truth, scores, names, choice["thresholds"])
                records.append({"system": "C_distilbert", "seed": seed, "split": split,
                                "threshold_mode": mode, "n_samples": len(truth),
                                **{key: metric[key] for key in METRICS}})
                labels.extend({"seed": seed, "split": split, "threshold_mode": mode,
                               "train_support": int(train_support[row["label_id"]]),
                               "rare_train_defined": row["label_id"] in rare_ids, **row}
                              for row in metric["per_label"])
                results.append({"threshold_mode": mode, **metric})
            if split == "test":
                write_json(folder / "test_results.json", {
                    "evaluated_at_utc": now(), "protocol_sha256": sha256(folder / "final_protocol.json"),
                    "scores_sha256": sha256(folder / "test_scores.npz"),
                    "validation_reload_max_abs_score_diff": replay_diff,
                    "n_samples": len(truth), "results": results})
            print(f"PASS real inference seed {seed} split {split}, N={len(truth)}", flush=True)
        del bundle, dataset, loader
        if config["device"] == "cuda":
            torch.cuda.empty_cache()
    rows = pd.DataFrame(records)
    rows.to_csv(output / "c3_runs.csv", index=False)
    pd.DataFrame(labels).to_csv(output / "per_label.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "history.csv", index=False)
    aggregated = []
    for (split, mode), part in rows.groupby(["split", "threshold_mode"]):
        aggregated.append({"system": "C_distilbert", "split": split, "threshold_mode": mode,
                           "n_runs": len(part), **{key + suffix: float(getattr(part[key], operation)())
                           for key in METRICS for suffix, operation in (("_mean", "mean"), ("_std", "std"))}})
    pd.DataFrame(aggregated).to_csv(output / "c3_mean_std.csv", index=False)
    write_json(output / "completion.json", {
        "completed_at_utc": now(), "training_repeated_from_pretrained": True,
        "seeds": list(SEEDS), "n_train": 43410, "n_validation": 5426, "n_test": 5427,
        "representative_seed": representative, "rare_labels_train_defined": [names[i] for i in rare_ids],
        "test_used_for_selection": False, "environment": metadata[SEEDS[0]]["environment"],
        "training_plan_sha256": sha256(plan_file), "runner_sha256": sha256(Path(__file__)),
        "outputs_sha256": {name: sha256(output / name) for name in
                           ("c3_runs.csv", "c3_mean_std.csv", "per_label.csv", "history.csv", "selection.json")}})
    print(rows.to_string(index=False), flush=True)
    print("PASS: 3 fresh full training runs; validation choices frozen before test.", flush=True)


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
