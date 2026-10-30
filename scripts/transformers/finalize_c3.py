"""Evaluate the three completed C3 runs, then update only C3 in the submission.

python -m scripts.transformers.finalize_c3
The old submission remains in results/archive_before_c3_rerun.
"""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.paths import ROOT
from src.datasets.goemotions import load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel
from src.evaluation.protocols import freeze_run, read_json, save_json
from src.models.baseline import load_aligned_scores
from src.models.transformer import METRIC_NAMES, load_transformer_run, run_folder, summarize_architectures
from scripts.transformers.evaluate_transformer_test import evaluate_run
from scripts.analysis.analyze_project_errors import analyze_arrays
from scripts.analysis.summarize_project import summarize

SEEDS = (42, 123, 2026)
NEW = ROOT / "results/c3_rerun"
OLD = ROOT / "results/archive_before_c3_rerun"
PUBLISHED = ROOT / "results/final_submission"
RUNTIME = ROOT / "artifacts/final_submission"


def replace_rows(name, rows, column, value):
    """Keep every non-C3 CSV row byte-for-byte, including original float text."""
    source = OLD / "report_data" / name
    destination = PUBLISHED / "report_data" / name
    raw = source.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    headers = next(csv.reader([lines[0]]))
    index = headers.index(column)
    # All benchmark CSV records occupy one line (no free-text fields).
    kept = [line for line in lines[1:] if next(csv.reader([line]))[index] != value]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    writer.writerows(rows)
    text = lines[0] + "".join(kept) + stream.getvalue()
    prefix = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    destination.write_bytes(prefix + text.encode("utf-8"))
    actual = destination.read_bytes().decode("utf-8-sig").splitlines(keepends=True)
    assert kept == [line for line in actual[1:] if next(csv.reader([line]))[index] != value]
    pd.DataFrame(rows, columns=headers).to_csv(NEW / "report_data" / name, index=False)
    return {"before_non_c3_sha256": __import__("hashlib").sha256("".join(kept).encode()).hexdigest(),
            "after_non_c3_sha256": __import__("hashlib").sha256("".join(kept).encode()).hexdigest(),
            "non_c3_rows": len(kept)}


def main():
    assert read_json(NEW / "training_complete.json")["seeds"] == list(SEEDS)
    if (NEW / "publication_complete.json").exists():
        raise FileExistsError("C3 publication already completed; inspect the saved receipts")
    out = NEW / "report_data"
    out.mkdir(exist_ok=True)
    metas = {s: load_transformer_run(run_folder(NEW, "distilbert", s), require_full=True) for s in SEEDS}
    # Freeze all validation choices before opening any test labels.
    for seed in SEEDS:
        folder = run_folder(NEW, "distilbert", seed)
        path = folder / "final_protocol.json"
        if not path.exists():
            save_json(path, freeze_run(ROOT, folder))
    representative = max(SEEDS, key=lambda s: (metas[s]["metrics"]["macro_f1"], -s))
    save_json(NEW / "selection.json", {"representative_seed": representative,
              "selection_split": "validation", "test_used": False})
    for seed in SEEDS:
        evaluate_run(run_folder(NEW, "distilbert", seed),
                     run_folder(NEW, "distilbert", seed) / "final_protocol.json", device="cuda")
    frames, labels, _ = load_goemotions(ROOT, write_metadata=False)
    truth = {split: multi_hot(frame.labels.tolist(), len(labels)) for split, frame in frames.items()}
    support = truth["train"].sum(0).astype(int)
    rare = sorted(range(28), key=lambda j: (support[j], j))[:5]
    records, per_label, curves = [], [], []
    for seed, meta in metas.items():
        folder = run_folder(NEW, "distilbert", seed)
        proto = read_json(folder / "final_protocol.json")
        for split in ("validation", "test"):
            scores = load_aligned_scores(folder / f"{split}_scores.npz", frames[split].id, labels)
            assert scores.shape == (5426 if split == "validation" else 5427, 28)
            for choice in proto["configurations"]:
                metric = evaluate_multilabel(truth[split], scores, labels, choice["thresholds"])
                records.append(dict(system="C_distilbert", seed=seed, split=split,
                    threshold_mode=choice["threshold_mode"], n_samples=len(scores),
                    **{k: metric[k] for k in METRIC_NAMES}))
                for row in metric["per_label"]:
                    per_label.append(dict(system="C_distilbert", architecture="distilbert", seed=seed,
                        split=split, threshold_mode=choice["threshold_mode"],
                        train_support=int(support[row["label_id"]]), **row))
        for row in meta["history"]:
            curves.append(dict(architecture="distilbert", seed=seed, epoch=row["epoch"],
                train_loss=row["train_loss"], validation_macro_f1=row["macro_f1"]))
        destination = run_folder(RUNTIME, "distilbert", seed)
        assert destination.resolve().is_relative_to(RUNTIME.resolve())
        shutil.rmtree(destination)
        shutil.copytree(folder, destination)
    pd.DataFrame([dict(seed=seed, epoch=h['epoch'], train_loss=h['train_loss'],
        validation_macro_f1=h['macro_f1'], validation_micro_f1=h['micro_f1'],
        validation_bce=h['validation_bce_eval'], epoch_seconds=h['epoch_seconds'],
        validation_seconds=h['validation_seconds'], selected_epoch=meta['selected_epoch'])
        for seed, meta in metas.items() for h in meta['history']]).to_csv(NEW / 'epoch_history.csv', index=False)
    pd.DataFrame([dict(seed=seed, selected_epoch=meta['selected_epoch'],
        validation_macro_f1=meta['metrics']['macro_f1'], elapsed_seconds=meta['elapsed_seconds'],
        started_at_utc=meta['created_at_utc'], completed_at_utc=meta['completed_at_utc'])
        for seed, meta in metas.items()]).to_csv(NEW / 'run_summary.csv', index=False)
    receipts = {}
    receipts["all_runs.csv"] = replace_rows("all_runs.csv", records, "system", "C_distilbert")
    receipts["mean_std.csv"] = replace_rows("mean_std.csv", summarize(records), "system", "C_distilbert")
    receipts["per_label.csv"] = replace_rows("per_label.csv", per_label, "architecture", "distilbert")
    test_per = [r for r in per_label if r["split"] == "test"]
    receipts["per_label_test.csv"] = replace_rows("per_label_test.csv", test_per, "architecture", "distilbert")
    receipts["rare_labels_test.csv"] = replace_rows("rare_labels_test.csv",
        [r for r in test_per if r["label_id"] in rare], "architecture", "distilbert")
    receipts["learning_curves.csv"] = replace_rows("learning_curves.csv", curves, "architecture", "distilbert")
    scores = load_aligned_scores(run_folder(NEW, "distilbert", representative) / "test_scores.npz", frames["test"].id, labels)
    counts, pairs, _, _ = analyze_arrays(frames["test"], labels, rare, support,
        {"distilbert": {"seed": representative}}, {"distilbert": scores}, {"distilbert": .5}, 5)
    for name, rows in (("error_counts.csv", counts), ("error_pairs.csv", pairs)):
        receipts[name] = replace_rows(name, rows, "architecture", "distilbert")
    all_pairs = pd.read_csv(PUBLISHED / "report_data/error_pairs.csv")
    wide = all_pairs.pivot_table(index=["missed_label", "extra_label"], columns="architecture",
                                values="sample_count", fill_value=0, aggfunc="sum").reset_index()
    wide["total"] = wide[["bert", "roberta", "distilbert"]].sum(axis=1)
    wide.sort_values(["total", "missed_label", "extra_label"], ascending=[False, True, True]).head(5).to_csv(
        PUBLISHED / "report_data/top_error_pairs.csv", index=False)
    matrices = []
    for arch in ("bert", "roberta", "distilbert"):
        lookup = {(r.missed_label, r.extra_label): int(r.sample_count)
                  for r in all_pairs[all_pairs.architecture == arch].itertuples()}
        matrices.extend(dict(architecture=arch, missed_label=a, extra_label=b,
            sample_count=lookup.get((a, b), 0)) for a in labels for b in labels)
    pd.DataFrame(matrices).to_csv(PUBLISHED / "report_data/error_pairs_28x28.csv", index=False)
    examples = read_json(OLD / "report_data/report_examples.json")
    by_id = {value: i for i, value in enumerate(frames["test"].id)}
    for row in examples:
        if row["architecture"] != "distilbert":
            continue
        i = by_id[row["id"]]; actual = truth["test"][i].astype(bool); guessed = scores[i] >= .5
        row.update(seed=representative, predicted_labels=np.array(labels)[guessed].tolist(),
            missed_labels=np.array(labels)[actual & ~guessed].tolist(),
            extra_labels=np.array(labels)[~actual & guessed].tolist(),
            scores_by_label=dict(zip(labels, map(float, scores[i]))))
    save_json(PUBLISHED / "report_data/report_examples.json", examples)
    all_runs = pd.read_csv(PUBLISHED / "report_data/all_runs.csv")
    old_runs = pd.read_csv(OLD / "report_data/all_runs.csv")
    comparison = []
    for split, mode in (("validation", "fixed"), ("test", "fixed"), ("test", "tuned")):
        before = old_runs[(old_runs.system == "C_distilbert") & (old_runs.split == split) & (old_runs.threshold_mode == mode)]
        after = all_runs[(all_runs.system == "C_distilbert") & (all_runs.split == split) & (all_runs.threshold_mode == mode)]
        for metric in METRIC_NAMES:
            comparison.append(dict(split=split, threshold_mode=mode, metric=metric,
                old_mean=before[metric].mean(), old_std=before[metric].std(ddof=1),
                new_mean=after[metric].mean(), new_std=after[metric].std(ddof=1),
                delta=after[metric].mean() - before[metric].mean()))
    pd.DataFrame(comparison).to_csv(NEW / "COMPARE_OLD_NEW.csv", index=False)
    # Refresh selection metadata without changing any A/B/C1/C2 run artifacts.
    groups = {a: [load_transformer_run(run_folder(RUNTIME, a, s), require_full=True) for s in SEEDS]
              for a in ("bert", "roberta", "distilbert")}
    ranking = summarize_architectures(groups)
    selection_path = RUNTIME / "data/processed/transformers/selected_model.json"
    selection = read_json(selection_path)
    assert ranking[0]["architecture"] == selection["architecture"], "Demo winner changed; update explicitly"
    selection["architecture_summary"] = ranking
    selection["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    for row in selection["run_sources"]:
        if row["architecture"] == "distilbert":
            row["run_metadata_sha256"] = sha256(run_folder(RUNTIME, "distilbert", row["seed"]) / "run_metadata.json")
    save_json(selection_path, selection)
    for name in ("all_runs.csv", "mean_std.csv", "per_label.csv", "per_label_test.csv", "rare_labels_test.csv", "learning_curves.csv"):
        target = RUNTIME / "reports/project_results" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PUBLISHED / "report_data" / name, target)
    save_json(NEW / "unchanged_non_c3.json", receipts)
    save_json(NEW / "publication_complete.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
        "representative_seed": representative, "n_train": 43410, "n_validation": 5426,
        "n_test": 5427, "labels": labels, "ddof": 1, "test_used_for_selection": False,
    })
    print(f"C3 published: seeds={SEEDS}; representative={representative}; non-C3 CSV bytes preserved", flush=True)


if __name__ == "__main__":
    main()
