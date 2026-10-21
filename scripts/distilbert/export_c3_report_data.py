"""Export checked report tables and errors after fresh C3 training completes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import ROOT
from src.datasets.goemotions import load_goemotions, multi_hot, sha256
from src.models.distilbert_study import read_json, write_json
from src.models.baseline import load_aligned_scores
from src.evaluation.metrics import evaluate_multilabel
from scripts.analysis.summarize_project import summarize
from scripts.analysis.analyze_project_errors import analyze_arrays
from scripts.distilbert.reproduce_c3 import SEEDS, METRICS, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    args = parser.parse_args()
    run_root, historical = args.run_root.resolve(), args.historical_root.resolve()
    output = run_root / "report_data"
    output.mkdir(parents=True, exist_ok=True)
    complete = read_json(run_root / "completion.json")
    require(complete["training_repeated_from_pretrained"], "Fresh training is required")
    for name, digest in complete["outputs_sha256"].items():
        require(sha256(run_root / name) == digest, f"Changed C3 output {name}")
    audit = read_json(run_root / "historical_verification/verification.json")
    require(audit["passed"] and audit["verified_rows"] == 72, "Historical results are not verified")
    old = pd.DataFrame(audit["records"])
    new = pd.read_csv(run_root / "c3_runs.csv")
    require(len(new) == 18, "Need all 18 replacement C3 rows")
    merged = pd.concat([old[old.system != "C_distilbert"], new], ignore_index=True)
    require(len(merged) == 72 and not merged.duplicated(["system", "seed", "split", "threshold_mode"]).any(),
            "Report must contain 72 unique result records")
    means = pd.DataFrame(summarize(merged.to_dict("records")))
    require(len(means) == 36, "Report must contain 36 summary groups")
    merged.to_csv(output / "all_runs.csv", index=False)
    means.to_csv(output / "mean_std.csv", index=False)
    frames, names, _ = load_goemotions(ROOT, write_metadata=False)
    train_support = multi_hot(frames["train"].labels.tolist()).sum(0)
    rare_ids = np.argsort(train_support)[:5].tolist()
    representatives, arrays, histories, per_labels = {}, {}, [], []
    for arch in ("bert", "roberta", "distilbert"):
        part = merged[(merged.system == "C_" + arch) & (merged.split == "validation") &
                      (merged.threshold_mode == "fixed")].sort_values(["macro_f1", "seed"], ascending=[False, True])
        rep = int(part.iloc[0].seed)
        representatives[arch] = {"seed": rep}
        for seed in SEEDS:
            if arch == "distilbert":
                folder = run_root / f"seed_{seed}"
                meta = read_json(folder / "run.json")
                history = read_json(folder / "history.json")
                epoch = meta["best_epoch"]
            else:
                folder = historical / f"data/processed/transformers/{arch}/seed_{seed}/full/standard"
                meta = read_json(folder / "run_metadata.json")
                history, epoch = meta["history"], meta["selected_epoch"]
            for row in history:
                histories.append({"architecture": arch, "seed": seed, "epoch": row["epoch"],
                                  "train_loss": row["train_loss"],
                                  "validation_macro_f1": row.get("validation_macro_f1", row.get("macro_f1"))})
            scores = load_aligned_scores(folder / "test_scores.npz", frames["test"].id, names)
            if seed == rep:
                representatives[arch].update(selected_epoch=epoch, run_dir=str(folder))
                arrays[arch] = scores
            protocol = read_json(folder / "final_protocol.json")
            result = read_json(folder / "test_results.json")
            require(result["protocol_sha256"] == sha256(folder / "final_protocol.json") and
                    result["scores_sha256"] == sha256(folder / "test_scores.npz"), "Changed test artifact")
            for choice in protocol["configurations"]:
                mode = choice["threshold_mode"]
                metric = evaluate_multilabel(multi_hot(frames["test"].labels.tolist()), scores, names,
                                            choice["thresholds"])
                reference = merged[(merged.system == "C_" + arch) & (merged.seed == seed) &
                                   (merged.split == "test") & (merged.threshold_mode == mode)].iloc[0]
                require(all(abs(metric[key] - reference[key]) <= 1e-12 for key in METRICS),
                        "Report metrics differ from actual test scores")
                per_labels.extend({"architecture": arch, "seed": seed, "threshold_mode": mode,
                                   "train_support": int(train_support[row["label_id"]]), **row}
                                  for row in metric["per_label"])
    require(representatives["distilbert"]["seed"] == complete["representative_seed"],
            "C3 representative selection changed")
    counts, pairs, _, examples = analyze_arrays(frames["test"], names, rare_ids, train_support,
                                               representatives, arrays, {arch: 0.5 for arch in arrays})
    pd.DataFrame(counts).to_csv(output / "error_counts.csv", index=False)
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(output / "error_pairs.csv", index=False)
    top_pairs = pair_frame.pivot_table(index=["missed_label", "extra_label"], columns="architecture",
                                      values="sample_count", fill_value=0, aggfunc="sum").reset_index()
    top_pairs["total"] = top_pairs[["bert", "roberta", "distilbert"]].sum(1)
    top_pairs = top_pairs.sort_values(["total", "missed_label", "extra_label"],
                                      ascending=[False, True, True]).head(5)
    top_pairs.to_csv(output / "top_error_pairs.csv", index=False)
    pd.DataFrame(examples).to_csv(output / "error_examples.csv", index=False)
    # Retain the same three report examples, independently recomputing all labels.
    prefixes = ("This!!! 🐃 and 💍 for your hard work!",
                "Try nonchalantly handing them your card as if they had dropped it.",
                "I always plan that, my wife usually has other ideas though.")
    selected_ids = []
    for prefix in prefixes:
        matches = frames["test"][frames["test"].text.str.startswith(prefix)]
        require(len(matches) == 1, "Each supplied report example must have one official ID")
        selected_ids.append(matches.iloc[0].id)
    example_rows = []
    test_truth = multi_hot(frames["test"].labels.tolist()).astype(bool)
    for example_id in selected_ids:
        matches = np.flatnonzero(frames["test"].id.to_numpy() == example_id)
        require(len(matches) == 1, "Report example must have one official ID")
        i = int(matches[0])
        for arch, scores in arrays.items():
            predicted = scores[i] >= 0.5
            example_rows.append({"id": example_id, "text": frames["test"].text.iloc[i],
                "architecture": arch, "seed": representatives[arch]["seed"],
                "true_labels": [names[j] for j in np.flatnonzero(test_truth[i])],
                "predicted_labels": [names[j] for j in np.flatnonzero(predicted)],
                "missed_labels": [names[j] for j in np.flatnonzero(test_truth[i] & ~predicted)],
                "extra_labels": [names[j] for j in np.flatnonzero(~test_truth[i] & predicted)],
                "scores": {name: float(value) for name, value in zip(names, scores[i])}})
    label_frame = pd.DataFrame(per_labels)
    label_frame.to_csv(output / "per_label_test.csv", index=False)
    rare = label_frame[label_frame.label.isin([names[i] for i in rare_ids])]
    rare.to_csv(output / "rare_labels_test.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "learning_curves.csv", index=False)
    write_json(output / "report_data.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": complete, "representatives": representatives,
        "rare_labels": [names[i] for i in rare_ids], "report_examples": example_rows,
        "top_pairs": top_pairs.to_dict("records"), "error_counts": counts,
        "all_runs": merged.astype(object).where(merged.notna(), None).to_dict("records"),
        "mean_std": means.astype(object).where(means.notna(), None).to_dict("records"),
        "scope": "Fresh C3 training on current environment; historical A B C1 C2 metrics rechecked from scores",
        "source_hashes": {"historical_audit": sha256(run_root / "historical_verification/verification.json"),
                          "new_completion": sha256(run_root / "completion.json"),
                          "exporter": sha256(Path(__file__))}})
    print("PASS: 72 records, 36 summaries; rare labels, errors and examples recomputed from scores.")


if __name__ == "__main__":
    main()
