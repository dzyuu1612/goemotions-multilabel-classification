"""Vẽ kết quả C3 từ CSV đã tính; không viết báo cáo, không train hoặc tune lại."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.distilbert_study import read_json
from src.paths import ROOT


def plot_results(folder):
    folder = Path(folder)
    protocol = read_json(folder / "protocol.json")
    history = pd.read_csv(folder / "history_all_seeds.csv")
    rare = pd.read_csv(folder / "rare_labels_mean_std.csv")
    if set(history.seed) != set(protocol["seeds"]):
        raise ValueError("Learning curve chưa có đủ seed của protocol")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for seed, rows in history.groupby("seed"):
        rows = rows.sort_values("epoch")
        axes[0].plot(rows.epoch, rows.train_loss, marker="o", label=f"Seed {seed}")
        axes[1].plot(rows.epoch, rows.validation_macro_f1, marker="o", label=f"Seed {seed}")
    for axis in axes:
        axis.set_xticks(sorted(history.epoch.unique()))
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel("Train BCE")
    axes[1].set_ylabel("Validation Macro-F1 @0.5")
    figure.savefig(folder / "learning_curves.png", dpi=180)
    plt.close(figure)

    labels = protocol["rare_labels_train_defined"]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for offset, mode, title in [(-.18, "fixed_0.5", "Fixed @0.5"),
                                (.18, "per_label_tuned_on_validation", "Validation tuned")]:
        rows = rare[rare["mode"] == mode].set_index("label").loc[labels]
        axis.bar(x + offset, rows.f1_mean, width=.36, yerr=rows.f1_std, capsize=3, label=title)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1 mean ± sample std")
    axis.set_title("C3 rare labels: validation tuning, not independent test")
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    figure.savefig(folder / "rare_labels_f1.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "reports/c3_distilbert/full")
    args = parser.parse_args()
    plot_results(args.folder)
    print("Saved learning_curves.png and rare_labels_f1.png")


if __name__ == "__main__":
    main()
