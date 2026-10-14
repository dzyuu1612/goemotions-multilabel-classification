"""Xuất bảng nhỏ; chỉ tính mean/std chính thức khi có >=3 seed full cùng cấu hình."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.distilbert_study import ROOT, read_json, write_json
from src.datasets.goemotions import sha256


def summarize(run_paths):
    rows, signature, seeds = [], None, set()
    for path in map(Path, run_paths):
        meta = read_json(path / "run.json")
        if meta.get("status") != "complete" or meta["mode"] != "full":
            raise ValueError("Bảng mean/std chính thức không nhận pilot hoặc run chưa hoàn thành")
        if meta["n_train"] != 43410 or meta["n_validation"] != 5426 or meta["test_used"]:
            raise ValueError("Run không phải full train/validation đúng protocol")
        for filename, expected in meta["artifact_sha256"].items():
            if sha256(path/filename) != expected:
                raise ValueError(f"Sai artifact hash: {path/filename}")
        seed = meta["config"]["seed"]
        if seed in seeds:
            raise ValueError("Seed bị lặp")
        seeds.add(seed)
        config = {k:v for k,v in meta["config"].items() if k != "seed"}
        current = (config, meta["data_sha256"], meta["label_names"], meta["source_sha256"])
        if signature is not None and current != signature:
            raise ValueError("Các run khác cấu hình, dữ liệu, mapping hoặc code")
        signature = current
        metrics = read_json(path/"validation_metrics.json")
        rows.append({"seed":seed, "best_epoch":meta["best_epoch"],
                     **{k:metrics[k] for k in ["macro_f1","micro_f1","macro_precision","macro_recall",
                                             "micro_precision","micro_recall","hamming_loss"]}})
    if len(seeds) < 3:
        raise ValueError("Cần ít nhất ba seed full khác nhau để báo cáo mean ± sample std")
    df = pd.DataFrame(rows).sort_values("seed")
    stats = [{"metric":k,"mean":float(df[k].mean()),"sample_std":float(df[k].std(ddof=1)),"n":len(df)}
             for k in df.columns if k not in ("seed","best_epoch")]
    return df, pd.DataFrame(stats)


def export_pilot(path, output):
    meta, metrics = read_json(path/"run.json"), read_json(path/"validation_metrics.json")
    if meta["mode"] != "pilot" or meta["status"] != "complete":
        raise ValueError("Chỉ xuất pilot hoàn thành")
    for filename, expected in meta["artifact_sha256"].items():
        if sha256(path/filename) != expected:
            raise ValueError(f"Sai artifact hash: {filename}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"pilot_run.json", meta)
    write_json(output/"pilot_metrics.json", metrics)
    history = read_json(path/"history.json")
    pd.DataFrame(history).to_csv(output/"pilot_history.csv", index=False)
    pd.DataFrame(metrics["per_label"]).to_csv(output/"pilot_per_label.csv", index=False, encoding="utf-8-sig")
    write_json(output/"pilot_inference_example.json", read_json(path/"inference_example.json"))



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--runs", nargs="*", type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/c3_distilbert")
    args=parser.parse_args()
    if bool(args.pilot) == bool(args.runs):
        parser.error("Chọn --pilot hoặc --runs")
    if args.pilot:
        export_pilot(args.pilot,args.output)
    else:
        rows,stats=summarize(args.runs)
        args.output.mkdir(parents=True,exist_ok=True)
        rows.to_csv(args.output/"full_validation_runs.csv",index=False)
        stats.to_csv(args.output/"full_validation_mean_std.csv",index=False)


if __name__ == "__main__":
    main()
