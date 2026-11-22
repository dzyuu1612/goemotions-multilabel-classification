"""Tính lại các bảng bổ sung của báo cáo IEEE v8 từ điểm và thời gian gốc."""
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.threshold_error_20261110 import analysis
from src.datasets.goemotions import load_goemotions
from src.paths import ROOT

HERE = Path(__file__).resolve().parent
CSV = HERE / "results"


def approval_curve(validation, names):
    record = next(r for r in analysis.records() if r["architecture"] == "roberta" and r["seed"] == 2026)
    j = names.index("approval")
    values = analysis.load_scores(ROOT / record["val_scores"], validation, names)[:, j]
    truth = analysis.targets(validation)[:, j]
    rows = []
    for t in analysis.GRID:
        pred = values >= t
        tp, fp, fn = (int(v) for v in analysis.counts(truth, pred))
        rows.append(dict(threshold=t, tp=tp, fp=fp, fn=fn,
                         precision=tp/(tp+fp) if tp+fp else 0.,
                         recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn)))
    table = pd.DataFrame(rows)
    winner = max(rows, key=lambda r: analysis.tie(r["f1"], r["threshold"]))
    thresholds, _ = analysis.frozen_thresholds(analysis.read(ROOT / record["protocol"]))
    assert winner["threshold"] == thresholds[j] and truth.sum() == 397
    return table


def test_counts(test, names):
    truth = analysis.targets(test)
    rows = []
    for record in analysis.records():
        if record["seed"] != 2026:
            continue
        values = analysis.load_scores(ROOT / record["test_scores"], test, names)
        tuned, _ = analysis.frozen_thresholds(analysis.read(ROOT / record["protocol"]))
        for mode, threshold in (("fixed", 0.5), ("tuned", tuned)):
            m = analysis.metrics(truth, values >= threshold)
            rows.append({"model": record["model"], "seed": record["seed"], "mode": mode,
                         **{k: m[k] for k in ("n_tp", "n_fp", "n_fn", "micro_f1", "hamming")}})
    return pd.DataFrame(rows)


def annotation_sample():
    registry = pd.read_csv(analysis.HERE/"results/annotation_sampling_registry.csv", keep_default_na=False)
    neutral = registry[registry.pair.eq("neutral->approval")].sample(48, random_state=20261122)
    relief = registry[registry.pair.eq("relief->admiration")]
    return pd.concat([neutral, relief]).sort_values("annotation_id").reset_index(drop=True)


def inference_summary():
    receipt = analysis.read(CSV / "inference_setup.json")
    timings = pd.read_csv(CSV / "inference_timings.csv")
    sample_ids = receipt["sample_ids"]
    assert len(sample_ids) == len(set(sample_ids)) == 512
    assert len(timings) == 2720 and (timings.seconds > 0).all()
    rows = []
    for model, recorded in sorted(receipt["systems"].items()):
        for batch in (1, 16):
            part = timings.loc[timings.model.eq(model) & timings.batch_size.eq(batch)]
            secs = part.seconds.to_numpy()
            assert part.batch_index.tolist() == list(range(1, 512//batch+1))
            assert part.sentences.eq(batch).all()
            rows.append(dict(
                model=model, device=recorded["device"], dtype=recorded["dtype"],
                batch=batch, n=int(part.sentences.sum()), calls=len(part),
                mean_ms=1000*np.mean(secs), p50_ms=1000*np.median(secs), p95_ms=1000*np.percentile(secs,95),
                sentences_per_sec=512/secs.sum(), **recorded["memory"][str(batch)],
                total_seconds=secs.sum(), package_bytes=sum(recorded["file_bytes"].values()),
            ))
    return pd.DataFrame(rows)


def verify():
    frames, names, manifest = load_goemotions(ROOT, write_metadata=False, splits=("validation","test"))
    approval = approval_curve(frames["validation"], names)
    counts = test_counts(frames["test"], names)
    times = inference_summary()
    sample = annotation_sample()
    setup = analysis.read(CSV / "inference_setup.json")
    validation = frames["validation"]
    indices = np.sort(np.random.default_rng(setup["sample_seed"]).choice(len(validation), 512, replace=False))
    assert validation.iloc[indices].id.tolist() == setup["sample_ids"]
    assert manifest["files"][0]["sha256"] == setup["validation_sha256"]
    for table, name in ((approval,"approval_thresholds"),(counts,"test_fp_fn"),
                        (times,"inference_summary"),(sample,"annotation_sample")):
        stored = pd.read_csv(CSV / f"{name}.csv", keep_default_na=name!="annotation_sample")
        pd.testing.assert_frame_equal(table,stored,check_dtype=False,rtol=1e-9,atol=1e-9)
    truth = analysis.targets(frames["test"])
    total, positives = truth.size, int(truth.sum())
    assert (total,positives) == (151956,6329)
    return approval, counts, times, sample, (total,positives)


if __name__ == "__main__":
    verify()
    print("IEEE v8: khớp lưới approval, số TP/FP/FN, số đo suy luận và mẫu đọc 50 mục.")
