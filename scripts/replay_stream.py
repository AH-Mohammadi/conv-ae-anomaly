"""Replay SKAB files through the streaming detector; write per-window scores and a
stability summary. One StreamingDetector per file (fresh reference/normalizer per
episode), matching the offline per-file normalization protocol.

Examples:
  python scripts/replay_stream.py --speed 0                      # as fast as possible (default)
  python scripts/replay_stream.py --speed 1 --files valve1/0.csv  # real time, one file
  python scripts/replay_stream.py --loops 5 --speed 0             # soak test, 5x all files
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_detector.config import load_config
from anomaly_detector.data import check_reference_segment, discover_files, file_key, load_series
from anomaly_detector.inference import TFLiteAutoencoder
from anomaly_detector.replay import REPLAY_SCHEMA, replay_samples
from anomaly_detector.stream import StreamingDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/conv_ae.yaml")
    ap.add_argument("--variant", default="fp32_tflite",
                    help="detector_<variant>.tflite in model_dir (default: fp32_tflite, "
                         "the Iteration 4 recommendation)")
    ap.add_argument("--files", nargs="*", default=None, help="e.g. valve1/0.csv (default: all)")
    ap.add_argument("--loops", type=int, default=1, help="repeat the file list N times (soak test)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="0 = as fast as possible; 1 = real time; 10 = 10x real time")
    ap.add_argument("--out", default="reports/metrics/stream_scores.jsonl")
    ap.add_argument("--summary-out", default="reports/metrics/replay_summary.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model_dir = Path(cfg.output.model_dir)
    meta = json.loads((model_dir / "tflite_meta.json").read_text(encoding="utf-8"))
    v = meta["variants"][a.variant]
    det_backend = TFLiteAutoencoder(model_path=model_dir / v["file"])

    all_files = discover_files(cfg.dataset.root)
    excluded = set(cfg.dataset.exclude_files)
    chosen = [p for p in all_files if file_key(p) not in excluded]
    if a.files:
        want = set(a.files)
        chosen = [p for p in chosen if file_key(p) in want]
        if not chosen:
            raise SystemExit(f"no files matched {a.files}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(a.out, "w", encoding="utf-8")
    episodes = []
    try:
        for loop in range(a.loops):
            for path in chosen:
                series = load_series(path)
                check_reference_segment(series, cfg.dataset.reference_rows)
                gaps = np.diff(series.timestamps).astype("timedelta64[ms]").astype(np.int64) / 1000.0
                gaps = np.concatenate([[0.0], gaps])

                score_fn = lambda w, _d=det_backend: float(_d.reconstruction_errors(w[None])[0])
                det = StreamingDetector(
                    score_fn,
                    n_channels=series.values.shape[1], window_size=cfg.windowing.window_size,
                    reference_rows=cfg.dataset.reference_rows, threshold=v["threshold"],
                    stride=cfg.windowing.stride,
                    normalization_method=cfg.preprocessing.normalization_method,
                )

                def cb(rec, name=file_key(path), labels=series.labels):
                    row = {"file": name, "sample_index": rec.sample_index, "score": rec.score,
                           "threshold": rec.threshold, "is_anomaly": rec.is_anomaly,
                           "label": int(labels[rec.sample_index]), "processing_ns": rec.processing_ns}
                    out_fh.write(json.dumps(row) + "\n")

                _, stats = replay_samples(series.values, det, gaps_s=gaps, speed=a.speed, record_cb=cb)
                s = stats.summary()
                s["file"] = file_key(path)
                s["loop"] = loop
                episodes.append(s)
                print(f"loop {loop} {file_key(path):<16} records={s['n_records']:>4} "
                      f"missed_windows={s['missed_windows']:>3} "
                      f"deadline_overruns={s['n_deadline_overruns']:>3} "
                      f"proc_p99_ms={s['processing_ms'].get('p99', float('nan')):.3f}")
    finally:
        out_fh.close()

    summary = {
        "schema_version": REPLAY_SCHEMA, "variant": a.variant, "speed": a.speed, "loops": a.loops,
        "n_episodes": len(episodes),
        "total_records": sum(e["n_records"] for e in episodes),
        "total_missed_windows": sum(e["missed_windows"] for e in episodes),
        "total_deadline_overruns": sum(e["n_deadline_overruns"] for e in episodes),
        "episodes": episodes,
    }
    Path(a.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nscores -> {a.out}\nsummary -> {a.summary_out}")
    if summary["total_missed_windows"] or summary["total_deadline_overruns"]:
        print("WARNING: missed windows or deadline overruns occurred; see summary for details.")


if __name__ == "__main__":
    main()
