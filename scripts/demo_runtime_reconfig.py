"""Iteration 7 demonstration (self-contained, no networking):

  detector running -> configuration changes (a local command file, simulating a
  future Coordinator) -> detector detects the change -> new configuration applied
  -> stream continues, uninterrupted.

Streams one SKAB file. Partway through, this script itself plays the role of the
Coordinator: it writes a new threshold to a local JSON command file (the same
mechanism scripts/set_runtime_config.py offers for manual use). The SAME running
StreamingDetector (same ring buffer, same sample position) notices the file change
on its next tick and calls apply_new_config()/reconfigure(). Prints a before/after
comparison and writes reports/metrics/runtime_reconfig_demo.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_detector.calibration import Calibrator
from anomaly_detector.config import load_config
from anomaly_detector.data import check_reference_segment, discover_files, file_key, load_series
from anomaly_detector.inference import TFLiteAutoencoder
from anomaly_detector.records import build_output_record
from anomaly_detector.replay import replay_samples
from anomaly_detector.runtime_config import ConfigError, apply_new_config
from anomaly_detector.stream import StreamingDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/conv_ae.yaml")
    ap.add_argument("--variant", default="fp32_tflite")
    ap.add_argument("--file", default=None, help="SKAB file, e.g. valve1/0.csv (default: first available)")
    ap.add_argument("--new-threshold", type=float, default=None,
                    help="threshold to hot-swap in mid-stream (default: 1.5x the deployed threshold)")
    ap.add_argument("--change-at-record", type=int, default=None,
                    help="emit this many records before issuing the command (default: halfway)")
    ap.add_argument("--command-file", default="reports/metrics/runtime_config_command.json")
    ap.add_argument("--out", default="reports/metrics/runtime_reconfig_demo.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model_dir = Path(cfg.output.model_dir)
    meta = json.loads((model_dir / "tflite_meta.json").read_text(encoding="utf-8"))
    v = meta["variants"][a.variant]
    det_backend = TFLiteAutoencoder(model_path=model_dir / v["file"])
    calibrator = Calibrator.from_dict(v["calibration"])
    model_version = f"{meta['model_version']}:{a.variant}"

    excluded = set(cfg.dataset.exclude_files)
    files = [p for p in discover_files(cfg.dataset.root) if file_key(p) not in excluded]
    path = next((p for p in files if file_key(p) == a.file), None) if a.file else files[0]
    if path is None:
        raise SystemExit(f"file {a.file!r} not found (or excluded)")
    series = load_series(path)
    check_reference_segment(series, cfg.dataset.reference_rows)
    print(f"Streaming {file_key(path)} ({len(series.values)} rows). "
          f"Deployed: {model_version}, factory threshold={v['threshold']:.4f}\n")

    score_fn = lambda w: float(det_backend.reconstruction_errors(w[None])[0])
    det = StreamingDetector(
        score_fn, n_channels=series.values.shape[1], window_size=cfg.windowing.window_size,
        reference_rows=cfg.dataset.reference_rows, threshold=v["threshold"], stride=cfg.windowing.stride,
        normalization_method=cfg.preprocessing.normalization_method,
        calibrator=calibrator, model_version=model_version,
    )

    expected_records = max(0, len(series.values) - cfg.dataset.reference_rows)
    change_at = a.change_at_record if a.change_at_record is not None else expected_records // 2
    new_threshold = a.new_threshold if a.new_threshold is not None else v["threshold"] * 1.5

    records: list[dict] = []
    buf_id_before = id(det._ring._buf)
    state = {"command_written": False, "applied": False, "applied_at_record": None,
            "applied_at_sample": None}

    def write_command() -> None:
        hparams = {"threshold": new_threshold}
        out_path = Path(a.command_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(hparams), encoding="utf-8")
        tmp.replace(out_path)
        print(f">>> Coordinator-style command written to {out_path}: {hparams}")

    def cb(rec) -> None:
        out = build_output_record(rec, model_version=model_version, file=file_key(path))
        records.append(out.to_dict())
        if not state["command_written"] and len(records) >= change_at:
            state["command_written"] = True
            write_command()

    def on_tick(_i, det_) -> None:
        if not state["command_written"] or state["applied"]:
            return
        try:
            hparams = json.loads(Path(a.command_file).read_text(encoding="utf-8"))
            new_cfg = apply_new_config(det_.current_runtime_config(), hparams)
            det_.reconfigure(new_cfg)
        except ConfigError as exc:
            print(f">>> REJECTED: {exc}")
            return
        state["applied"] = True
        state["applied_at_record"] = len(records)
        state["applied_at_sample"] = det_.sample_index
        print(f">>> Detector detected and applied the change at record #{len(records)} "
              f"(sample {det_.sample_index}): threshold {v['threshold']:.4f} -> "
              f"{det_.threshold:.4f}, config_version -> {det_.config_version}\n")

    _, stats = replay_samples(series.values, det, speed=0.0, record_cb=cb,
                              on_tick=on_tick, tick_every=1, rss_every=0)

    buf_id_after = id(det._ring._buf)
    before = [r for r in records if r["window_id"] < (state["applied_at_sample"] or 0)]
    after = [r for r in records if r["window_id"] >= (state["applied_at_sample"] or 0)]
    window_ids = [r["window_id"] for r in records]
    contiguous = window_ids == list(range(window_ids[0], window_ids[0] + len(window_ids))) if window_ids else True

    summary = {
        "file": file_key(path), "expected_records": expected_records, "actual_records": len(records),
        "missed_windows": stats.summary()["missed_windows"],
        "ring_buffer_identity_unchanged": buf_id_before == buf_id_after,
        "window_id_contiguous_across_change": contiguous,
        "config_applied": state["applied"],
        "applied_at_record": state["applied_at_record"], "applied_at_sample": state["applied_at_sample"],
        "before_change": {"n_records": len(before),
                          "threshold": before[0]["threshold"] if before else None,
                          "config_version": before[0]["config_version"] if before else None},
        "after_change": {"n_records": len(after),
                         "threshold": after[0]["threshold"] if after else None,
                         "config_version": after[0]["config_version"] if after else None},
    }
    Path(a.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Summary:")
    print(f"  records: expected={summary['expected_records']} actual={summary['actual_records']} "
          f"missed_windows={summary['missed_windows']}")
    print(f"  ring buffer object unchanged across the reconfiguration: "
          f"{summary['ring_buffer_identity_unchanged']}")
    print(f"  window_id contiguous across the change (no gap/reset): "
          f"{summary['window_id_contiguous_across_change']}")
    print(f"  before: n={summary['before_change']['n_records']} "
          f"threshold={summary['before_change']['threshold']} "
          f"config_version={summary['before_change']['config_version']}")
    print(f"  after:  n={summary['after_change']['n_records']} "
          f"threshold={summary['after_change']['threshold']} "
          f"config_version={summary['after_change']['config_version']}")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
