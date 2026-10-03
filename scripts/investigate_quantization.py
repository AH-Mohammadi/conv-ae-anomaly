"""Exploratory: alternative quantization strategies + input-range diagnostics.

Not part of the deployment pipeline. Reproduces the investigation behind the 'wide'
INT8 variant: dynamic-range and float16 strategies, and which sensors produce
extreme z-scores in the test region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from anomaly_detector.compress import convert_dynamic_range, convert_float16, evaluate_variant
from anomaly_detector.config import load_config
from anomaly_detector.data import SENSOR_COLUMNS
from anomaly_detector.evaluate import load_prepared, test_windows
from anomaly_detector.inference import TFLiteAutoencoder
from anomaly_detector.train import pooled_reference_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conv_ae.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model = keras.saving.load_model(Path(cfg.output.model_dir) / "model.keras")
    prepared, excluded = load_prepared(cfg)
    _, val_w = pooled_reference_windows(prepared, cfg)

    out: dict = {"strategies": {}, "input_range": {}}
    print(f"{'strategy':<16}{'size KB':>9}{'f1':>8}{'far':>8}{'mar':>8}{'auc':>8}")
    for name, conv in (("dynamic_range", convert_dynamic_range), ("float16", convert_float16)):
        blob = conv(model)
        det = TFLiteAutoencoder(model_content=blob)
        card, _ = evaluate_variant(name, det.reconstruction_errors, prepared, val_w, cfg, excluded)
        out["strategies"][name] = {k: card[k] for k in ("f1", "far", "mar", "roc_auc", "threshold")}
        out["strategies"][name]["size_bytes"] = len(blob)
        print(f"{name:<16}{len(blob) / 1024:>9.1f}{card['f1']:>8.3f}{card['far']:>8.3f}"
              f"{card['mar']:>8.3f}{card['roc_auc']:>8.3f}")

    tw = np.concatenate([test_windows(pf, cfg)[0] for pf in prepared])
    ref_std = np.median(np.array([pf.normalizer["scale"] for pf in prepared]), axis=0)
    print(f"\n{'sensor':<22}{'median ref std':>16}{'share |z|>10':>14}{'max |z|':>10}")
    for j, s in enumerate(SENSOR_COLUMNS):
        a = np.abs(tw[:, :, j])
        out["input_range"][s] = {"median_reference_std": float(ref_std[j]),
                                 "share_abs_z_gt_10": float((a > 10).mean()), "max_abs_z": float(a.max())}
        print(f"{s:<22}{ref_std[j]:>16.4f}{(a > 10).mean():>14.4f}{a.max():>10.0f}")

    path = Path(cfg.output.metrics_dir) / "quantization_alternatives.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
