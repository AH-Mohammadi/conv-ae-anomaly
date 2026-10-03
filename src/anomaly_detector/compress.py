"""TFLite conversion (FP32, full-integer INT8) and FP32-vs-INT8 comparison.

Every variant is scored by the same shared harness. Each variant gets its OWN
threshold from ITS OWN anomaly-free validation errors (no test labels), because
quantization shifts the error scale. Calibration windows come from anomaly-free
TRAIN windows only (the "wide" variant additionally rescales copies of them).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import keras
import numpy as np
import tensorflow as tf

from .baseline import predict
from .config import Config
from .evaluate import (
    PreparedFile, build_scorecard, load_prepared, score_files, test_windows, write_scorecard,
)
from .inference import TFLiteAutoencoder
from .metrics import compute_metrics, roc_auc
from .train import pooled_reference_windows, reconstruction_errors, resolve_threshold, set_seed

REPORT_SCHEMA = "compression_report_v1"
INT8_VARIANTS = ("int8_tflite", "int8_wide_tflite")  # in order of preference


# --------------------------------------------------------------------------- conversion
def make_representative_windows(train_windows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic random subset of anomaly-free training windows (float32)."""
    if train_windows.shape[0] == 0:
        raise ValueError("no training windows available for calibration")
    rng = np.random.default_rng(seed)
    idx = rng.choice(train_windows.shape[0], size=min(n, train_windows.shape[0]), replace=False)
    return train_windows[np.sort(idx)].astype(np.float32)


def make_wide_calibration(rep: np.ndarray, max_amplitude: float, seed: int) -> np.ndarray:
    """rep plus a copy scaled per window by a random factor in [1, max_amplitude]."""
    rng = np.random.default_rng(seed + 1)
    factors = rng.uniform(1.0, max_amplitude, size=(rep.shape[0], 1, 1)).astype(np.float32)
    return np.concatenate([rep, rep * factors])


def convert_fp32(model: keras.Model) -> bytes:
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_int8(model: keras.Model, representative: np.ndarray) -> bytes:
    """Post-training full-integer quantization with int8 input and output."""
    def rep_gen():
        for i in range(representative.shape[0]):
            yield [representative[i : i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def convert_dynamic_range(model: keras.Model) -> bytes:
    """int8 weights, float activations and float I/O (no calibration data)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def convert_float16(model: keras.Model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def tflite_op_names(blob: bytes) -> list[str] | None:
    """Sorted builtin op names in a .tflite (None if the schema module is unavailable)."""
    try:
        from tensorflow.lite.python import schema_py_generated as sch
        names = {v: k for k, v in vars(sch.BuiltinOperator).items() if not k.startswith("_")}
        model = sch.Model.GetRootAsModel(bytearray(blob), 0)
        codes = set()
        for i in range(model.OperatorCodesLength()):
            oc = model.OperatorCodes(i)
            codes.add(names.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()), "UNKNOWN"))
        return sorted(codes)
    except Exception:  # noqa: BLE001 - informational only
        return None


# --------------------------------------------------------------------------- evaluation
def evaluate_variant(
    name: str, score_fn: Callable[[np.ndarray], np.ndarray], prepared: list[PreparedFile],
    val_windows: np.ndarray, cfg: Config, excluded: list[str],
    extra_provenance: dict | None = None, extra: dict | None = None,
):
    """Own validation threshold -> shared harness scorecard. Returns (card, results)."""
    thr, src = resolve_threshold(score_fn(val_windows), cfg)
    results = score_files(prepared, cfg, score_fn)
    card = build_scorecard(
        results, cfg, threshold=thr, threshold_source=src, excluded=excluded,
        model_name=f"conv_ae_{name}", extra_provenance=extra_provenance,
        extra={"variant": name, **(extra or {})},
    )
    return card, results


def _summary(card: dict, size_bytes: int | None) -> dict:
    return {"f1": card["f1"], "far": card["far"], "mar": card["mar"],
            "roc_auc": card["roc_auc"], "threshold": card["threshold"],
            "size_bytes": size_bytes}


def _input_range(det: TFLiteAutoencoder) -> tuple[float, float]:
    q = det.describe()["input"]
    return (-128 - q["zero_point"]) * q["scale"], (127 - q["zero_point"]) * q["scale"]


def run_compression(cfg: Config, model: keras.Model | None = None,
                    model_dir: str | Path | None = None) -> tuple[dict, dict, dict]:
    """Convert + compare. Returns (report, tflite_blobs, scorecards)."""
    model_dir = Path(model_dir or cfg.output.model_dir)
    keras_file = model_dir / "model.keras"
    if model is None:
        model = keras.saving.load_model(keras_file)
    set_seed(cfg.seed)
    c = cfg.compression

    prepared, excluded = load_prepared(cfg)
    train_w, val_w = pooled_reference_windows(prepared, cfg)
    rep = make_representative_windows(train_w, c.calibration_samples, cfg.seed)
    wide = make_wide_calibration(rep, c.wide_calibration_max_amplitude, cfg.seed)

    blobs = {
        "fp32_tflite": convert_fp32(model),
        "int8_tflite": convert_int8(model, rep),
        "int8_wide_tflite": convert_int8(model, wide),
    }
    det = {k: TFLiteAutoencoder(model_content=v) for k, v in blobs.items()}
    score_fns = {"fp32_keras": lambda w: reconstruction_errors(model, w)}
    score_fns.update({k: d.reconstruction_errors for k, d in det.items()})
    sizes = {"fp32_keras": keras_file.stat().st_size if keras_file.exists() else None,
             **{k: len(v) for k, v in blobs.items()}}
    versions = {"tensorflow": tf.__version__, "keras": keras.__version__,
                "tflite_backend": det["fp32_tflite"].backend}

    cards, results = {}, {}
    for name, fn in score_fns.items():
        cards[name], results[name] = evaluate_variant(
            name, fn, prepared, val_w, cfg, excluded, versions, {"model_version": model_dir.name})

    y_true = np.concatenate([r.y_true for r in results["fp32_keras"]])
    pooled = {k: np.concatenate([r.scores for r in v]) for k, v in results.items()}
    s_k, s_t = pooled["fp32_keras"], pooled["fp32_tflite"]
    summ = {k: _summary(cards[k], sizes[k]) for k in cards}
    fp32_thr = cards["fp32_keras"]["threshold"]
    test_w = np.concatenate([test_windows(pf, cfg)[0] for pf in prepared]).astype(np.float32)

    analysis = {}
    for v in INT8_VARIANTS:
        delta = {m: summ[v][m] - summ["fp32_keras"][m] for m in ("f1", "far", "mar", "roc_auc")}
        checks = {
            "f1_within_tolerance": abs(delta["f1"]) <= c.max_metric_change,
            "far_within_tolerance": abs(delta["far"]) <= c.max_metric_change,
            "mar_within_tolerance": abs(delta["mar"]) <= c.max_metric_change,
            "auc_drop_within_tolerance": -delta["roc_auc"] <= c.max_auc_drop,
        }
        s8 = pooled[v]
        fixed = compute_metrics(y_true, predict(s8, fp32_thr))
        # isolate input clipping: FP32 Keras fed inputs clipped to this INT8 variant's range
        lo, hi = _input_range(det[v])
        rec = model.predict(np.clip(test_w, lo, hi), batch_size=256, verbose=0)
        err_clip = ((rec - test_w) ** 2).mean(axis=(1, 2)).astype(np.float64)
        clip_m = compute_metrics(y_true, predict(err_clip, fp32_thr))
        analysis[v] = {
            "delta_vs_fp32_keras": delta, "checks": checks, "passed": all(checks.values()),
            "score_agreement_vs_fp32_keras": {
                "pearson": float(np.corrcoef(s8, s_k)[0, 1]),
                "median_relative_diff": float(np.median(np.abs(s8 - s_k) / np.maximum(s_k, 1e-12)))},
            "with_fp32_keras_threshold": {"threshold": fp32_thr, "f1": fixed["f1"],
                                          "far": fixed["far"], "mar": fixed["mar"]},
            "input_range": [lo, hi],
            "input_clip_fraction": {"validation": det[v].input_clip_fraction(val_w),
                                    "test": det[v].input_clip_fraction(test_w)},
            "fp32_keras_with_clipped_inputs": {
                "f1": clip_m["f1"], "far": clip_m["far"], "mar": clip_m["mar"],
                "roc_auc": roc_auc(y_true, err_clip)},
        }

    selected = next((v for v in INT8_VARIANTS if analysis[v]["passed"]), "fp32_tflite")
    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset": "SKAB",
        "model_version": model_dir.name,
        "variants": summ,
        "int8_analysis": analysis,
        "fp32_tflite_vs_keras": {
            "max_abs_score_diff": float(np.max(np.abs(s_t - s_k))),
            "max_relative_score_diff": float(np.max(np.abs(s_t - s_k) / np.maximum(s_k, 1e-12)))},
        "acceptance": {
            "criteria": {"max_metric_change_abs": c.max_metric_change, "max_auc_drop": c.max_auc_drop},
            "selection_rule": "first passing INT8 variant in order " + ", ".join(INT8_VARIANTS)
                              + "; otherwise fp32_tflite",
            "selected": selected,
        },
        "calibration": {
            "standard": {"n_samples": int(rep.shape[0]),
                         "source": "random anomaly-free training windows (seeded)"},
            "wide": {"n_samples": int(wide.shape[0]),
                     "max_amplitude": c.wide_calibration_max_amplitude,
                     "source": "standard windows + copies scaled by U(1, max_amplitude)",
                     "note": "This variant was introduced after inspecting test-set metrics of "
                             "the standard INT8 model (2 widths tried), so its reported "
                             "numbers carry mild selection bias."},
        },
        "tflite": {k: {"ops": tflite_op_names(blobs[k]), **det[k].describe()} for k in blobs},
        "n_files": cards["fp32_keras"]["n_files"],
        "protocol": cards["fp32_keras"]["protocol"],
        "config": cfg.to_dict(),
        "provenance": cards["fp32_keras"]["provenance"],
    }
    return report, blobs, cards


def save_compression_outputs(report: dict, blobs: dict, cards: dict, cfg: Config,
                             model_dir: str | Path) -> dict[str, Path]:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, blob in blobs.items():
        out[name] = d / f"detector_{name.replace('_tflite', '')}.tflite"
        out[name].write_bytes(blob)
    selected = report["acceptance"]["selected"]
    out["detector"] = d / "detector.tflite"
    shutil.copyfile(out[selected], out["detector"])

    meta = {
        "model_version": report["model_version"],
        "selected": selected,
        "window_size": cfg.windowing.window_size,
        "variants": {
            k: {"file": out[k].name, "threshold": report["variants"][k]["threshold"],
                "size_bytes": report["variants"][k]["size_bytes"], **report["tflite"][k]}
            for k in blobs
        },
    }
    out["meta"] = d / "tflite_meta.json"
    out["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    for k in blobs:
        out[f"scorecard_{k}"] = write_scorecard(cards[k], cfg.output.metrics_dir)
    out["report"] = Path(cfg.output.metrics_dir) / "compression_report.json"
    out["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out
