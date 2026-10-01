"""Train the Conv-AE on anomaly-free reference windows and evaluate it offline.

Protocol (no test labels are used for any decision):
  * per file: normalizer fitted on the reference rows; reference split in time into
    train rows [0, train_rows) and validation rows [train_rows, reference_rows)
  * ONE global model trained on the pooled train windows of all files
  * validation windows -> early stopping and threshold (percentile of pooled errors)
  * test region -> final metrics only
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import Config
from .evaluate import (
    PreparedFile, build_scorecard, load_prepared, reference_windows, score_files,
)
from .model import build_conv_ae


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def reconstruction_errors(model: keras.Model, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Per-window mean squared reconstruction error, shape (N,)."""
    if windows.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    x = windows.astype(np.float32)
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return ((recon - x) ** 2).mean(axis=(1, 2)).astype(np.float64)


def select_threshold(val_errors: np.ndarray, percentile: float) -> float:
    """Threshold = percentile of anomaly-free validation errors (no test labels)."""
    if val_errors.size == 0:
        raise ValueError("no validation windows to select a threshold from")
    return float(np.percentile(val_errors, percentile))


def fit_conv_ae(train_w: np.ndarray, val_w: np.ndarray, cfg: Config) -> tuple[keras.Model, dict]:
    m, t, w = cfg.model, cfg.training, cfg.windowing
    model = build_conv_ae(w.window_size, train_w.shape[2], m.latent_dim, m.filter_count, m.kernel_size)
    model.compile(optimizer=keras.optimizers.Adam(t.learning_rate), loss="mse")
    x_tr, x_val = train_w.astype(np.float32), val_w.astype(np.float32)
    hist = model.fit(
        x_tr, x_tr, validation_data=(x_val, x_val),
        epochs=t.epochs, batch_size=t.batch_size, shuffle=True, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=t.patience, restore_best_weights=True)],
    )
    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    return model, history


def pooled_reference_windows(prepared: list[PreparedFile], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    pairs = [reference_windows(pf, cfg) for pf in prepared]
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def run_conv_ae(cfg: Config, model_dir: str | Path | None = None) -> tuple[dict, keras.Model, dict]:
    """Full offline pipeline. Returns (scorecard, model, artifacts_dict)."""
    set_seed(cfg.seed)
    prepared, excluded = load_prepared(cfg)
    train_w, val_w = pooled_reference_windows(prepared, cfg)
    model, history = fit_conv_ae(train_w, val_w, cfg)

    val_err = reconstruction_errors(model, val_w)
    if cfg.detector.threshold is None:
        p = cfg.training.threshold_percentile
        threshold = select_threshold(val_err, p)
        source = f"{p}th percentile of anomaly-free validation reconstruction errors"
    else:
        threshold = float(cfg.detector.threshold)
        source = "fixed value from config"

    results = score_files(prepared, cfg, lambda w: reconstruction_errors(model, w))

    # Diagnostic ONLY (never used to choose the threshold): how sensitive are the
    # metrics to the validation percentile.
    from .baseline import predict
    from .metrics import compute_metrics
    y_true = np.concatenate([r.y_true for r in results])
    scores = np.concatenate([r.scores for r in results])
    sens = {}
    for p in (95.0, 99.0, 99.9):
        m = compute_metrics(y_true, predict(scores, select_threshold(val_err, p)))
        sens[str(p)] = {"threshold": select_threshold(val_err, p),
                        "f1": m["f1"], "far": m["far"], "mar": m["mar"]}

    card = build_scorecard(
        results, cfg, threshold=threshold, threshold_source=source, excluded=excluded,
        extra_provenance={"tensorflow": tf.__version__, "keras": keras.__version__},
        extra={
            "model_version": Path(model_dir or cfg.output.model_dir).name,
            "training": {
                "n_train_windows": int(train_w.shape[0]),
                "n_val_windows": int(val_w.shape[0]),
                "epochs_run": len(history["loss"]),
                "best_val_loss": float(min(history["val_loss"])),
                "n_parameters": int(model.count_params()),
            },
            "diagnostics": {
                "note": "threshold sensitivity, informational only; not used for selection",
                "threshold_sensitivity": sens,
            },
        },
    )
    artifacts = {
        "history": history,
        "test_scores": {r.name: r.scores for r in results},
        "test_labels": {r.name: r.y_true for r in results},
        "normalizers": {pf.name: pf.normalizer for pf in prepared},
        "val_errors": val_err,
    }
    return card, model, artifacts


def save_artifacts(card: dict, model: keras.Model, artifacts: dict, cfg: Config,
                   model_dir: str | Path) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model.keras")
    meta = {
        "model_version": d.name,
        "threshold": card["threshold"],
        "threshold_source": card["protocol"]["threshold_source"],
        "window_size": cfg.windowing.window_size,
        "n_channels": int(model.input_shape[-1]),
        "config": cfg.to_dict(),
        "provenance": card["provenance"],
        "normalizers": artifacts["normalizers"],
    }
    (d / "artifact.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (d / "training_history.json").write_text(json.dumps(artifacts["history"], indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        d / "scores.npz",
        val_errors=artifacts["val_errors"],
        **{f"score::{k}": v for k, v in artifacts["test_scores"].items()},
        **{f"label::{k}": v for k, v in artifacts["test_labels"].items()},
    )
    return d
