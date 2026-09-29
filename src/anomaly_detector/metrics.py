"""F1 / FAR / MAR (point-wise). Undefined ratios are returned as None."""
from __future__ import annotations

import numpy as np


def _as_binary(a, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1")
    return arr.astype(np.int8)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else float(num) / float(den)


def compute_metrics(y_true, y_pred) -> dict:
    """Point-wise metrics.

    f1  = 2TP / (2TP + FP + FN)
    far = FP / (FP + TN)      false alarm rate (fraction of normal points flagged)
    mar = FN / (FN + TP)      missing alarm rate (fraction of anomalies missed)

    All in [0, 1]. A metric with a zero denominator is None.
    """
    yt = _as_binary(y_true, "y_true")
    yp = _as_binary(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("cannot compute metrics on zero points")

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "far": _ratio(fp, fp + tn),
        "mar": _ratio(fn, fn + tp),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_points": int(yt.size),
    }
