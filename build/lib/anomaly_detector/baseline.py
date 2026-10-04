"""Trivial per-sensor z-score baseline. Not meant to be a good detector."""
from __future__ import annotations

import numpy as np


def window_scores(windows_z: np.ndarray) -> np.ndarray:
    """Score = max over sensors of (mean over time of |z|).

    Args:
        windows_z: z-scored windows, shape (N, W, C).
    Returns:
        (N,) float array; higher means more anomalous.
    """
    if windows_z.ndim != 3:
        raise ValueError(f"expected (N, W, C), got shape {windows_z.shape}")
    if windows_z.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return np.abs(windows_z).mean(axis=1).max(axis=1)


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions: 1 where score > threshold."""
    return (np.asarray(scores) > threshold).astype(np.int8)
