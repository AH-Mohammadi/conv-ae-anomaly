"""Per-sensor normalization fitted on reference data only."""
from __future__ import annotations

import numpy as np

EPS = 1e-8


class Normalizer:
    """Per-sensor normalizer. ``fit`` must only ever see anomaly-free reference data."""

    def __init__(self, method: str = "zscore") -> None:
        if method not in ("zscore", "minmax"):
            raise ValueError(f"Unknown normalization method {method!r}")
        self.method = method
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> "Normalizer":
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[0] < 2:
            raise ValueError("reference must be 2-D (T, C) with T >= 2")
        if self.method == "zscore":
            center, scale = ref.mean(axis=0), ref.std(axis=0)
        else:
            center = ref.min(axis=0)
            scale = ref.max(axis=0) - center
        # Constant sensors in the reference would divide by ~0; use scale 1.
        self.center_ = center
        self.scale_ = np.where(scale < EPS, 1.0, scale)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (np.asarray(x, dtype=np.float64) - self.center_) / self.scale_

    def to_dict(self) -> dict:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Normalizer is not fitted")
        return {
            "method": self.method,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
        }
