"""Map raw reconstruction error to a calibrated score in [0, 1].

Method: log-ratio sigmoid, anchored at the operating threshold.
    r(e)     = log((e + eps) / (threshold + eps))      -- log of how far above/
                                                            below the decision
                                                            boundary e is
    score(e) = sigmoid(r(e) / scale)

``scale`` is the standard deviation of r() computed over anomaly-free VALIDATION
errors only (never test data) -- "how much the error naturally wobbles during
normal operation, in log-ratio units". This makes the [0, 1] range meaningful
rather than arbitrary:
  * score(threshold) == 0.5 EXACTLY, so `calibrated_score > 0.5` reproduces the
    raw decision `raw_error > threshold` with no change in F1/FAR/MAR.
  * errors within the normal operating noise band around the threshold map to a
    graded region around 0.5; errors many log-ratio-sigmas beyond it saturate
    towards 0 or 1 (log space keeps this graded even though raw errors span
    several orders of magnitude -- see README).
Log space was chosen over a raw-error scale because test-region errors reach
roughly 1000x the threshold for some files; a linear scale saturates almost
every such point to 1.0 and loses the distinction documented in Iteration 3's
out-of-distribution-input finding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPS = 1e-9
MIN_SCALE = 1e-6  # floor so a (near-)constant validation error can't blow up the sigmoid


@dataclass(frozen=True)
class CalibrationParams:
    method: str
    center: float   # raw-error threshold; score(center) == 0.5
    scale: float     # > 0; std of validation log-ratios

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationParams":
        return cls(method=d["method"], center=float(d["center"]), scale=float(d["scale"]))


class Calibrator:
    def __init__(self, params: CalibrationParams) -> None:
        self.params = params

    @classmethod
    def fit(cls, val_errors: np.ndarray, threshold: float) -> "Calibrator":
        """Fit ``scale`` from anomaly-free validation reconstruction errors only."""
        val_errors = np.asarray(val_errors, dtype=np.float64)
        if val_errors.size < 2:
            raise ValueError("need at least 2 validation errors to fit a calibration scale")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        r = np.log((val_errors + EPS) / (threshold + EPS))
        scale = max(float(np.std(r)), MIN_SCALE)
        return cls(CalibrationParams("log_ratio_sigmoid", float(threshold), scale))

    def transform(self, raw_errors) -> np.ndarray:
        """raw reconstruction error(s) -> calibrated score(s) in [0, 1]."""
        e = np.asarray(raw_errors, dtype=np.float64)
        r = np.log((e + EPS) / (self.params.center + EPS))
        z = np.clip(r / self.params.scale, -50.0, 50.0)  # avoid overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    def transform_one(self, raw_error: float) -> float:
        return float(self.transform(np.asarray([raw_error]))[0])

    def to_dict(self) -> dict:
        return self.params.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(CalibrationParams.from_dict(d))
