"""Explicit runtime configuration for the detector's hot-swappable hyperparameters,
with a local apply_new_config(hparams) interface.

Hyperparameter classification (see also config.py's module docstring):
  hot-swappable (this module):              threshold, calibration_scale ("alert
      sensitivity": how fast the calibrated score saturates away from the boundary)
  requires a new model artifact (config.py / compress.py): window_size, stride,
      latent_dim, filter_count, kernel_size, normalization_method, and of course
      the trained weights / TFLite variant itself.

apply_new_config() only ever touches the hot-swappable fields; anything else is
rejected with a clear error naming which artifact-dependent change would be needed
instead. This is the LOCAL interface a future Coordinator would call -- no
networking or MQTT here (see scripts/set_runtime_config.py and
scripts/demo_runtime_reconfig.py for how it's exercised without one).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .calibration import MIN_SCALE, CalibrationParams, Calibrator
from .records import config_fingerprint

HOT_SWAPPABLE_FIELDS = frozenset({"threshold", "calibration_scale"})

# Illustrative only (not exhaustive) -- makes rejection messages concrete.
ARTIFACT_DEPENDENT_EXAMPLES = (
    "window_size", "stride", "latent_dim", "filter_count", "kernel_size",
    "normalization_method", "model weights / TFLite variant",
)


class ConfigError(ValueError):
    """Raised by apply_new_config / reconfigure for an unknown or invalid request."""


def classify_param(name: str) -> str:
    """'hot_swappable' or 'requires_new_model_artifact'."""
    return "hot_swappable" if name in HOT_SWAPPABLE_FIELDS else "requires_new_model_artifact"


@dataclass(frozen=True)
class RuntimeConfig:
    """The currently active, hot-swappable detector configuration. Immutable:
    apply_new_config() returns a NEW instance rather than mutating this one."""

    threshold: float
    calibration_scale: float
    model_version: str   # artifact identity; informational, never changed by apply_new_config

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ConfigError("threshold must be a finite positive number")
        if not math.isfinite(self.calibration_scale) or self.calibration_scale < MIN_SCALE:
            raise ConfigError(f"calibration_scale must be finite and >= {MIN_SCALE}")

    @property
    def config_version(self) -> str:
        return config_fingerprint({"threshold": self.threshold,
                                   "calibration_scale": self.calibration_scale})

    @property
    def calibrator(self) -> Calibrator:
        """The calibration CENTER always tracks the active threshold, so
        score(threshold) == 0.5 stays exact; only the scale ("sensitivity") is an
        independently adjustable knob."""
        return Calibrator(CalibrationParams("log_ratio_sigmoid", self.threshold, self.calibration_scale))

    @classmethod
    def from_artifact(cls, threshold: float, calibration: dict, model_version: str) -> "RuntimeConfig":
        return cls(threshold=float(threshold), calibration_scale=float(calibration["scale"]),
                   model_version=model_version)

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "calibration_scale": self.calibration_scale,
                "model_version": self.model_version, "config_version": self.config_version}


def apply_new_config(current: RuntimeConfig, hparams: dict) -> RuntimeConfig:
    """Return a NEW RuntimeConfig with only the given hot-swappable fields changed.
    Never mutates ``current``. Raises ConfigError for any unknown or invalid field,
    BEFORE anything is applied (fail fast, no partial update)."""
    unknown = set(hparams) - HOT_SWAPPABLE_FIELDS
    if unknown:
        raise ConfigError(
            f"not hot-swappable, would need a new model artifact (retrain/reconvert): "
            f"{sorted(unknown)}. Hot-swappable fields are {sorted(HOT_SWAPPABLE_FIELDS)}. "
            f"Examples of artifact-dependent parameters: {list(ARTIFACT_DEPENDENT_EXAMPLES)}"
        )
    return replace(current, **hparams)  # re-runs __post_init__, so invalid values raise ConfigError
