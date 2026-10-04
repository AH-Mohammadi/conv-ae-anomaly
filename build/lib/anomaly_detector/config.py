"""Explicit, validated configuration loaded from YAML.

Hyperparameter classes (relevant for the future runtime-config iteration):
  * hot-swappable (no new model artifact needed): detector.threshold
  * model-artifact dependent (needs retraining/new .tflite): windowing.window_size,
    model.latent_dim, model.filter_count, model.kernel_size, trained weights
  * training-only: training.*
  * compression-only (offline conversion): compression.*
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

NORMALIZATION_METHODS = ("zscore", "minmax")
DETECTORS = ("zscore_baseline", "conv_ae")


@dataclass(frozen=True)
class DatasetConfig:
    root: str = "data/raw/SKAB/data"
    reference_rows: int = 400
    # Split of the anomaly-free reference: rows [0, train_rows) train the model,
    # rows [train_rows, reference_rows) are validation (threshold / early stopping).
    train_rows: int = 300
    # Files whose first `reference_rows` rows are NOT anomaly-free. Verified on the
    # SKAB repo: only other/2.csv (first labeled anomaly at row 104).
    exclude_files: list[str] = field(default_factory=lambda: ["other/2.csv"])


@dataclass(frozen=True)
class WindowingConfig:
    window_size: int = 32
    stride: int = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    normalization_method: str = "zscore"


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 16
    filter_count: int = 16
    kernel_size: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 10                    # early stopping on validation loss
    threshold_percentile: float = 99.0    # of validation reconstruction errors


@dataclass(frozen=True)
class CompressionConfig:
    # Representative (calibration) windows for INT8: anomaly-free TRAIN windows only.
    calibration_samples: int = 500
    # Acceptance criteria for INT8 vs the FP32 Keras model (set before measuring):
    max_metric_change: float = 0.03   # absolute |delta| allowed for F1, FAR and MAR
    max_auc_drop: float = 0.02        # allowed ROC-AUC decrease
    # "Wide" INT8 variant: calibration set = the windows above plus a copy scaled by a
    # random per-window factor in [1, k]. Added after observing that test inputs reach
    # |z| ~ 250 while anomaly-free data stays within |z| ~ 5.
    wide_calibration_max_amplitude: float = 4.0


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "zscore_baseline"
    # zscore_baseline: required, fixed. conv_ae: None -> derived from validation
    # errors via training.threshold_percentile; a number overrides that.
    threshold: float | None = 3.0


@dataclass(frozen=True)
class OutputConfig:
    metrics_dir: str = "reports/metrics"
    model_dir: str = "models/conv_ae_v1"


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        d, w, m, t = self.dataset, self.windowing, self.model, self.training
        if d.reference_rows < 2:
            raise ValueError("dataset.reference_rows must be >= 2")
        if w.window_size < 1:
            raise ValueError("windowing.window_size must be >= 1")
        if w.stride < 1:
            raise ValueError("windowing.stride must be >= 1")
        if self.preprocessing.normalization_method not in NORMALIZATION_METHODS:
            raise ValueError(
                f"normalization_method must be one of {NORMALIZATION_METHODS}, "
                f"got {self.preprocessing.normalization_method!r}"
            )
        if self.detector.name not in DETECTORS:
            raise ValueError(f"detector.name must be one of {DETECTORS}")

        thr = self.detector.threshold
        if thr is not None and (not math.isfinite(thr) or thr <= 0):
            raise ValueError("detector.threshold must be a finite positive number")

        if self.detector.name == "zscore_baseline":
            if thr is None:
                raise ValueError("zscore_baseline requires a numeric detector.threshold")
            if self.preprocessing.normalization_method != "zscore":
                raise ValueError("zscore_baseline requires normalization_method='zscore'")

        if self.detector.name == "conv_ae":
            if w.window_size % 4 != 0:
                raise ValueError("conv_ae requires windowing.window_size divisible by 4")
            if m.latent_dim < 1 or m.filter_count < 1:
                raise ValueError("model.latent_dim and model.filter_count must be >= 1")
            if m.kernel_size < 1 or m.kernel_size % 2 == 0:
                raise ValueError("model.kernel_size must be a positive odd integer")
            if not (w.window_size <= d.train_rows < d.reference_rows):
                raise ValueError(
                    "need window_size <= dataset.train_rows < dataset.reference_rows"
                )
            if d.reference_rows - d.train_rows < w.window_size:
                raise ValueError(
                    "validation region (reference_rows - train_rows) must hold "
                    "at least one window"
                )
            if t.epochs < 1 or t.batch_size < 1 or t.patience < 1:
                raise ValueError("training.epochs/batch_size/patience must be >= 1")
            if not t.learning_rate > 0:
                raise ValueError("training.learning_rate must be > 0")
            if not (50.0 < t.threshold_percentile < 100.0):
                raise ValueError("training.threshold_percentile must be in (50, 100)")

        c = self.compression
        if c.calibration_samples < 1:
            raise ValueError("compression.calibration_samples must be >= 1")
        if c.max_metric_change < 0 or c.max_auc_drop < 0:
            raise ValueError("compression tolerances must be >= 0")
        if c.wide_calibration_max_amplitude < 1.0:
            raise ValueError("compression.wide_calibration_max_amplitude must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls, raw: dict[str, Any] | None, name: str):
    raw = raw or {}
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid keys in config section '{name}': {exc}") from exc


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    known = {"dataset", "windowing", "preprocessing", "model", "training",
             "compression", "detector", "output", "seed"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return Config(
        dataset=_section(DatasetConfig, raw.get("dataset"), "dataset"),
        windowing=_section(WindowingConfig, raw.get("windowing"), "windowing"),
        preprocessing=_section(
            PreprocessingConfig, raw.get("preprocessing"), "preprocessing"
        ),
        model=_section(ModelConfig, raw.get("model"), "model"),
        training=_section(TrainingConfig, raw.get("training"), "training"),
        compression=_section(CompressionConfig, raw.get("compression"), "compression"),
        detector=_section(DetectorConfig, raw.get("detector"), "detector"),
        output=_section(OutputConfig, raw.get("output"), "output"),
        seed=int(raw.get("seed", 0)),
    )
