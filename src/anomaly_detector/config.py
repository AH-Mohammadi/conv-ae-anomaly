"""Explicit, validated configuration loaded from YAML."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

NORMALIZATION_METHODS = ("zscore", "minmax")
DETECTORS = ("zscore_baseline",)


@dataclass(frozen=True)
class DatasetConfig:
    root: str = "data/raw/SKAB/data"
    reference_rows: int = 400
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
class DetectorConfig:
    name: str = "zscore_baseline"
    threshold: float = 3.0


@dataclass(frozen=True)
class OutputConfig:
    metrics_dir: str = "reports/metrics"


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.dataset.reference_rows < 2:
            raise ValueError("dataset.reference_rows must be >= 2")
        if self.windowing.window_size < 1:
            raise ValueError("windowing.window_size must be >= 1")
        if self.windowing.stride < 1:
            raise ValueError("windowing.stride must be >= 1")
        if self.preprocessing.normalization_method not in NORMALIZATION_METHODS:
            raise ValueError(
                f"normalization_method must be one of {NORMALIZATION_METHODS}, "
                f"got {self.preprocessing.normalization_method!r}"
            )
        if self.detector.name not in DETECTORS:
            raise ValueError(f"detector.name must be one of {DETECTORS}")
        if not math.isfinite(self.detector.threshold) or self.detector.threshold <= 0:
            raise ValueError("detector.threshold must be a finite positive number")
        if (
            self.detector.name == "zscore_baseline"
            and self.preprocessing.normalization_method != "zscore"
        ):
            raise ValueError("zscore_baseline requires normalization_method='zscore'")

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
    known = {"dataset", "windowing", "preprocessing", "detector", "output", "seed"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return Config(
        dataset=_section(DatasetConfig, raw.get("dataset"), "dataset"),
        windowing=_section(WindowingConfig, raw.get("windowing"), "windowing"),
        preprocessing=_section(
            PreprocessingConfig, raw.get("preprocessing"), "preprocessing"
        ),
        detector=_section(DetectorConfig, raw.get("detector"), "detector"),
        output=_section(OutputConfig, raw.get("output"), "output"),
        seed=int(raw.get("seed", 0)),
    )
