"""SKAB loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMN = "datetime"
LABEL_COLUMN = "anomaly"
SENSOR_COLUMNS = (
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
)
SUBDIRS = ("valve1", "valve2", "other")  # anomaly-free/ is intentionally excluded


@dataclass(frozen=True)
class Series:
    """One SKAB file: sensor values (T, C), point labels (T,), timestamps (T,)."""

    name: str
    timestamps: np.ndarray
    values: np.ndarray
    labels: np.ndarray


def discover_files(root: str | Path) -> list[Path]:
    """Return labeled SKAB CSVs in a stable order (valve1, valve2, other; numeric)."""
    root = Path(root)
    files: list[Path] = []
    for sub in SUBDIRS:
        d = root / sub
        if d.is_dir():
            files.extend(sorted(d.glob("*.csv"), key=lambda p: int(p.stem)))
    if not files:
        raise FileNotFoundError(
            f"No SKAB CSV files found under {root}/{{{','.join(SUBDIRS)}}}. "
            "Clone https://github.com/waico/SKAB into data/raw/SKAB and check "
            "dataset.root in the config."
        )
    return files


def file_key(path: str | Path) -> str:
    """Stable identifier such as 'other/2.csv'."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def load_series(path: str | Path) -> Series:
    """Load and validate a single SKAB CSV."""
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    if len(df) == 0:
        raise ValueError(f"{path}: file is empty")
    required = [TIME_COLUMN, *SENSOR_COLUMNS, LABEL_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    values = df[list(SENSOR_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: sensor values contain NaN/inf")

    raw_labels = df[LABEL_COLUMN]
    if raw_labels.isna().any() or not raw_labels.isin([0, 1]).all():
        raise ValueError(f"{path}: '{LABEL_COLUMN}' must contain only 0/1")
    labels = raw_labels.to_numpy().astype(np.int8)

    timestamps = pd.to_datetime(df[TIME_COLUMN]).to_numpy()
    name = file_key(path)
    return Series(name=name, timestamps=timestamps, values=values, labels=labels)


def check_reference_segment(series: Series, reference_rows: int) -> None:
    """Fail loudly if the assumed anomaly-free reference segment is not usable."""
    if len(series.labels) <= reference_rows:
        raise ValueError(
            f"{series.name}: only {len(series.labels)} rows, need more than "
            f"reference_rows={reference_rows} to have a test region"
        )
    if series.labels[:reference_rows].any():
        raise ValueError(
            f"{series.name}: reference segment (first {reference_rows} rows) "
            "contains labeled anomalies; the reference-interval assumption is wrong"
        )
