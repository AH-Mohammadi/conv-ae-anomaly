"""Evaluation harness: run a detector over all SKAB files, produce a scorecard."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import predict, window_scores
from .config import Config
from .data import Series, check_reference_segment, discover_files, file_key, load_series
from .metrics import compute_metrics
from .preprocessing import Normalizer
from .windowing import make_windows

SCHEMA_VERSION = "scorecard_v1"


@dataclass
class FileResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray


def evaluate_series(series: Series, cfg: Config) -> FileResult:
    """Score one file.

    reference rows -> fit normalizer -> transform whole file -> window ->
    score windows whose last sample lies in the test region (>= reference_rows).
    """
    ref_rows = cfg.dataset.reference_rows
    check_reference_segment(series, ref_rows)

    normalizer = Normalizer(cfg.preprocessing.normalization_method)
    normalizer.fit(series.values[:ref_rows])  # reference only, never the full file
    x = normalizer.transform(series.values)

    windows, end_idx = make_windows(
        x, cfg.windowing.window_size, cfg.windowing.stride
    )
    keep = end_idx >= ref_rows
    scores = window_scores(windows[keep])
    y_pred = predict(scores, cfg.detector.threshold)
    y_true = series.labels[end_idx[keep]]
    return FileResult(series.name, y_true, y_pred, scores)


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_provenance(cfg: Config) -> dict:
    repo = _git_head(Path.cwd())
    skab = _git_head(Path(cfg.dataset.root).parent)
    if skab == repo:  # data dir is not its own git checkout
        skab = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": repo,
        "skab_commit": skab,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_evaluation(cfg: Config) -> dict:
    """Evaluate the configured detector over every labeled SKAB file."""
    np.random.seed(cfg.seed)  # nothing stochastic yet; kept for later iterations
    all_files = discover_files(cfg.dataset.root)
    excluded = [file_key(p) for p in all_files if file_key(p) in cfg.dataset.exclude_files]
    files = [p for p in all_files if file_key(p) not in cfg.dataset.exclude_files]
    results = [evaluate_series(load_series(p), cfg) for p in files]

    pooled = compute_metrics(
        np.concatenate([r.y_true for r in results]),
        np.concatenate([r.y_pred for r in results]),
    )
    per_file = [{"file": r.name, **compute_metrics(r.y_true, r.y_pred)} for r in results]

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SKAB",
        "model": cfg.detector.name,
        "f1": pooled["f1"],
        "far": pooled["far"],
        "mar": pooled["mar"],
        "counts": {k: pooled[k] for k in ("tp", "fp", "tn", "fn", "n_points")},
        "protocol": {
            "reference": f"first {cfg.dataset.reference_rows} rows of each file",
            "excluded_files": excluded,
            "exclusion_reason": "reference segment contains labeled anomalies",
            "normalization_fit": "reference rows only, per file",
            "scored_points": "windows whose last sample index >= reference_rows",
            "label_assignment": "window end",
            "pooling": "point-wise, pooled across files",
            "threshold_source": "fixed value from config (not tuned)",
        },
        "config": cfg.to_dict(),
        "n_files": len(results),
        "provenance": collect_provenance(cfg),
        "per_file": per_file,
    }


def write_scorecard(scorecard: dict, metrics_dir: str | Path) -> Path:
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scorecard_{scorecard['model']}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)
        fh.write("\n")
    return path
