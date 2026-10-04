"""Shared evaluation harness: every detector is scored by exactly this code."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .baseline import predict, window_scores
from .config import Config
from .data import Series, check_reference_segment, discover_files, file_key, load_series
from .metrics import compute_metrics, roc_auc
from .preprocessing import Normalizer
from .windowing import make_windows

SCHEMA_VERSION = "scorecard_v1"  # v1 is additive: roc_auc / diagnostics may be present


@dataclass
class PreparedFile:
    """One file after reference-only normalization."""

    name: str
    x: np.ndarray            # normalized (T, C)
    labels: np.ndarray       # (T,)
    normalizer: dict         # fitted stats, for later deployment


@dataclass
class FileResult:
    name: str
    y_true: np.ndarray
    scores: np.ndarray


def prepare_file(series: Series, cfg: Config) -> PreparedFile:
    """check reference -> fit normalizer on reference rows only -> transform file."""
    ref_rows = cfg.dataset.reference_rows
    check_reference_segment(series, ref_rows)
    normalizer = Normalizer(cfg.preprocessing.normalization_method)
    normalizer.fit(series.values[:ref_rows])  # reference only, never the full file
    return PreparedFile(
        series.name, normalizer.transform(series.values), series.labels,
        normalizer.to_dict(),
    )


def load_prepared(cfg: Config) -> tuple[list[PreparedFile], list[str]]:
    """Load, validate and normalize every non-excluded file. Returns (files, excluded)."""
    all_files = discover_files(cfg.dataset.root)
    excl = set(cfg.dataset.exclude_files)
    excluded = [file_key(p) for p in all_files if file_key(p) in excl]
    kept = [p for p in all_files if file_key(p) not in excl]
    return [prepare_file(load_series(p), cfg) for p in kept], excluded


def test_windows(pf: PreparedFile, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose last sample lies in the test region (>= reference_rows).

    Returns (windows (N, W, C), y_true (N,)) with the label taken at the window end.
    """
    windows, end_idx = make_windows(pf.x, cfg.windowing.window_size, cfg.windowing.stride)
    keep = end_idx >= cfg.dataset.reference_rows
    return windows[keep], pf.labels[end_idx[keep]]


def reference_windows(pf: PreparedFile, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) windows, both fully inside the anomaly-free reference."""
    d, w = cfg.dataset, cfg.windowing
    train, _ = make_windows(pf.x[: d.train_rows], w.window_size, w.stride)
    val, _ = make_windows(pf.x[d.train_rows : d.reference_rows], w.window_size, w.stride)
    return train, val


def score_files(
    prepared: list[PreparedFile], cfg: Config,
    score_fn: Callable[[np.ndarray], np.ndarray],
) -> list[FileResult]:
    out = []
    for pf in prepared:
        windows, y_true = test_windows(pf, cfg)
        out.append(FileResult(pf.name, y_true, score_fn(windows)))
    return out


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_provenance(cfg: Config, extra_versions: dict | None = None) -> dict:
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
        **(extra_versions or {}),
    }


def far_by_segment(results: list[FileResult], threshold: float) -> dict:
    """False-alarm rate on normal test points before the first / after the last
    labeled anomaly (files without anomalies are skipped). Diagnostic only."""
    seg = {"pre_anomaly": [0, 0], "post_anomaly": [0, 0]}
    for r in results:
        if r.y_true.sum() == 0:
            continue
        first = int(r.y_true.argmax())
        last = len(r.y_true) - 1 - int(r.y_true[::-1].argmax())
        pred = r.scores > threshold
        for key, sl in (("pre_anomaly", slice(0, first)), ("post_anomaly", slice(last + 1, None))):
            normal = r.y_true[sl] == 0
            seg[key][0] += int((pred[sl] & normal).sum())
            seg[key][1] += int(normal.sum())
    return {
        k: {"false_alarms": v[0], "normal_points": v[1], "far": (v[0] / v[1] if v[1] else None)}
        for k, v in seg.items()
    }


def build_scorecard(
    results: list[FileResult], cfg: Config, *, threshold: float, threshold_source: str,
    excluded: list[str], extra_provenance: dict | None = None, extra: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """Apply one threshold to all scores and assemble the versioned scorecard."""
    y_true = np.concatenate([r.y_true for r in results])
    scores = np.concatenate([r.scores for r in results])
    pooled = compute_metrics(y_true, predict(scores, threshold))
    per_file = [
        {"file": r.name, **compute_metrics(r.y_true, predict(r.scores, threshold))}
        for r in results
    ]
    card = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SKAB",
        "model": model_name or cfg.detector.name,
        "f1": pooled["f1"],
        "far": pooled["far"],
        "mar": pooled["mar"],
        "roc_auc": roc_auc(y_true, scores),
        "threshold": float(threshold),
        "counts": {k: pooled[k] for k in ("tp", "fp", "tn", "fn", "n_points")},
        "protocol": {
            "reference": f"first {cfg.dataset.reference_rows} rows of each file",
            "excluded_files": excluded,
            "exclusion_reason": "reference segment contains labeled anomalies",
            "normalization_fit": "reference rows only, per file",
            "scored_points": "windows whose last sample index >= reference_rows",
            "label_assignment": "window end",
            "pooling": "point-wise, pooled across files (single global threshold)",
            "threshold_source": threshold_source,
        },
        "config": cfg.to_dict(),
        "n_files": len(results),
        "provenance": collect_provenance(cfg, extra_provenance),
        "per_file": per_file,
    }
    extra = dict(extra or {})
    diagnostics = {"far_by_segment": far_by_segment(results, threshold)}
    diagnostics.update(extra.pop("diagnostics", {}))
    card["diagnostics"] = diagnostics
    card.update(extra)
    return card


def run_evaluation(cfg: Config) -> dict:
    """Evaluate the z-score baseline over every labeled SKAB file."""
    np.random.seed(cfg.seed)
    prepared, excluded = load_prepared(cfg)
    results = score_files(prepared, cfg, window_scores)
    return build_scorecard(
        results, cfg, threshold=cfg.detector.threshold,
        threshold_source="fixed value from config (not tuned)", excluded=excluded,
    )


def write_scorecard(scorecard: dict, metrics_dir: str | Path) -> Path:
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scorecard_{scorecard['model']}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)
        fh.write("\n")
    return path


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def format_scorecard(card: dict) -> str:
    lines = [f"{'file':<16}{'points':>8}{'f1':>8}{'far':>8}{'mar':>8}"]
    for row in card["per_file"]:
        lines.append(
            f"{row['file']:<16}{row['n_points']:>8}"
            f"{_fmt(row['f1']):>8}{_fmt(row['far']):>8}{_fmt(row['mar']):>8}"
        )
    lines.append("-" * 48)
    lines.append(
        f"{'POOLED':<16}{card['counts']['n_points']:>8}"
        f"{_fmt(card['f1']):>8}{_fmt(card['far']):>8}{_fmt(card['mar']):>8}"
    )
    lines.append(
        f"\nmodel={card['model']}  files={card['n_files']}  "
        f"threshold={card['threshold']:.6g}  roc_auc={_fmt(card['roc_auc'])}"
    )
    lines.append(f"threshold source: {card['protocol']['threshold_source']}")
    lines.append(f"excluded={card['protocol']['excluded_files']}")
    return "\n".join(lines)
