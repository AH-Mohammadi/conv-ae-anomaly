"""Streaming detector: ring buffer + normalizer, fed one sample at a time.

Startup protocol mirrors the offline harness exactly (evaluate.py):
  * samples [0, reference_rows) are treated as the anomaly-free reference -> no
    score is emitted for them; once all reference_rows have arrived, the
    normalizer is fit (reference only) and the last window_size of them seed
    the ring buffer, so the next sample is immediately scorable
  * from sample index reference_rows onward, every `stride`-th sample (counted
    from the start of scoring) produces a score, exactly as make_windows() would
Offline parity is exact for stride=1 (the default throughout this project); for
stride>1 the phase is anchored to the start of scoring rather than to sample 0,
which is a documented, deliberate simplification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter_ns
from typing import Callable

import numpy as np

from .calibration import Calibrator
from .preprocessing import Normalizer
from .ringbuffer import RingBuffer


class Stage(Enum):
    REFERENCE = "reference"   # collecting the anomaly-free reference, not scoring yet
    SCORING = "scoring"


@dataclass
class StreamRecord:
    sample_index: int
    score: float                       # raw reconstruction error
    threshold: float
    is_anomaly: bool
    processing_ns: int
    stage: str
    calibrated_score: float | None = None  # in [0, 1]; set only if a Calibrator was given


class StreamingDetector:
    """Feed samples one at a time via ``process_sample``; get a StreamRecord or None."""

    def __init__(self, score_window_fn: Callable[[np.ndarray], float], n_channels: int,
                window_size: int, reference_rows: int, threshold: float,
                stride: int = 1, normalization_method: str = "zscore",
                calibrator: Calibrator | None = None) -> None:
        if reference_rows <= window_size:
            raise ValueError("reference_rows must be > window_size")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.score_window_fn = score_window_fn
        self.n_channels = n_channels
        self.window_size = window_size
        self.reference_rows = reference_rows
        self.threshold = threshold
        self.stride = stride
        self.calibrator = calibrator

        self._normalizer = Normalizer(normalization_method)
        self._ref_buf = np.empty((reference_rows, n_channels), dtype=np.float64)
        self._ring = RingBuffer(window_size, n_channels)
        self.stage = Stage.REFERENCE
        self.sample_index = 0          # 0-based count of samples seen
        self._since_scoring_start = 0  # for stride alignment

    def _enter_scoring(self) -> None:
        self._normalizer.fit(self._ref_buf)
        for row in self._ref_buf[-self.window_size :]:
            self._ring.push(self._normalizer.transform(row[None, :])[0])
        self.stage = Stage.SCORING

    def process_sample(self, raw_sample) -> StreamRecord | None:
        """Returns a StreamRecord once scoring has started and this sample lands on
        a scored stride position; otherwise None (startup, or a skipped stride)."""
        t0 = perf_counter_ns()
        raw = np.asarray(raw_sample, dtype=np.float64)
        if raw.shape != (self.n_channels,):
            raise ValueError(f"expected shape ({self.n_channels},), got {raw.shape}")

        if self.stage is Stage.REFERENCE:
            self._ref_buf[self.sample_index] = raw
            self.sample_index += 1
            if self.sample_index == self.reference_rows:
                self._enter_scoring()
            return None

        self._ring.push(self._normalizer.transform(raw[None, :])[0])
        self.sample_index += 1
        emit = self._since_scoring_start % self.stride == 0
        self._since_scoring_start += 1
        if not emit:
            return None
        score = float(self.score_window_fn(self._ring.window()))
        calibrated = self.calibrator.transform_one(score) if self.calibrator is not None else None
        rec = StreamRecord(self.sample_index - 1, score, self.threshold,
                           score > self.threshold, perf_counter_ns() - t0, self.stage.value,
                           calibrated_score=calibrated)
        return rec
