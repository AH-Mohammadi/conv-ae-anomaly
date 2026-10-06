"""Replay driver: feeds samples into a StreamingDetector at (scaled) real time,
and tracks stability over a long run: missed deadlines, memory growth, latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .stream import StreamingDetector, StreamRecord

REPLAY_SCHEMA = "replay_summary_v1"


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return round(int(line.split()[1]) / 1024, 2)
    except OSError:
        pass
    return None


@dataclass
class StabilityStats:
    n_samples: int = 0
    n_records: int = 0
    n_expected_records: int = 0
    n_missed_deadlines: int = 0           # processing took longer than the inter-arrival gap
    max_deadline_overrun_ms: float = 0.0
    processing_ms: list[float] = field(default_factory=list)
    rss_samples_mb: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        p = np.asarray(self.processing_ms) if self.processing_ms else np.empty(0)
        rss = [v for v in self.rss_samples_mb if v is not None]
        growth = None
        if len(rss) >= 3:
            slope = float(np.polyfit(np.arange(len(rss)), rss, 1)[0])
            growth = {"first_mb": rss[0], "last_mb": rss[-1], "slope_mb_per_sample_window": slope}
        return {
            "n_samples": self.n_samples, "n_records": self.n_records,
            "n_expected_records": self.n_expected_records,
            "missed_windows": self.n_expected_records - self.n_records,
            "n_deadline_overruns": self.n_missed_deadlines,
            "max_deadline_overrun_ms": self.max_deadline_overrun_ms,
            "processing_ms": {} if p.size == 0 else {
                "mean": float(p.mean()), "median": float(np.median(p)),
                "p99": float(np.percentile(p, 99)), "max": float(p.max())},
            "memory": {"rss_samples_mb": rss, "growth": growth},
        }


def replay_samples(samples: np.ndarray, detector: StreamingDetector, *,
                   gaps_s: np.ndarray | None = None, speed: float = 0.0,
                   rss_every: int = 500, record_cb=None) -> tuple[list[StreamRecord], StabilityStats]:
    """Feed ``samples`` (N, C) into ``detector`` one at a time.

    gaps_s[i] is the real-world gap BEFORE sample i (gaps_s[0] ignored). If given and
    speed > 0, the driver sleeps gaps_s[i] / speed before processing sample i, and a
    deadline is considered missed if processing took longer than that gap. speed <= 0
    (default) processes as fast as possible, for tests and long stability soaks.
    record_cb(record) is called for every emitted StreamRecord (e.g. to stream to disk
    instead of keeping every record in memory).
    """
    n = samples.shape[0]
    stats = StabilityStats(n_samples=n)
    stats.n_expected_records = max(0, (n - detector.reference_rows + detector.stride - 1)
                                   // detector.stride) if n > detector.reference_rows else 0
    records: list[StreamRecord] = []

    for i in range(n):
        gap = float(gaps_s[i]) if gaps_s is not None and i > 0 else 0.0
        if speed > 0 and gap > 0:
            time.sleep(gap / speed)
        t0 = time.perf_counter()  # measured AFTER the pacing sleep: processing time only
        rec = detector.process_sample(samples[i])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats.processing_ms.append(elapsed_ms)
        if speed > 0 and gap > 0:
            budget_ms = (gap / speed) * 1000.0
            if elapsed_ms > budget_ms:
                stats.n_missed_deadlines += 1
                stats.max_deadline_overrun_ms = max(stats.max_deadline_overrun_ms, elapsed_ms - budget_ms)
        if rec is not None:
            stats.n_records += 1
            if record_cb is not None:
                record_cb(rec)
            else:
                records.append(rec)
        if rss_every and i % rss_every == 0:
            stats.rss_samples_mb.append(_rss_mb())

    return records, stats
