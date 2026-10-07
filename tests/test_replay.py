import time

import numpy as np
import pytest

from anomaly_detector.data import SENSOR_COLUMNS
from anomaly_detector.replay import replay_samples
from anomaly_detector.stream import StreamingDetector

C = len(SENSOR_COLUMNS)


def _detector(**kw):
    defaults = dict(score_window_fn=lambda w: float(np.abs(w).mean()), n_channels=C,
                    window_size=8, reference_rows=20, threshold=0.5, stride=1)
    defaults.update(kw)
    return StreamingDetector(**defaults)


def test_fast_replay_counts_and_no_missed_deadlines():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((100, C))
    records, stats = replay_samples(samples, _detector(), speed=0.0, rss_every=10)
    s = stats.summary()
    assert s["n_samples"] == 100
    assert s["n_records"] == s["n_expected_records"] == 100 - 20
    assert s["missed_windows"] == 0
    assert s["n_deadline_overruns"] == 0  # speed=0 => no deadlines enforced
    assert len(records) == s["n_records"]


def test_record_cb_used_instead_of_list():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal((50, C))
    seen = []
    records, stats = replay_samples(samples, _detector(), speed=0.0, record_cb=seen.append)
    assert records == []  # callback mode: nothing accumulated in memory
    assert len(seen) == stats.n_records > 0


def test_deadline_overrun_detected_when_too_slow():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal((25, C))
    gaps = np.full(25, 0.01)  # 10 ms budget per sample

    def slow_score(w):
        time.sleep(0.03)
        return float(np.abs(w).mean())

    det = _detector(score_window_fn=slow_score)
    _, stats = replay_samples(samples, det, gaps_s=gaps, speed=1.0, rss_every=0)
    s = stats.summary()
    assert s["n_deadline_overruns"] > 0
    assert s["max_deadline_overrun_ms"] > 0


def test_no_deadline_overrun_when_fast_enough():
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((25, C))
    gaps = np.full(25, 0.05)
    _, stats = replay_samples(samples, _detector(), gaps_s=gaps, speed=1.0, rss_every=0)
    assert stats.summary()["n_deadline_overruns"] == 0


def test_memory_growth_summary_shape():
    rng = np.random.default_rng(4)
    samples = rng.standard_normal((200, C))
    _, stats = replay_samples(samples, _detector(), speed=0.0, rss_every=20)
    mem = stats.summary()["memory"]
    assert "rss_samples_mb" in mem and "growth" in mem


def test_on_tick_called_every_sample_by_default():
    rng = np.random.default_rng(5)
    samples = rng.standard_normal((30, C))
    ticks = []
    replay_samples(samples, _detector(), speed=0.0, on_tick=lambda i, det: ticks.append(i))
    assert ticks == list(range(30))


def test_on_tick_throttled_by_tick_every():
    rng = np.random.default_rng(6)
    samples = rng.standard_normal((30, C))
    ticks = []
    replay_samples(samples, _detector(), speed=0.0, on_tick=lambda i, det: ticks.append(i), tick_every=10)
    assert ticks == [0, 10, 20]


def test_on_tick_can_reconfigure_mid_stream_without_disrupting_records():
    from anomaly_detector.calibration import CalibrationParams, Calibrator
    from anomaly_detector.runtime_config import RuntimeConfig

    cal = Calibrator(CalibrationParams("log_ratio_sigmoid", 0.5, 0.2))
    det = _detector(threshold=0.5, calibrator=cal, model_version="m")
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((100, C))

    def on_tick(i, d):
        if i == 60:
            d.reconfigure(RuntimeConfig(threshold=0.9, calibration_scale=0.2, model_version="m"))

    records, stats = replay_samples(samples, det, speed=0.0, on_tick=on_tick)
    s = stats.summary()
    assert s["missed_windows"] == 0 and s["n_records"] == s["n_expected_records"]
    thresholds = [r.threshold for r in records]
    assert thresholds[0] == 0.5 and thresholds[-1] == 0.9
    assert sorted(set(thresholds)) == [0.5, 0.9]  # exactly one clean switch, no flapping
