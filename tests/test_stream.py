import numpy as np
import pytest

from anomaly_detector.data import SENSOR_COLUMNS
from anomaly_detector.ringbuffer import RingBuffer
from anomaly_detector.stream import Stage, StreamingDetector

C = len(SENSOR_COLUMNS)


def _const_score_fn(w):
    return float(np.abs(w).mean())


def _detector(**kw):
    defaults = dict(score_window_fn=_const_score_fn, n_channels=C, window_size=8,
                    reference_rows=20, threshold=1.0, stride=1)
    defaults.update(kw)
    return StreamingDetector(**defaults)


def test_no_record_during_reference_phase():
    det = _detector()
    rng = np.random.default_rng(0)
    for i in range(19):
        assert det.process_sample(rng.standard_normal(C)) is None
        assert det.stage is Stage.REFERENCE
    assert det.sample_index == 19


def test_first_record_right_after_reference_fills():
    det = _detector(reference_rows=20, window_size=8)
    rng = np.random.default_rng(1)
    rec = None
    for i in range(21):  # reference_rows=20 fills on call 20; first record arrives on call 21
        rec = det.process_sample(rng.standard_normal(C))
    assert det.stage is Stage.SCORING
    assert rec is not None and rec.sample_index == 20


def test_stride_skips_samples():
    det = _detector(reference_rows=20, window_size=8, stride=3)
    rng = np.random.default_rng(2)
    records = [det.process_sample(rng.standard_normal(C)) for _ in range(20 + 9)]
    emitted = [r for r in records if r is not None]
    assert len(emitted) == 3  # samples 19, 22, 25 (0-indexed) within the loop


def test_threshold_decision():
    det = _detector(score_window_fn=lambda w: 0.0, threshold=0.5)
    rng = np.random.default_rng(3)
    rec = None
    for _ in range(21):
        rec = det.process_sample(rng.standard_normal(C))
    assert rec.score == 0.0 and not rec.is_anomaly


def test_invalid_sample_shape():
    det = _detector()
    with pytest.raises(ValueError):
        det.process_sample(np.zeros(C - 1))


def test_invalid_construction():
    with pytest.raises(ValueError):
        _detector(reference_rows=5, window_size=8)  # reference must exceed window
    with pytest.raises(ValueError):
        _detector(stride=0)


def test_repeated_inference_is_stable():
    """Feeding a long stream must not grow the ring buffer or drift its identity."""
    det = _detector(reference_rows=20, window_size=8)
    rng = np.random.default_rng(4)
    buf_id = id(det._ring._buf)
    for _ in range(5000):
        det.process_sample(rng.standard_normal(C))
    assert id(det._ring._buf) == buf_id
    assert det._ring.total_pushed <= 5000  # never exceeds samples actually pushed into it


def test_matches_offline_harness(synthetic_root):
    """The streaming detector, fed one file sample-by-sample, must reproduce the SAME
    scores/labels/predictions as the offline shared harness (baseline score) on that file."""
    from anomaly_detector.baseline import predict as offline_predict
    from anomaly_detector.baseline import window_scores
    from anomaly_detector.config import Config, DatasetConfig
    from anomaly_detector.data import load_series
    from anomaly_detector.evaluate import prepare_file, test_windows

    cfg = Config(dataset=DatasetConfig(root=str(synthetic_root), reference_rows=400))
    path = next((synthetic_root / "valve1").glob("*.csv"))
    series = load_series(path)
    pf = prepare_file(series, cfg)
    windows, y_true = test_windows(pf, cfg)
    offline_scores = window_scores(windows)
    offline_pred = offline_predict(offline_scores, cfg.detector.threshold)

    def score_fn(w):  # same rule as baseline.window_scores for a single window
        return float(np.abs(w).mean(axis=0).max())

    det = StreamingDetector(score_fn, n_channels=C, window_size=cfg.windowing.window_size,
                            reference_rows=cfg.dataset.reference_rows, threshold=cfg.detector.threshold,
                            stride=cfg.windowing.stride)
    records = [r for r in (det.process_sample(row) for row in series.values) if r is not None]

    assert len(records) == len(y_true)
    stream_scores = np.array([r.score for r in records])
    assert np.allclose(stream_scores, offline_scores, atol=1e-9)
    stream_pred = np.array([int(r.is_anomaly) for r in records])
    assert np.array_equal(stream_pred, offline_pred)
    stream_labels = series.labels[[r.sample_index for r in records]]
    assert np.array_equal(stream_labels, y_true)


def test_streaming_with_calibrator_matches_raw_decision_and_is_bounded():
    from anomaly_detector.calibration import Calibrator
    rng = np.random.default_rng(7)
    val = np.abs(rng.standard_normal(500)).astype(np.float64) + 0.1
    threshold = float(np.percentile(val, 99))
    cal = Calibrator.fit(val, threshold)

    det = _detector(reference_rows=20, window_size=8, threshold=threshold)
    det.calibrator = cal
    records = []
    for _ in range(500):
        rec = det.process_sample(rng.standard_normal(C))
        if rec is not None:
            records.append(rec)
    assert records  # some were emitted
    for r in records:
        assert r.calibrated_score is not None
        assert 0.0 <= r.calibrated_score <= 1.0
        assert (r.calibrated_score > 0.5) == r.is_anomaly


def test_streaming_without_calibrator_leaves_calibrated_score_none():
    det = _detector()
    rng = np.random.default_rng(8)
    rec = None
    for _ in range(21):
        rec = det.process_sample(rng.standard_normal(C))
    assert rec.calibrated_score is None
