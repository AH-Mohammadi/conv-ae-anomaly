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


def _detector_with_calibration(threshold=1.0, scale=0.2, model_version="m1"):
    from anomaly_detector.calibration import CalibrationParams, Calibrator

    cal = Calibrator(CalibrationParams("log_ratio_sigmoid", threshold, scale))
    return _detector(threshold=threshold, calibrator=cal, model_version=model_version)


def test_reconfigure_atomically_swaps_without_touching_buffer_or_position():
    from anomaly_detector.runtime_config import RuntimeConfig

    det = _detector_with_calibration(threshold=1.0, scale=0.2)
    rng = np.random.default_rng(10)
    for _ in range(25):  # cross into SCORING
        det.process_sample(rng.standard_normal(C))
    assert det.stage is Stage.SCORING

    buf_id = id(det._ring._buf)
    buf_contents = det._ring.window().copy()
    sample_index_before = det.sample_index
    old_version = det.config_version

    new_cfg = RuntimeConfig(threshold=2.0, calibration_scale=0.5, model_version="m1")
    det.reconfigure(new_cfg)

    assert id(det._ring._buf) == buf_id                     # same object, not reallocated
    assert np.array_equal(det._ring.window(), buf_contents)  # contents untouched
    assert det.sample_index == sample_index_before            # stream position untouched
    assert det.threshold == 2.0 and det.calibrator.params.scale == 0.5
    assert det.config_version != old_version
    assert det.calibrator.transform_one(2.0) == pytest.approx(0.5, abs=1e-12)

    rec = det.process_sample(rng.standard_normal(C))  # stream continues normally
    assert rec is not None and rec.threshold == 2.0 and rec.config_version == det.config_version


def test_reconfigure_rejects_mismatched_model_version_without_mutating_state():
    from anomaly_detector.runtime_config import ConfigError, RuntimeConfig

    det = _detector_with_calibration(model_version="model_A")
    before = (det.threshold, det.calibrator.params.scale, det.config_version)
    with pytest.raises(ConfigError, match="model_A"):
        det.reconfigure(RuntimeConfig(threshold=9.0, calibration_scale=0.9, model_version="model_B"))
    assert (det.threshold, det.calibrator.params.scale, det.config_version) == before


def test_reconfigure_without_calibrator_raises():
    from anomaly_detector.runtime_config import ConfigError, RuntimeConfig

    det = _detector()  # no calibrator given
    with pytest.raises(ConfigError):
        det.current_runtime_config()
    with pytest.raises(ConfigError):
        det.reconfigure(RuntimeConfig(threshold=1.0, calibration_scale=0.2, model_version=""))


def test_current_runtime_config_round_trip_is_a_no_op():
    det = _detector_with_calibration(threshold=1.3, scale=0.25, model_version="m")
    snap = det.current_runtime_config()
    assert snap.threshold == 1.3 and snap.calibration_scale == 0.25 and snap.model_version == "m"
    det.reconfigure(snap)
    assert det.threshold == 1.3 and det.calibrator.params.scale == 0.25


def test_records_carry_config_version_active_at_emission_time():
    """Records emitted before a mid-stream reconfigure() keep the OLD config_version;
    records emitted after carry the NEW one -- each record reflects the config that
    was active when it was produced, not whatever the detector's config is later."""
    from anomaly_detector.runtime_config import RuntimeConfig

    det = _detector_with_calibration(threshold=1.0, scale=0.2)
    rng = np.random.default_rng(11)
    for _ in range(21):  # advance past the reference phase (reference_rows=20) into SCORING
        det.process_sample(rng.standard_normal(C))
    assert det.stage is Stage.SCORING

    before = [rec for _ in range(5)
             if (rec := det.process_sample(rng.standard_normal(C))) is not None]
    assert len(before) == 5

    det.reconfigure(RuntimeConfig(threshold=3.0, calibration_scale=0.2, model_version="m1"))

    after = [rec for _ in range(5)
            if (rec := det.process_sample(rng.standard_normal(C))) is not None]
    assert len(after) == 5

    before_versions = {r.config_version for r in before}
    after_versions = {r.config_version for r in after}
    assert len(before_versions) == 1 and len(after_versions) == 1
    assert before_versions != after_versions
    assert all(r.threshold == 1.0 for r in before)
    assert all(r.threshold == 3.0 for r in after)
