import numpy as np
import pytest

from anomaly_detector.calibration import Calibrator, CalibrationParams
from anomaly_detector.records import config_fingerprint


def _val_errors(n=500, seed=0, mean=0.6, std=0.1):
    return np.random.default_rng(seed).normal(mean, std, n).clip(min=1e-3)


def test_boundary_is_exactly_half():
    thr = 0.884
    cal = Calibrator.fit(_val_errors(), thr)
    assert cal.transform_one(thr) == pytest.approx(0.5, abs=1e-12)


def test_monotonic_and_bounded():
    cal = Calibrator.fit(_val_errors(), 0.884)
    errors = np.sort(np.concatenate([np.linspace(0.01, 5, 200), [1e-6, 1e6]]))
    scores = cal.transform(errors)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert np.all(np.diff(scores) >= -1e-12)  # monotonic non-decreasing


def test_decision_matches_raw_threshold():
    rng = np.random.default_rng(1)
    thr = 0.9
    val = _val_errors(seed=2)
    cal = Calibrator.fit(val, thr)
    test_errors = rng.lognormal(mean=0.0, sigma=2.0, size=5000) * thr
    raw_decision = test_errors > thr
    cal_decision = cal.transform(test_errors) > 0.5
    assert np.array_equal(raw_decision, cal_decision)


def test_fit_requires_positive_threshold_and_enough_data():
    with pytest.raises(ValueError):
        Calibrator.fit(_val_errors(), 0.0)
    with pytest.raises(ValueError):
        Calibrator.fit(np.array([0.5]), 0.9)


def test_near_constant_validation_errors_do_not_blow_up():
    val = np.full(100, 0.5) + np.random.default_rng(3).normal(0, 1e-8, 100)
    cal = Calibrator.fit(val, 0.5)
    assert cal.params.scale >= 1e-6  # floored, not ~0
    assert np.isfinite(cal.transform_one(1000.0))
    assert cal.transform_one(1000.0) == pytest.approx(1.0, abs=1e-6)


def test_serialization_round_trip():
    cal = Calibrator.fit(_val_errors(), 0.7)
    d = cal.to_dict()
    assert set(d) == {"method", "center", "scale"}
    restored = Calibrator.from_dict(d)
    assert restored.transform_one(0.7) == cal.transform_one(0.7) == pytest.approx(0.5)
    assert restored.params == cal.params


def test_deterministic():
    val = _val_errors(seed=5)
    a = Calibrator.fit(val, 0.8)
    b = Calibrator.fit(val, 0.8)
    assert a.params == b.params


def test_config_fingerprint_changes_with_params():
    f1 = config_fingerprint({"threshold": 0.88})
    f2 = config_fingerprint({"threshold": 0.88})
    f3 = config_fingerprint({"threshold": 0.90})
    assert f1 == f2 and f1 != f3 and len(f1) == 8
