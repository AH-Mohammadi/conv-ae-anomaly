import numpy as np
import pytest

from anomaly_detector.preprocessing import Normalizer


def test_zscore_on_reference():
    rng = np.random.default_rng(0)
    ref = rng.normal(5, 2, size=(200, 3))
    z = Normalizer("zscore").fit(ref).transform(ref)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1, atol=1e-9)


def test_fit_uses_only_given_data():
    ref = np.random.default_rng(1).normal(size=(50, 2))
    n1 = Normalizer().fit(ref)
    n2 = Normalizer().fit(ref.copy())
    assert np.array_equal(n1.center_, n2.center_)
    assert np.allclose(n1.center_, ref.mean(axis=0))


def test_constant_sensor_no_nan():
    ref = np.ones((10, 2))
    out = Normalizer().fit(ref).transform(ref * 2)
    assert np.isfinite(out).all()


def test_minmax_reference_range():
    ref = np.array([[0.0], [5.0], [10.0]])
    out = Normalizer("minmax").fit(ref).transform(ref)
    assert out.min() == 0.0 and out.max() == 1.0


def test_errors():
    with pytest.raises(RuntimeError):
        Normalizer().transform(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Normalizer().fit(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        Normalizer("nope")
