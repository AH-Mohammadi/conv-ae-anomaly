import numpy as np
import pytest

from anomaly_detector.windowing import make_windows


def test_stride_one():
    x = np.arange(10).reshape(10, 1)
    w, end = make_windows(x, 4, 1)
    assert w.shape == (7, 4, 1)
    assert w[0, :, 0].tolist() == [0, 1, 2, 3]
    assert end.tolist() == list(range(3, 10))


def test_stride_three():
    x = np.arange(10).reshape(10, 1)
    w, end = make_windows(x, 4, 3)
    assert w.shape == (3, 4, 1)
    assert w[-1, :, 0].tolist() == [6, 7, 8, 9]
    assert end.tolist() == [3, 6, 9]


def test_exact_and_insufficient_history():
    x = np.zeros((4, 2))
    assert make_windows(x, 4)[0].shape == (1, 4, 2)
    w, end = make_windows(x, 5)
    assert w.shape == (0, 5, 2) and end.shape == (0,)


def test_multichannel_layout():
    x = np.stack([np.arange(6), 10 * np.arange(6)], axis=1)
    w, _ = make_windows(x, 3)
    assert w[1].tolist() == [[1, 10], [2, 20], [3, 30]]


def test_invalid_params():
    with pytest.raises(ValueError):
        make_windows(np.zeros((5, 1)), 0)
    with pytest.raises(ValueError):
        make_windows(np.zeros((5, 1)), 2, 0)
    with pytest.raises(ValueError):
        make_windows(np.zeros(5), 2)
