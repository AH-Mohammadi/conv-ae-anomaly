import numpy as np
import pytest

from anomaly_detector.metrics import compute_metrics


def test_known_values():
    y_true = [1, 1, 0, 0, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 0]
    m = compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["f1"] == pytest.approx(4 / 6)
    assert m["far"] == pytest.approx(1 / 5)
    assert m["mar"] == pytest.approx(1 / 3)


def test_perfect():
    y = [0, 1, 1, 0]
    m = compute_metrics(y, y)
    assert (m["f1"], m["far"], m["mar"]) == (1.0, 0.0, 0.0)


def test_always_normal_and_always_anomaly():
    y_true = np.array([0, 0, 1, 1, 0])
    m0 = compute_metrics(y_true, np.zeros(5))
    assert (m0["f1"], m0["far"], m0["mar"]) == (0.0, 0.0, 1.0)
    m1 = compute_metrics(y_true, np.ones(5))
    assert (m1["far"], m1["mar"]) == (1.0, 0.0)


def test_undefined_is_none():
    m = compute_metrics([0, 0, 0], [0, 0, 0])
    assert m["mar"] is None and m["f1"] is None and m["far"] == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        compute_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError):
        compute_metrics([], [])


from anomaly_detector.metrics import roc_auc


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all ties


def test_roc_auc_undefined_and_invalid():
    assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1])
