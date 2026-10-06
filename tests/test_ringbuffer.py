import numpy as np
import pytest

from anomaly_detector.ringbuffer import RingBuffer


def test_not_full_before_capacity():
    rb = RingBuffer(4, 2)
    assert not rb.is_full
    for i in range(3):
        rb.push([i, i])
        assert not rb.is_full
    with pytest.raises(RuntimeError):
        rb.window()


def test_window_order_simple_and_wrapped():
    rb = RingBuffer(3, 1)
    for i in range(3):
        rb.push([i])
    assert rb.is_full
    assert rb.window()[:, 0].tolist() == [0, 1, 2]
    rb.push([3])  # overwrites 0
    assert rb.window()[:, 0].tolist() == [1, 2, 3]
    rb.push([4])
    assert rb.window()[:, 0].tolist() == [2, 3, 4]


def test_no_reallocation_after_push():
    rb = RingBuffer(5, 2)
    buf_id = id(rb._buf)
    for i in range(50):
        rb.push([i, -i])
    assert id(rb._buf) == buf_id  # same underlying array object throughout


def test_total_pushed_and_reset():
    rb = RingBuffer(3, 1)
    for i in range(7):
        rb.push([i])
    assert rb.total_pushed == 7
    rb.reset()
    assert not rb.is_full and rb.total_pushed == 0


def test_shape_validation():
    rb = RingBuffer(3, 2)
    with pytest.raises(ValueError):
        rb.push([1, 2, 3])
    with pytest.raises(ValueError):
        RingBuffer(0, 2)
