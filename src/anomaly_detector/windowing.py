"""Reusable sliding-window utility."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(
    x: np.ndarray, window_size: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a (T, C) array into windows.

    Returns:
        windows: (N, window_size, C)
        end_idx: (N,) index in ``x`` of each window's last sample.

    If T < window_size, returns N = 0 (no partial windows are ever produced).
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (T, C), got shape {x.shape}")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1")

    n, c = x.shape
    if n < window_size:
        return (
            np.empty((0, window_size, c), dtype=x.dtype),
            np.empty(0, dtype=np.int64),
        )
    view = sliding_window_view(x, window_size, axis=0)  # (n-W+1, C, W)
    windows = np.ascontiguousarray(view[::stride].transpose(0, 2, 1))
    end_idx = np.arange(0, n - window_size + 1, stride, dtype=np.int64) + (
        window_size - 1
    )
    return windows, end_idx
